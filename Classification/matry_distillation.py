import time
import os

from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed
import shutil
import json
from tqdm import tqdm
import math
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    AutoModel,
)
from transformers.integrations import HfDeepSpeedConfig
from SentencePair.arguments import get_args
from SentencePair.matry_distiller import Distiller
from SentencePair.data_utils.distill_datasets import DistillDataset
from SentencePair.utils import (
    initialize,
    get_optimizer, 
    get_learning_rate_scheduler,
    print_rank, 
    log_rank,
    all_gather,
)
from SentencePair.criterions import build_criterion
# from rouge_metric import compute_metrics

torch.set_num_threads(4) # giới hạn số lượng thread torch sử dụng cho cpu

def prepare_dataset(args, distiller):
    data = {}
    if args.do_train:
        data["train"] = DistillDataset(
            args, "train", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of train data: {}".format(len(data["train"])))
        
        data["dev"] = DistillDataset(
            args, "dev", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of dev data: {}".format(len(data["dev"])))

        if os.path.exists(os.path.join(args.data_dir, "test.csv")):
            data["test"] = DistillDataset(
                args, "test", distiller.student_tokenizer,
                distiller.teacher_tokenizers
            )
            log_rank("Num of test data: {}".format(len(data["test"])))

    elif args.do_eval:
        data["test"] = DistillDataset(
            args, "test", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of test data: {}".format(len(data["test"])))
    else:
        raise ValueError("Do train and do eval must set one")
        
    return data

def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device):
    log_rank("Start Fine-tuning")
    start_time = time.time()

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        criterion = build_criterion(args)

    sampler = DistributedSampler(
        dataset["train"], 
        shuffle=True, 
        drop_last=True, 
        rank=dp_rank, 
        num_replicas=dp_world_size
    )
    train_loader = DataLoader(
        dataset['train'], 
        sampler=sampler, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=dataset["train"].collate
    )
    
    step = 0
    model_list = []
    logging_output = {
        "epoch": 0,
        "global_step": 0,
        "loss": [], 
        "nll_loss": [],
        "kd_loss": [],
        "accuracy": [],
        "micro_step_time": [],
        "step_time": []
    }
    
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        logging_output["epoch"] += 1
        
        log_rank("Start iterations of epoch {}".format(epoch + 1))
        model.train()
        print("Training mode?", model.student_model.training)  # True

        epoch_start_time = time.time()
        step = 0
        total_samples = 0
        total_time = 0.0

        data_iter = train_loader
        if dist.get_rank() == 0:
            data_iter = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

        for batch in data_iter:
            st_time = time.time()
            input_batch, output_batch = batch
            dataset["train"].move_to_device([input_batch, output_batch], device)

            loss, logging_output = model(
                criterion,
                {"input_batch": input_batch, "output_batch": output_batch},
                logging_output,
                loss_denom=1, #deepspeed support sync gradient, no need to calculate loss_denom
            )
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Loss is NaN or Inf. Skipping this step.")
                continue

            
            model.backward(loss)
            model.step()
            torch.cuda.synchronize()  # correctlyc compute time

            elapsed_time = time.time() - st_time
            num_samples = input_batch["input_ids"].size(0)
            total_samples += num_samples
            total_time += elapsed_time
            step += 1

            logging_output["global_step"] += 1
            logging_output["micro_step_time"].append(elapsed_time)
            logging_output["step_time"].append(elapsed_time)

            if dist.get_rank() == 0:
                data_iter.set_postfix(loss=loss.item())

        dist.barrier()

        if args.save_dir and (epoch + 1) % args.save_interval == 0 and dist.get_rank() == 0: #save_interval = 1 then save each epoch
            #eval_interval = 1 then evaluate each epoch
            log_rank("Evaluating before saving model...")
            eval_results = evaluate_mrl(args, tokenizer, model.module.student_model, dataset["dev"], "dev", device)
            
            # Use the best granularity for model selection (largest by default)
            eval_loss = eval_results['overall']['loss']
            eval_accu = eval_results['overall']['accuracy']
            
            if "test" in dataset: #evaluate for test, no affect
                _ = evaluate_mrl(args, tokenizer, model.module.student_model, dataset["test"], "test", device)
            
            # ckpt_name = "epoch{}_step{}_loss{:.4f}".format(epoch + 1, logging_output["global_step"], eval_loss)
            # we dont need to save student model checkpoint
            # save_dir_path = os.path.join(args.save_dir, ckpt_name)
            
            # os.makedirs(save_dir_path, exist_ok=True)
            # if not args.only_save_projector:
            #     log_rank("Saving tokenizer...")
            #     tokenizer.save_pretrained(save_dir_path)
            #     log_rank("Saving model...")
            #     model.module.student_model.save_pretrained(save_dir_path, safe_serialization=False)
                
            #     # Save classifier heads for MRL
            #     classifier_path = os.path.join(save_dir_path, "classifier_head.bin")
            #     if hasattr(model.module.student_model, 'mrl_classifier'):  # MRL model
            #         log_rank("Saving MRL classifier heads...")
            #         torch.save(model.module.student_model.mrl_classifier.state_dict(), classifier_path)
            #     elif hasattr(model.module.student_model, 'score'):  # Mistral model
            #         log_rank("Saving Mistral classifier head (score)...")
            #         torch.save(model.module.student_model.score.state_dict(), classifier_path)
            #     elif hasattr(model.module.student_model, 'classifier'):  # BERT model
            #         log_rank("Saving BERT classifier head (classifier)...")
            #         torch.save(model.module.student_model.classifier.state_dict(), classifier_path)
            #     else:
            #         log_rank("Warning: Could not identify classifier head structure, no classifier saved.")
                
            #     log_rank("Saving config")
            #     model.module.student_model.config.save_pretrained(save_dir_path)
                
            #     # Save MRL specific config
            #     mrl_config = {
            #         'nesting_list': getattr(args, 'mrl_nesting_list', [64, 128, 256, 512, 768]),
            #         'mrl_efficient': getattr(args, 'mrl_efficient', False),
            #         'mrl_relative_importance': getattr(args, 'mrl_relative_importance', None)
            #     }
            #     with open(os.path.join(save_dir_path, "mrl_config.json"), "w") as f:
            #         json.dump(mrl_config, f, indent=2)
                    
            # if hasattr(model.module, "projectors"):
            #     log_rank("Saving projector...")
            #     torch.save(
            #         model.module.projectors.state_dict(), 
            #         os.path.join(save_dir_path, "projector.pt")
            #     )
            
            # model_list.append({"path": save_dir_path, "score": eval_accu}) #store model list in term of eval_accuracy
            # model_list = sorted(model_list, key=lambda x: x["score"], reverse=True)  # Higher accuracy is better
            
            # if len(model_list) > args.keep_best_n_checkpoints:
            #     removed_model = model_list.pop()  # Remove the worst model
            #     shutil.rmtree(removed_model["path"])

            # log_rank(f"Model has been saved to {save_dir_path}")
        
            
    total_seconds = time.time() - start_time
    log_rank("Done training in {:0>2}:{:0>2}:{:0>2}".format(
        int(total_seconds // 3600), 
        int(total_seconds % 3600 // 60), 
        int(total_seconds % 60)
    ))

@torch.no_grad
def evaluate_mrl(args, tokenizer, student_model, dataset, split, device):
    """
    Evaluate model with Matryoshka representation learning across all granularities.
    Returns metrics for each nesting dimension and overall metrics.
    """
    if dist.get_rank() != 0:
        return None

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )

    student_model.eval()
    
    # Get nesting list from args or model
    nesting_list = getattr(args, 'mrl_nesting_list', [64, 128, 256, 512, 768])
    mrl_efficient = getattr(args, 'mrl_efficient', False)
    
    # Check if model has MRL capability
    is_mrl_model = hasattr(student_model, 'mrl_classifier') or hasattr(student_model, 'nesting_list')
    
    # Initialize metrics storage
    if is_mrl_model and not mrl_efficient:
        granularity_metrics = {dim: {'preds': [], 'labels': [], 'losses': []} for dim in nesting_list}
    else:
        granularity_metrics = {'full': {'preds': [], 'labels': [], 'losses': []}}
    
    all_labels = []
    
    for input_batch, output_batch in tqdm(dataloader, desc=f"Evaluating {split}"):
        dataset.move_to_device([input_batch, output_batch], device)
        labels = output_batch["labels"]
        all_labels.append(labels.cpu())
        
        outputs = student_model(
            input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            position_ids=input_batch.get("position_ids", None),
            labels=labels
        )
        
        if is_mrl_model and isinstance(outputs, dict) and 'logits' in outputs:
            # MRL model: handle multiple logits
            logits_dict = outputs['logits']
            total_loss = outputs.get('loss', 0)
            
            if not mrl_efficient:
                # Regular MRL: evaluate each granularity
                for dim in nesting_list:
                    key = f"logits_{dim}"
                    if key in logits_dict:
                        logits = logits_dict[key]
                        preds = logits.argmax(dim=-1)
                        
                        # Compute individual loss for this granularity
                        individual_loss = F.cross_entropy(logits, labels)
                        
                        granularity_metrics[dim]['preds'].append(preds.cpu())
                        granularity_metrics[dim]['labels'].append(labels.cpu())
                        granularity_metrics[dim]['losses'].append(individual_loss.item())
            else:
                # Efficient MRL: only one logits
                logits = list(logits_dict.values())[0]
                preds = logits.argmax(dim=-1)
                loss = F.cross_entropy(logits, labels)
                
                granularity_metrics['full']['preds'].append(preds.cpu())
                granularity_metrics['full']['labels'].append(labels.cpu())
                granularity_metrics['full']['losses'].append(loss.item())
        else:
            # Regular model: single logits
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            if isinstance(logits, dict):
                logits = list(logits.values())[0]
            
            preds = logits.argmax(dim=-1)
            loss = outputs.loss if hasattr(outputs, 'loss') else F.cross_entropy(logits, labels)
            
            granularity_metrics['full']['preds'].append(preds.cpu())
            granularity_metrics['full']['labels'].append(labels.cpu())
            granularity_metrics['full']['losses'].append(loss.item())

    # Concatenate all predictions and labels
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Compute metrics for each granularity
    results = {}
    overall_metrics = {'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
    num_granularities = 0
    
    for gran_key, gran_data in granularity_metrics.items():
        if gran_data['preds']:  # Check if we have data for this granularity
            all_preds = torch.cat(gran_data['preds'], dim=0).numpy()
            all_gran_labels = torch.cat(gran_data['labels'], dim=0).numpy()
            avg_loss = sum(gran_data['losses']) / len(gran_data['losses'])
            
            accuracy = (all_preds == all_gran_labels).mean()
            precision = precision_score(all_gran_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_gran_labels, all_preds, average='macro', zero_division=0)
            
            metrics = {
                'loss': round(avg_loss, 6),
                'accuracy': round(accuracy, 6),
                'precision': round(precision, 6),
                'recall': round(recall, 6),
                'sample_num': len(all_preds)
            }
            
            results[f'granularity_{gran_key}'] = metrics
            
            # Accumulate for overall metrics
            overall_metrics['loss'] += avg_loss
            overall_metrics['accuracy'] += accuracy
            overall_metrics['precision'] += precision
            overall_metrics['recall'] += recall
            num_granularities += 1
            
            print(f"Evaluated {split} - Granularity {gran_key}: {metrics}")
    
    # Compute overall metrics (average across granularities)
    if num_granularities > 0:
        for key in overall_metrics:
            overall_metrics[key] = round(overall_metrics[key] / num_granularities, 6)
        
        results['overall'] = overall_metrics
        print(f"Evaluated {split} - Overall: {overall_metrics}")
    
    # Print summary table
    if is_mrl_model and not mrl_efficient:
        print(f"\n{'='*60}")
        print(f"MRL Evaluation Summary for {split.upper()}")
        print(f"{'='*60}")
        print(f"{'Granularity':<12} {'Loss':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<8}")
        print(f"{'-'*60}")
        
        for dim in nesting_list:
            key = f'granularity_{dim}'
            if key in results:
                metrics = results[key]
                print(f"{dim:<12} {metrics['loss']:<8.4f} {metrics['accuracy']:<10.4f} "
                      f"{metrics['precision']:<10.4f} {metrics['recall']:<8.4f}")
        
        print(f"{'-'*60}")
        print(f"{'Overall':<12} {overall_metrics['loss']:<8.4f} {overall_metrics['accuracy']:<10.4f} "
              f"{overall_metrics['precision']:<10.4f} {overall_metrics['recall']:<8.4f}")
        print(f"{'='*60}\n")

    student_model.train()
    return results

@torch.no_grad
def evaluate(args, tokenizer, student_model, dataset, split, device):
    """
    Legacy evaluation function for backward compatibility.
    """
    results = evaluate_mrl(args, tokenizer, student_model, dataset, split, device)
    if results is None:
        return None, None, None, None
    
    overall = results.get('overall', {})
    return (overall.get('loss', 0.0), 
            overall.get('accuracy', 0.0), 
            overall.get('precision', 0.0), 
            overall.get('recall', 0.0))

def main():
    torch.backends.cudnn.enabled = False
    args = get_args()
    initialize(args)
    dp_world_size = dist.get_world_size()

    # save arguments
    if dist.get_rank() == 0:
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()

    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30)
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    print('user ds_config', ds_config)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["train_batch_size"] = args.batch_size * args.gradient_accumulation_steps * dp_world_size

    log_rank("Initializing a distiller for knowledge distillation...")
    distiller = Distiller(args, device)
    dataset = prepare_dataset(args, distiller)
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size))
        assert args.total_iters is not None or args.num_epochs is not None
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.num_epochs
        if args.num_epochs is None:
            args.num_epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)

        log_rank("Total_iters = {}".format(args.total_iters))
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    optimizer_grouped_parameters = get_optimizer(args, distiller.student_model)
    optimizer_grouped_parameters = distiller.add_optimizer_param_group(optimizer_grouped_parameters)

    lr_scheduler = get_learning_rate_scheduler(args, optimizer_grouped_parameters)

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=distiller,
        optimizer=optimizer_grouped_parameters,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    if args.do_train:
        finetune(args, distiller.student_tokenizer, model_engine, optimizer, lr_scheduler, dataset, device)
       
    if args.do_eval:
        evaluate_mrl(args, distiller.student_tokenizer, model_engine.module.student_model, dataset["test"], "test", device)
        
    
if __name__ == "__main__":
    main()
