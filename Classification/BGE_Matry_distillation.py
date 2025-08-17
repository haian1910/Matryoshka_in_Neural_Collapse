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
from Classification.arguments import get_args
from Classification.BGE_Matry_distiller import Distiller
from Classification.data_utils.distill_datasets import DistillDataset
from Classification.utils import (
    initialize,
    get_optimizer, 
    get_learning_rate_scheduler,
    print_rank, 
    log_rank,
    all_gather,
)
from Classification.criterions import build_criterion
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
            torch.cuda.synchronize()  # correctly compute time
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
            eval_results = evaluate_mrl_with_nc(args, tokenizer, model.module.student_model, dataset["dev"], "dev", device)
            
            # Use the best granularity for model selection (overall metrics)
            eval_loss = eval_results['overall']['loss']
            eval_accu = eval_results['overall']['accuracy']
            
            if "test" in dataset: #evaluate for test, no effect
                test_results = evaluate_mrl_with_nc(args, tokenizer, model.module.student_model, dataset["test"], "test", device)
            
    total_seconds = time.time() - start_time
    log_rank("Done training in {:0>2}:{:0>2}:{:0>2}".format(
        int(total_seconds // 3600), 
        int(total_seconds % 3600 // 60), 
        int(total_seconds % 60)
    ))

# Import neural collapse measurement functions
try:
    from neural_collapse.measure import (
        variability_cdnv, 
        mean_norms, 
        interference_stats,  # Changed from interference_grid
        simplex_etf_error
    )
    from neural_collapse.accumulate import MeanAccumulator, VarNormAccumulator
    NC_AVAILABLE = True
except ImportError:
    print("Warning: neural_collapse library not available. Neural collapse metrics will be skipped.")
    NC_AVAILABLE = False

def evaluate_mrl_with_nc(args, tokenizer, student_model, dataset, split, device):
    """
    Evaluate model with Matryoshka representation learning across all granularities,
    including Neural Collapse metrics (NC1, NC2) for each embedding dimension.
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
        # Initialize neural collapse accumulators for each dimension
        if NC_AVAILABLE:
            nc_accumulators = {
                dim: {
                    'mean_acc': MeanAccumulator(args.num_labels, dim, device, torch.float32),
                    'var_acc': VarNormAccumulator(args.num_labels, dim, device, torch.float32),
                    'embeddings': [],  # Store all embeddings for this dimension
                    'labels': []  # Store all labels
                } for dim in nesting_list
            }
        else:
            nc_accumulators = {}
    else:
        granularity_metrics = {'full': {'preds': [], 'labels': [], 'losses': []}}
        if NC_AVAILABLE:
            # For non-MRL or efficient MRL models
            hidden_size = getattr(student_model, 'hidden_size', 768)
            if hasattr(student_model, 'config') and hasattr(student_model.config, 'hidden_size'):
                hidden_size = student_model.config.hidden_size
            nc_accumulators = {
                'full': {
                    'mean_acc': MeanAccumulator(args.num_labels, hidden_size, device, torch.float32),
                    'var_acc': VarNormAccumulator(args.num_labels, hidden_size, device, torch.float32),
                    'embeddings': [],
                    'labels': []
                }
            }
        else:
            nc_accumulators = {}
    
    all_labels = []
    
    for input_batch, output_batch in tqdm(dataloader, desc=f"Evaluating {split}"):
        dataset.move_to_device([input_batch, output_batch], device)
        labels = output_batch["labels"]
        all_labels.append(labels.cpu())
        
        # Get model outputs
        outputs = student_model(
            input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            position_ids=input_batch.get("position_ids", None),
            labels=labels
        )
        
        # Extract embeddings for neural collapse analysis
        embeddings = None
        if is_mrl_model and isinstance(outputs, dict):
            # For MRL models, we need to get the pooled embeddings before classification
            if hasattr(student_model, 'bert'):
                # Get BERT outputs to extract embeddings
                bert_outputs = student_model.bert(
                    input_ids=input_batch["input_ids"],
                    attention_mask=input_batch["attention_mask"],
                    token_type_ids=input_batch.get("token_type_ids", None)
                )
                if hasattr(bert_outputs, 'pooler_output') and bert_outputs.pooler_output is not None:
                    embeddings = bert_outputs.pooler_output
                else:
                    embeddings = bert_outputs.last_hidden_state[:, 0]  # [CLS] token
        else:
            # For non-MRL models, try to extract hidden states
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                embeddings = outputs.hidden_states[-1][:, 0]  # Last layer [CLS] token
            elif hasattr(student_model, 'bert'):
                # Fallback: get BERT embeddings directly
                bert_outputs = student_model.bert(
                    input_ids=input_batch["input_ids"],
                    attention_mask=input_batch["attention_mask"],
                    token_type_ids=input_batch.get("token_type_ids", None)
                )
                if hasattr(bert_outputs, 'pooler_output') and bert_outputs.pooler_output is not None:
                    embeddings = bert_outputs.pooler_output
                else:
                    embeddings = bert_outputs.last_hidden_state[:, 0]
        
        # Process logits and collect neural collapse data
        if is_mrl_model and isinstance(outputs, dict) and 'logits' in outputs:
            # MRL model: handle multiple logits
            logits_dict = outputs['logits']
            
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
                        
                        # Collect neural collapse data for this dimension
                        if NC_AVAILABLE and embeddings is not None:
                            # Truncate embeddings to current dimension
                            truncated_embeddings = embeddings[:, :dim].detach()
                            
                            # Store the batch of embeddings and labels
                            nc_accumulators[dim]['embeddings'].append(truncated_embeddings)
                            nc_accumulators[dim]['labels'].append(labels)
            else:
                # Efficient MRL: only one logits
                logits = list(logits_dict.values())[0]
                preds = logits.argmax(dim=-1)
                loss = F.cross_entropy(logits, labels)
                
                granularity_metrics['full']['preds'].append(preds.cpu())
                granularity_metrics['full']['labels'].append(labels.cpu())
                granularity_metrics['full']['losses'].append(loss.item())
                
                # Collect neural collapse data
                if NC_AVAILABLE and embeddings is not None:
                    embeddings_detached = embeddings.detach()
                    nc_accumulators['full']['embeddings'].append(embeddings_detached)
                    nc_accumulators['full']['labels'].append(labels)
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
            
            # Collect neural collapse data
            if NC_AVAILABLE and embeddings is not None:
                embeddings_detached = embeddings.detach()
                nc_accumulators['full']['embeddings'].append(embeddings_detached)
                nc_accumulators['full']['labels'].append(labels)

    # Concatenate all predictions and labels
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Compute standard metrics for each granularity
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
    
    # Compute Neural Collapse metrics for each granularity
    if NC_AVAILABLE and nc_accumulators:
        print(f"\nComputing Neural Collapse metrics for {split}...")
        
        for gran_key, nc_data in nc_accumulators.items():
            if nc_data['embeddings']:  # Check if we have data
                print(f"Computing NC metrics for granularity {gran_key}...")
                
                # Concatenate all embeddings and labels for this granularity
                all_embeddings = torch.cat(nc_data['embeddings'], dim=0)  # (N, D)
                all_nc_labels = torch.cat(nc_data['labels'], dim=0)  # (N,)
                
                # First pass: accumulate means
                print(f"  Accumulating means for {all_embeddings.shape[0]} samples...")
                for emb_batch, lbl_batch in zip(nc_data['embeddings'], nc_data['labels']):
                    # Process the batch as-is (already 2D: batch_size x dim)
                    nc_data['mean_acc'].accumulate(emb_batch, lbl_batch)
                
                # Compute class means (M) and global mean (mG)
                M, mG = nc_data['mean_acc'].compute()
                print(f"  Computed means: M shape {M.shape}, mG shape {mG.shape}")
                
                # Filter out classes with too few samples if needed
                min_samples_per_class = getattr(args, 'min_samples_per_class', 10)
                valid_classes = nc_data['mean_acc'].ns_samples >= min_samples_per_class
                num_valid_classes = valid_classes.sum().item()
                
                if num_valid_classes < args.num_labels:
                    print(f"  Warning: Only {num_valid_classes}/{args.num_labels} classes have >= {min_samples_per_class} samples")
                    
                    if num_valid_classes < 2:
                        print(f"  Skipping NC metrics: need at least 2 valid classes, got {num_valid_classes}")
                        nc_metrics = {
                            'nc1_cdnv': None,
                            'nc2_norms_mean': None,
                            'nc2_norms_var': None,
                            'nc2_etf_error': None,
                            'nc2_interference_mean': None,
                            'nc2_interference_var': None,
                            'nc_valid_classes': num_valid_classes
                        }
                        if f'granularity_{gran_key}' in results:
                            results[f'granularity_{gran_key}'].update(nc_metrics)
                        continue
                    
                    # Filter to valid classes
                    M = M[valid_classes]
                    valid_class_indices = torch.where(valid_classes)[0]
                    
                    # Filter embeddings and labels to only include valid classes
                    mask = torch.zeros(all_nc_labels.shape[0], dtype=torch.bool)
                    for valid_idx in valid_class_indices:
                        mask |= (all_nc_labels == valid_idx)
                    
                    all_embeddings = all_embeddings[mask]
                    all_nc_labels = all_nc_labels[mask]
                    
                    # Remap labels to 0, 1, 2, ... for valid classes
                    label_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(valid_class_indices)}
                    remapped_labels = torch.tensor([label_mapping[label.item()] for label in all_nc_labels], device=device)
                    
                    # Create new filtered data lists for variance computation
                    filtered_embeddings = []
                    filtered_labels = []
                    for emb_batch, lbl_batch in zip(nc_data['embeddings'], nc_data['labels']):
                        batch_mask = torch.zeros(lbl_batch.shape[0], dtype=torch.bool)
                        for valid_idx in valid_class_indices:
                            batch_mask |= (lbl_batch == valid_idx)
                        if batch_mask.any():
                            filtered_emb = emb_batch[batch_mask]
                            filtered_lbl = lbl_batch[batch_mask]
                            # Remap labels
                            remapped_batch_labels = torch.tensor([label_mapping[l.item()] for l in filtered_lbl], device=device)
                            filtered_embeddings.append(filtered_emb)
                            filtered_labels.append(remapped_batch_labels)
                    
                    # Reset and recompute variance with filtered data
                    nc_data['var_acc'].n_classes = num_valid_classes
                    nc_data['var_acc'].totals = torch.zeros(nc_data['var_acc'].n_classes, dtype=torch.float32).to(device)
                    nc_data['var_acc'].ns_samples = torch.zeros(nc_data['var_acc'].n_classes, dtype=torch.int32).to(device)
                else:
                    # All classes are valid, use original data
                    filtered_embeddings = nc_data['embeddings']
                    filtered_labels = nc_data['labels']
                
                try:
                    # Second pass: accumulate variances
                    print(f"  Accumulating variances...")
                    for emb_batch, lbl_batch in zip(filtered_embeddings, filtered_labels):
                        nc_data['var_acc'].accumulate(emb_batch, lbl_batch, M)
                    
                    # Compute variance norms
                    V = nc_data['var_acc'].compute()
                    print(f"  Computed variance norms: V shape {V.shape}")
                    
                    # NC1: Within-class variability (CDNV)
                    cdnv = variability_cdnv(V, M, 2, getattr(args, 'tile_size', 1024))
                    
                    # NC2: Mean norms (Equinorm property)
                    norms_stats = mean_norms(M, mG)
                    norms_mean = norms_stats.mean().item()
                    norms_var = norms_stats.var().item()
                    
                    # NC2: Interference/ETF error (Simplex ETF property)
                    # etf_error = simplex_etf_error(M, mG)
                    # interference_stats = interference_grid(M, mG)
                    # interference_mean = interference_stats.mean().item()
                    # interference_var = interference_stats.var().item()

                    # NC2: Interference/ETF error (Simplex ETF property)
                    etf_error = simplex_etf_error(M, mG)
                    # Use the new interference_stats function instead of interference_grid
                    interference_mean, interference_var = interference_stats(M, mG)
                    
                    # Add NC metrics to results
                    # nc_metrics = {
                    #     'nc1_cdnv': round(cdnv, 6) if not torch.isnan(torch.tensor(cdnv)) else None,
                    #     'nc2_norms_mean': round(norms_mean, 6),
                    #     'nc2_norms_var': round(norms_var, 6),
                    #     'nc2_etf_error': round(etf_error, 6) if not torch.isnan(torch.tensor(etf_error)) else None,
                    #     'nc2_interference_mean': round(interference_mean, 6),
                    #     'nc2_interference_var': round(interference_var, 6),
                    #     'nc_valid_classes': num_valid_classes
                    # }

                    nc_metrics = {
                        'nc1_cdnv': round(cdnv, 6) if not torch.isnan(torch.tensor(cdnv)) else None,
                        'nc2_norms_mean': round(norms_mean, 6),
                        'nc2_norms_var': round(norms_var, 6),
                        'nc2_etf_error': round(etf_error, 6) if not torch.isnan(torch.tensor(etf_error)) else None,
                        'nc2_interference_mean': round(interference_mean, 6),
                        'nc2_interference_var': round(interference_var, 6),
                        'nc_valid_classes': num_valid_classes
                    }
                    
                    # Add to results
                    if f'granularity_{gran_key}' in results:
                        results[f'granularity_{gran_key}'].update(nc_metrics)
                    else:
                        results[f'granularity_{gran_key}'] = nc_metrics
                    
                    print(f"  NC1 (CDNV): {nc_metrics['nc1_cdnv']}")
                    print(f"  NC2 (Norms): mean={nc_metrics['nc2_norms_mean']}, var={nc_metrics['nc2_norms_var']}")
                    print(f"  NC2 (ETF error): {nc_metrics['nc2_etf_error']}")
                    print(f"  NC2 (Interference): mean={nc_metrics['nc2_interference_mean']}, var={nc_metrics['nc2_interference_var']}")
                    
                except Exception as e:
                    import traceback
                    print(f"  Error computing NC metrics for {gran_key}: {e}")
                    print(f"  Full traceback: {traceback.format_exc()}")
                    nc_metrics = {
                        'nc1_cdnv': None,
                        'nc2_norms_mean': None,
                        'nc2_norms_var': None,
                        'nc2_etf_error': None,
                        'nc2_interference_mean': None,
                        'nc2_interference_var': None,
                        'nc_valid_classes': 0
                    }
                    if f'granularity_{gran_key}' in results:
                        results[f'granularity_{gran_key}'].update(nc_metrics)
    
    # Compute overall metrics (average across granularities)
    if num_granularities > 0:
        for key in overall_metrics:
            overall_metrics[key] = round(overall_metrics[key] / num_granularities, 6)
        
        results['overall'] = overall_metrics
        print(f"Evaluated {split} - Overall: {overall_metrics}")
    
    # Print comprehensive summary table
    if is_mrl_model and not mrl_efficient:
        print(f"\n{'='*120}")
        print(f"MRL + Neural Collapse Evaluation Summary for {split.upper()}")
        print(f"{'='*120}")
        header = f"{'Dim':<6} {'Loss':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8}"
        if NC_AVAILABLE:
            header += f" {'NC1(CDNV)':<10} {'NC2(Norm)':<10} {'NC2(ETF)':<10} {'NC2(Interf)':<12} {'ValidCls':<8}"
        print(header)
        print(f"{'-'*120}")
        
        for dim in nesting_list:
            key = f'granularity_{dim}'
            if key in results:
                metrics = results[key]
                row = f"{dim:<6} {metrics['loss']:<8.4f} {metrics['accuracy']:<8.4f} " \
                      f"{metrics['precision']:<8.4f} {metrics['recall']:<8.4f}"
                
                if NC_AVAILABLE:
                    cdnv = metrics.get('nc1_cdnv', 'N/A')
                    norms = metrics.get('nc2_norms_mean', 'N/A')
                    etf = metrics.get('nc2_etf_error', 'N/A')
                    interf = metrics.get('nc2_interference_mean', 'N/A')
                    valid_cls = metrics.get('nc_valid_classes', 'N/A')
                    
                    cdnv_str = f"{cdnv:.4f}" if isinstance(cdnv, (int, float)) and cdnv is not None else str(cdnv)
                    norms_str = f"{norms:.4f}" if isinstance(norms, (int, float)) and norms is not None else str(norms)
                    etf_str = f"{etf:.4f}" if isinstance(etf, (int, float)) and etf is not None else str(etf)
                    interf_str = f"{interf:.4f}" if isinstance(interf, (int, float)) and interf is not None else str(interf)
                    
                    row += f" {cdnv_str:<10} {norms_str:<10} {etf_str:<10} {interf_str:<12} {valid_cls:<8}"
                
                print(row)
        
        print(f"{'-'*120}")
        row = f"{'Avg':<6} {overall_metrics['loss']:<8.4f} {overall_metrics['accuracy']:<8.4f} " \
              f"{overall_metrics['precision']:<8.4f} {overall_metrics['recall']:<8.4f}"
        if NC_AVAILABLE:
            row += f" {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<8}"
        print(row)
        print(f"{'='*120}\n")

    student_model.train()
    return results

@torch.no_grad
def evaluate(args, tokenizer, student_model, dataset, split, device):
    """
    Legacy evaluation function for backward compatibility.
    Now calls the enhanced MRL evaluation with Neural Collapse metrics.
    """
    results = evaluate_mrl_with_nc(args, tokenizer, student_model, dataset, split, device)
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
        evaluate_mrl_with_nc(args, distiller.student_tokenizer, model_engine.module.student_model, dataset["test"], "test", device)
        
    
if __name__ == "__main__":
    main()
