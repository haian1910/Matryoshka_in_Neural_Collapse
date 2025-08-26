import time
import os

from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support
from Classification.criterions.full_nc import FULL_NC

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
from Classification.QWEN_Matry_distiller import Distiller
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

# Replace the compute_teacher_targets function in matry_distillation.py with this fixed version:

def compute_teacher_targets(args, distiller, dataset, device):
    """
    Compute teacher class means and gram matrix over the full dataset.
    This should be called after teacher training is complete and before distillation begins.
    
    Args:
        args: Training arguments
        distiller: Distiller object containing teacher and student models
        dataset: Dataset dictionary containing train/dev/test splits
        device: Device to run computations on
        
    Returns:
        teacher_class_means: Tensor of shape [num_classes, teacher_hidden_size]
        teacher_gram: Normalized Gram matrix of shape [num_classes, num_classes]
    """
    if dist.get_rank() != 0:
        # Only compute on rank 0, then broadcast
        return None, None
    
    log_rank("Computing teacher targets over full training dataset...")
    
    # Use training dataset for computing teacher targets
    train_dataset = dataset["train"]
    
    # Create dataloader for teacher target computation
    # Use smaller batch size to handle memory constraints
    target_batch_size = min(args.eval_batch_size, 32)
    
    dataloader = DataLoader(
        train_dataset,
        shuffle=False,  # Don't shuffle for consistent computation
        batch_size=target_batch_size,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate
    )
    
    # Set teacher model to eval mode and move to device
    teacher_model = distiller.teacher_model
    teacher_model.eval()
    teacher_model = teacher_model.to(device)  # Ensure teacher model is on correct device
    
    # Get teacher hidden size
    teacher_hidden_size = getattr(args, 'teacher_hidden_size', 2048)
    if hasattr(teacher_model, 'config') and hasattr(teacher_model.config, 'hidden_size'):
        teacher_hidden_size = teacher_model.config.hidden_size
    elif hasattr(teacher_model, 'hidden_size'):
        teacher_hidden_size = teacher_model.hidden_size
    
    num_classes = args.num_labels
    
    # Initialize accumulators for computing class means
    class_embeddings_sum = torch.zeros(num_classes, teacher_hidden_size, device=device, dtype=torch.float32)
    class_counts = torch.zeros(num_classes, device=device, dtype=torch.long)
    
    log_rank(f"Processing {len(dataloader)} batches for teacher target computation...")
    
    # First pass: accumulate embeddings for each class
    with torch.no_grad():
        for batch_idx, (input_batch, output_batch) in enumerate(tqdm(dataloader, desc="Computing teacher targets")):
            # Move batch to device
            train_dataset.move_to_device([input_batch, output_batch], device)
            labels = output_batch["labels"]  # [batch_size]
            
            # Ensure teacher input tensors are on the correct device
            teacher_input_ids = input_batch["teacher_input_ids"].to(device)
            teacher_attention_mask = input_batch["teacher_attention_mask"].to(device)
            
            # Get teacher outputs with hidden states
            teacher_outputs = teacher_model(
                teacher_input_ids,
                attention_mask=teacher_attention_mask,
                output_hidden_states=True
            )
            
            # Extract teacher hidden states - Handle different model architectures
            if hasattr(teacher_outputs, 'hidden_states') and teacher_outputs.hidden_states is not None:
                # For LLM2Vec model loaded with AutoModelForSequenceClassification
                teacher_hidden = teacher_outputs.hidden_states[-1]  # Last layer hidden states
            elif isinstance(teacher_outputs, dict) and 'hidden_states' in teacher_outputs:
                # If teacher_outputs is a dict with hidden_states key
                teacher_hidden = teacher_outputs['hidden_states'][-1]
            else:
                raise ValueError("Cannot extract teacher hidden states")
            
            # Extract CLS token representation from teacher
            if teacher_hidden.dim() == 3:  # [batch_size, sequence_length, hidden_size]
                teacher_embeddings = teacher_hidden.mean(dim=1)  # Take mean across sequence length
            elif teacher_hidden.dim() == 2:  # [batch_size, hidden_size] - already CLS representation
                teacher_embeddings = teacher_hidden
            else:
                raise ValueError(f"Unexpected dimension for teacher_hidden: {teacher_hidden.shape}")
            # Ensure embeddings are float32 for numerical stability
            teacher_embeddings = teacher_embeddings.float()
            
            # Accumulate embeddings for each class
            for class_idx in range(num_classes):
                mask = (labels == class_idx)
                if mask.sum() > 0:
                    class_embeddings_sum[class_idx] += teacher_embeddings[mask].sum(dim=0)
                    class_counts[class_idx] += mask.sum()
            
            if (batch_idx + 1) % 100 == 0:
                log_rank(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Compute class means
    # Avoid division by zero for classes with no samples
    class_means = torch.zeros_like(class_embeddings_sum)
    for class_idx in range(num_classes):
        if class_counts[class_idx] > 0:
            class_means[class_idx] = class_embeddings_sum[class_idx] / class_counts[class_idx]
        else:
            log_rank(f"Warning: Class {class_idx} has no samples in the training dataset")
    
    # Remove classes with no samples from consideration
    valid_classes = class_counts > 0
    valid_class_means = class_means[valid_classes]
    num_valid_classes = valid_classes.sum().item()
    
    log_rank(f"Computed class means for {num_valid_classes}/{num_classes} classes")
    log_rank(f"Class sample counts: {class_counts.tolist()}")
    
    if num_valid_classes == 0:
        raise ValueError("No valid classes found in the training dataset")
    
    # Compute Gram matrix: G = M @ M^T where M is the class means matrix
    # Shape: [num_valid_classes, num_valid_classes]
    teacher_gram = torch.mm(valid_class_means, valid_class_means.t())
    
    # Normalize Gram matrix by Frobenius norm for numerical stability
    gram_norm = torch.norm(teacher_gram, p='fro')
    if gram_norm > 1e-8:
        teacher_gram = teacher_gram / gram_norm
    else:
        log_rank("Warning: Teacher Gram matrix has very small norm, using unnormalized matrix")
    
    # If we filtered out some classes, we need to expand back to full size with zeros
    if num_valid_classes < num_classes:
        full_class_means = torch.zeros(num_classes, teacher_hidden_size, device=device, dtype=torch.float32)
        full_class_means[valid_classes] = valid_class_means
        
        full_gram = torch.zeros(num_classes, num_classes, device=device, dtype=torch.float32)
        valid_indices = torch.where(valid_classes)[0]
        for i, idx_i in enumerate(valid_indices):
            for j, idx_j in enumerate(valid_indices):
                full_gram[idx_i, idx_j] = teacher_gram[i, j]
        
        teacher_class_means = full_class_means
        teacher_gram = full_gram
    else:
        teacher_class_means = valid_class_means
    
    log_rank(f"Teacher targets computed:")
    log_rank(f"  Class means shape: {teacher_class_means.shape}")
    log_rank(f"  Gram matrix shape: {teacher_gram.shape}")
    log_rank(f"  Gram matrix norm: {torch.norm(teacher_gram, p='fro').item():.6f}")
    
    # Save teacher targets to disk for future use
    if args.save_dir:
        teacher_targets_path = os.path.join(args.save_dir, "teacher_targets.pt")
        torch.save({
            'teacher_class_means': teacher_class_means.cpu(),
            'teacher_gram': teacher_gram.cpu(),
            'class_counts': class_counts.cpu(),
            'valid_classes': valid_classes.cpu(),
            'teacher_hidden_size': teacher_hidden_size,
            'num_classes': num_classes
        }, teacher_targets_path)
        log_rank(f"Saved teacher targets to {teacher_targets_path}")
    
    return teacher_class_means, teacher_gram

def load_or_compute_teacher_targets(args, distiller, dataset, device):
    """
    Load teacher targets from disk if available, otherwise compute them.
    
    Args:
        args: Training arguments
        distiller: Distiller object
        dataset: Dataset dictionary
        device: Device to run computations on
        
    Returns:
        teacher_class_means: Tensor of shape [num_classes, teacher_hidden_size]
        teacher_gram: Normalized Gram matrix of shape [num_classes, num_classes]
    """
    teacher_targets_path = os.path.join(args.save_dir, "teacher_targets.pt") if args.save_dir else None
    
    # Check if we should force recomputation
    force_recompute = getattr(args, 'recompute_teacher_targets', False)
    
    if (not force_recompute and teacher_targets_path and 
        os.path.exists(teacher_targets_path) and dist.get_rank() == 0):
        
        log_rank("Loading pre-computed teacher targets...")
        try:
            saved_targets = torch.load(teacher_targets_path, map_location=device)
            teacher_class_means = saved_targets['teacher_class_means'].to(device)
            teacher_gram = saved_targets['teacher_gram'].to(device)
            
            log_rank(f"Loaded teacher targets:")
            log_rank(f"  Class means shape: {teacher_class_means.shape}")
            log_rank(f"  Gram matrix shape: {teacher_gram.shape}")
            
            return teacher_class_means, teacher_gram
        except Exception as e:
            log_rank(f"Failed to load teacher targets: {e}")
            log_rank("Computing teacher targets from scratch...")
    
    # Compute teacher targets
    teacher_class_means, teacher_gram = compute_teacher_targets(args, distiller, dataset, device)
    
    # Broadcast to all ranks if using distributed training
    if dist.get_world_size() > 1:
        # Broadcast from rank 0 to all other ranks
        if dist.get_rank() == 0:
            # Send tensors to all ranks
            objects = [teacher_class_means, teacher_gram]
        else:
            objects = [None, None]
        
        dist.broadcast_object_list(objects, src=0)
        teacher_class_means, teacher_gram = objects
        
        # Move to correct device
        if teacher_class_means is not None:
            teacher_class_means = teacher_class_means.to(device)
            teacher_gram = teacher_gram.to(device)
    
    return teacher_class_means, teacher_gram

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
        
        # NEW CODE: Ensure FULL_NC criterion has teacher targets set
        if isinstance(criterion, FULL_NC):
            # Check if teacher targets are already set in the model's criterion
            model_criterion = None
            if hasattr(model.module, 'criterion'):
                model_criterion = model.module.criterion
            elif hasattr(model.module, 'loss_function'):
                model_criterion = model.module.loss_function
            
            # If model has FULL_NC criterion and our criterion has targets, copy them
            if (isinstance(model_criterion, FULL_NC) and 
                hasattr(criterion, 'teacher_class_means') and 
                hasattr(criterion, 'teacher_gram')):
                model_criterion.set_teacher_targets(
                    criterion.teacher_class_means, 
                    criterion.teacher_gram
                )
                log_rank("Teacher targets copied to model's FULL_NC criterion")
        
    # Rest of the finetune function remains the same...
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
                
                all_embeddings = torch.cat(nc_data['embeddings'], dim=0)  # (N, D)
                all_nc_labels = torch.cat(nc_data['labels'], dim=0)  # (N,)
                
                # Ensure all_nc_labels is on the same device as other tensors
                all_nc_labels = all_nc_labels.to(device)
                
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
                    mask = torch.zeros(all_nc_labels.shape[0], dtype=torch.bool, device=device)
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
                        # Ensure batch labels are on correct device
                        lbl_batch = lbl_batch.to(device)
                        batch_mask = torch.zeros(lbl_batch.shape[0], dtype=torch.bool, device=device)
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
                    nc_data['var_acc'].totals = torch.zeros(nc_data['var_acc'].n_classes, dtype=torch.float32, device=device)
                    nc_data['var_acc'].ns_samples = torch.zeros(nc_data['var_acc'].n_classes, dtype=torch.int32, device=device)
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


# Replace the relevant parts in main() function:

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
    
    # Pre-compute teacher targets if using FULL_NC criterion - DO THIS BEFORE DEEPSPEED INITIALIZATION
    teacher_class_means = None
    teacher_gram = None
    criterion = build_criterion(args)
    if isinstance(criterion, FULL_NC) and args.do_train:
        log_rank("FULL_NC criterion detected. Pre-computing teacher targets...")
        
        # Load or compute teacher targets over the full dataset
        teacher_class_means, teacher_gram = load_or_compute_teacher_targets(
            args, distiller, dataset, device
        )
        
        if teacher_class_means is None or teacher_gram is None:
            raise RuntimeError("Failed to compute or load teacher targets for FULL_NC")
        
        log_rank("Teacher targets successfully computed")
    
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
    
    # IMPORTANT: Set teacher targets AFTER deepspeed initialization
    if isinstance(criterion, FULL_NC) and args.do_train and teacher_class_means is not None:
        log_rank("Setting teacher targets in model engine...")
        
        # The distiller inside the model_engine will have the actual criterion used during training
        # We need to find and set the teacher targets in that criterion
        if hasattr(model_engine.module, 'forward'):
            # Try to access the criterion that will actually be used
            # This is a bit hacky but necessary due to deepspeed wrapping
            
            # Method 1: Set in any FULL_NC criterion we can find in the distiller
            def set_teacher_targets_recursive(obj, class_means, gram):
                if isinstance(obj, FULL_NC):
                    obj.set_teacher_targets(class_means, gram)
                    log_rank("Set teacher targets in FULL_NC criterion")
                    return True
                elif hasattr(obj, '__dict__'):
                    for attr_name, attr_value in obj.__dict__.items():
                        if set_teacher_targets_recursive(attr_value, class_means, gram):
                            return True
                return False
            
            success = set_teacher_targets_recursive(model_engine.module, teacher_class_means, teacher_gram)
            
            if not success:
                log_rank("Warning: Could not find FULL_NC criterion in model engine to set teacher targets")
        
        # Method 2: Also try to pass the teacher targets through the args for the criterion to access
        if not hasattr(args, '_teacher_class_means'):
            args._teacher_class_means = teacher_class_means
            args._teacher_gram = teacher_gram
            log_rank("Stored teacher targets in args as fallback")
    
    if args.do_train:
        finetune(args, distiller.student_tokenizer, model_engine, optimizer, lr_scheduler, dataset, device)
       
    if args.do_eval:
        evaluate_mrl_with_nc(args, distiller.student_tokenizer, model_engine.module.student_model, dataset["test"], "test", device)
        
if __name__ == "__main__":
    main()
