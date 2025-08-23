import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import shutil
import json
from tqdm import tqdm
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score

from Classification.arguments import get_args
from Classification.distiller import Distiller
from Classification.data_utils.image_datasets import ImageDistillDataset
from Classification.criterions import build_criterion

def prepare_dataset(args):
    data = {}
    
    # Define image transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if args.do_train:
        data["train"] = ImageDistillDataset(
            args, "train", transform=train_transform
        )
        print("Num of train data: {}".format(len(data["train"])))
        
        data["dev"] = ImageDistillDataset(
            args, "dev", transform=val_transform
        )
        print("Num of dev data: {}".format(len(data["dev"])))

        if os.path.exists(os.path.join(args.data_dir, "test")):
            data["test"] = ImageDistillDataset(
                args, "test", transform=val_transform
            )
            print("Num of test data: {}".format(len(data["test"])))

    elif args.do_eval:
        data["dev"] = ImageDistillDataset(
            args, "dev", transform=val_transform
        )
        print("Num of test data: {}".format(len(data["dev"])))
    else:
        raise ValueError("Do train and do eval must set one")
        
    return data

def finetune(args, model, optimizer, scheduler, dataset, device):
    print("Start Fine-tuning")
    start_time = time.time()
    
    criterion = build_criterion(args)
    
    train_loader = DataLoader(
        dataset['train'], 
        shuffle=True,
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=dataset["train"].collate
    )
    
    model_list = []
    
    for epoch in range(args.num_epochs):
        print("Start iterations of epoch {}".format(epoch + 1))
        model.train()
        
        total_loss = 0
        num_batches = 0
        
        data_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in data_iter:
            optimizer.zero_grad()
            
            input_batch, output_batch = batch
            dataset["train"].move_to_device([input_batch, output_batch], device)
            
            batch_dict = {
                "input_batch": input_batch,
                "output_batch": output_batch
            }
            
            # Simple logging output structure
            logging_output = {
                "loss": [],
                "nll_loss": [],
                "kd_loss": [],
                "accuracy": []
            }

            loss, _ = model(
                criterion,
                batch_dict,
                logging_output,
                loss_denom=1
            )
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Loss is NaN or Inf. Skipping this step.")
                continue

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            data_iter.set_postfix(loss=loss.item())
        
        if scheduler:
            scheduler.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Evaluate and save model
        if args.save_dir and (epoch + 1) % args.save_interval == 0:
            print("Evaluating before saving model...")
            eval_loss, eval_accu, eval_precision, eval_recall = evaluate(
                args, model.student_model, dataset["dev"], "dev", device
            )
            print(f'Eval - Loss: {eval_loss:.4f}, Accuracy: {eval_accu:.4f}, '
                  f'Precision: {eval_precision:.4f}, Recall: {eval_recall:.4f}')
            
            if "test" in dataset:
                test_loss, test_accu, test_precision, test_recall = evaluate(
                    args, model.student_model, dataset["test"], "test", device
                )
                print(f'Test - Loss: {test_loss:.4f}, Accuracy: {test_accu:.4f}, '
                      f'Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')
            
            ckpt_name = f"epoch{epoch+1}_loss{eval_loss:.4f}_acc{eval_accu:.4f}"
            save_dir_path = os.path.join(args.save_dir, ckpt_name)
            
            os.makedirs(save_dir_path, exist_ok=True)
            
            # Save student model
            print("Saving student model...")
            torch.save({
                'model_state_dict': model.student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': eval_loss,
                'accuracy': eval_accu
            }, os.path.join(save_dir_path, "student_model.pth"))
            
            # Save model config
            model_config = {
                'model_name': 'resnet18' if hasattr(args, 'distillation_mode') and args.distillation_mode == 'KD' else 'resnet50',
                'num_classes': getattr(args, 'num_classes', 1000),
                'distillation_mode': getattr(args, 'distillation_mode', 'SFT')
            }
            with open(os.path.join(save_dir_path, "model_config.json"), 'w') as f:
                json.dump(model_config, f)
            
            # Save projector if it exists
            if hasattr(model, "projectors"):
                print("Saving projector...")
                torch.save(
                    model.projectors.state_dict(), 
                    os.path.join(save_dir_path, "projector.pt")
                )
            
            # Keep only best N checkpoints
            model_list.append({"path": save_dir_path, "score": eval_accu})
            model_list = sorted(model_list, key=lambda x: x["score"], reverse=True)
            
            if len(model_list) > args.keep_best_n_checkpoints:
                removed_model = model_list.pop()
                shutil.rmtree(removed_model["path"])

            print(f"Model has been saved to {save_dir_path}")
            
    total_seconds = time.time() - start_time
    print("Done training in {:0>2}:{:0>2}:{:0>2}".format(
        int(total_seconds // 3600), 
        int(total_seconds % 3600 // 60), 
        int(total_seconds % 60)
    ))

@torch.no_grad
def evaluate(args, student_model, dataset, split, device):
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )

    student_model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    for input_batch, output_batch in tqdm(dataloader, desc=f"Evaluating {split}"):
        dataset.move_to_device([input_batch, output_batch], device)
        
        images = input_batch["images"]
        labels = output_batch["labels"]
        
        # Forward pass through student model
        logits = student_model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        total_loss += loss.item()

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    avg_loss = total_loss / len(dataloader)

    print(f"Evaluated: {split} | Loss: {avg_loss:.6f}, Accuracy: {accuracy:.6f}, "
          f"Precision: {precision:.6f}, Recall: {recall:.6f}")

    student_model.train()
    return avg_loss, accuracy, precision, recall

def main():
    args = get_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30)

    print("Initializing a distiller for knowledge distillation...")
    distiller = Distiller(args, device)
    dataset = prepare_dataset(args)
    
    if args.do_train:
        # Simple optimizer setup
        optimizer = Adam(distiller.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # Optional learning rate scheduler
        scheduler = None
        if hasattr(args, 'use_scheduler') and args.use_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        finetune(args, distiller, optimizer, scheduler, dataset, device)
       
    if args.do_eval:
        evaluate(args, distiller.student_model, dataset["dev"], "dev", device)
        
if __name__ == "__main__":
    main()