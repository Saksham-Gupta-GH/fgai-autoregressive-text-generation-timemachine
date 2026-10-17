import os
import time
import math
import torch
import torch.nn as nn
import numpy as np


def calculate_accuracy(logits, targets):
    """Computes Top-1 classification accuracy."""
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


def train_one_epoch(model, dataloader, criterion, optimizer, device, clip_grad=1.0):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients in vanilla/BiRNN
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct_preds += (preds == targets).sum().item()
        total_preds += targets.size(0)
        
    epoch_loss = running_loss / total_preds
    epoch_acc = correct_preds / total_preds
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_preds = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs)
            loss = criterion(logits, targets)
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct_top1 += (preds == targets).sum().item()
            
            # Top-5 candidates
            k = min(5, logits.size(-1))
            top5 = torch.topk(logits, k=k, dim=-1).indices
            correct_top5 += (top5 == targets.unsqueeze(-1)).any(dim=-1).sum().item()
            
            total_preds += targets.size(0)
            
    val_loss = running_loss / total_preds
    val_acc = correct_top1 / total_preds
    val_top5 = correct_top5 / total_preds
    val_perplexity = math.exp(min(val_loss, 20))  # numerical guard against overflow
    return val_loss, val_acc, val_top5, val_perplexity


def train_model(model, train_loader, val_loader, test_loader=None, epochs=12, lr=0.001, device="cpu", checkpoint_path=None):
    """
    Trains a model under controlled, comparable conditions.
    Tracks loss, perplexity, top-1 accuracy, top-5 accuracy, best-epoch metrics,
    and evaluates on the unseen test set using the best checkpoint.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_top5_acc": [],
        "val_perplexity": [],
        "epoch_times": []
    }
    
    best_val_loss = float("inf")
    best_epoch = 1
    best_val_acc = 0.0
    best_val_top5 = 0.0
    best_ppl = float("inf")
    total_start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Training: {model.name} on {device.upper()}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*60}")
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_top5, val_ppl = evaluate(model, val_loader, criterion, device)
        
        epoch_duration = time.time() - epoch_start
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_top5_acc"].append(val_top5)
        history["val_perplexity"].append(val_ppl)
        history["epoch_times"].append(epoch_duration)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_acc = val_acc
            best_val_top5 = val_top5
            best_ppl = val_ppl
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
            
        print(f"Epoch {epoch:2d}/{epochs:2d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val PPL: {val_ppl:7.2f} | "
              f"Top-1: {val_acc*100:5.2f}% | "
              f"Top-5: {val_top5*100:5.2f}% | "
              f"Time: {epoch_duration:.2f}s")
        
    total_time = time.time() - total_start_time
    history["total_time"] = total_time
    history["avg_epoch_time"] = np.mean(history["epoch_times"])
    
    # Best-epoch metrics
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history["best_val_acc"] = best_val_acc
    history["best_val_top5_acc"] = best_val_top5
    history["best_perplexity"] = best_ppl
    
    # Final-epoch metrics
    history["final_val_loss"] = history["val_loss"][-1]
    history["final_val_acc"] = history["val_acc"][-1]
    history["final_val_top5_acc"] = history["val_top5_acc"][-1]
    history["final_perplexity"] = history["val_perplexity"][-1]
    history["param_count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Evaluate best model checkpoint on test set
    if test_loader is not None and checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        test_loss, test_acc, test_top5, test_ppl = evaluate(model, test_loader, criterion, device)
        history["test_loss"] = test_loss
        history["test_acc"] = test_acc
        history["test_top5_acc"] = test_top5
        history["test_perplexity"] = test_ppl
        print(f"Test Set Evaluation (Best Epoch {best_epoch}): Loss: {test_loss:.4f} | PPL: {test_ppl:.2f} | Top-1: {test_acc*100:.2f}% | Top-5: {test_top5*100:.2f}%")
        
    print(f"{model.name} completed in {total_time:.1f}s (Avg: {history['avg_epoch_time']:.2f}s/epoch).")
    return history
