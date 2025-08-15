import torch
import numpy as np
from tqdm import tqdm


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(train_loader)


def find_optimal_threshold(model, val_loader, device):
    """
    Find the optimal prediction threshold for F1-score on the validation set.
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels).float()

    best_f1 = 0
    best_threshold = 0.5
    
    # Search for the best global threshold on F1-micro
    for threshold in tqdm(np.arange(0.05, 0.95, 0.01), desc="Finding optimal threshold"):
        preds = (all_probs > threshold).float()
        
        tp_total = (preds * all_labels).sum()
        fp_total = (preds * (1 - all_labels)).sum()
        fn_total = ((1 - preds) * all_labels).sum()
        
        precision_micro = tp_total / (tp_total + fp_total + 1e-8)
        recall_micro = tp_total / (tp_total + fn_total + 1e-8)
        f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro + 1e-8)
        
        if f1_micro > best_f1:
            best_f1 = f1_micro
            best_threshold = threshold
            
    print(f"Best threshold found: {best_threshold:.4f} with F1-Micro: {best_f1:.4f}")
    return best_threshold


def evaluate(model, val_loader, criterion, device , predictions_threshold = 0.5):
    """
    Evaluate the model on the validation set with proper multi-label metrics.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Debug: print some prediction probabilities
            probs = torch.sigmoid(outputs)
            if len(all_preds) == 0:  # Only print for first batch
                print(f"Sample prediction probabilities: {probs[0][:10]}")
                print(f"Max probability in batch: {probs.max():.4f}")
                print(f"Mean probability in batch: {probs.mean():.4f}")

            # Get predictions - use a lower threshold for imbalanced data
            preds = (torch.sigmoid(outputs) > predictions_threshold).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu().float())
            all_outputs.append(probs.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_outputs = torch.cat(all_outputs, dim=0).numpy()

    # Now all calculations will work with float tensors
    
    # 1. Exact Match Accuracy (very strict - all classes must be correct)
    exact_match_accuracy = (all_preds == all_labels).all(dim=1).float().mean()
    
    # 2. Hamming Loss (fraction of wrong labels)
    hamming_loss = (all_preds != all_labels).float().mean()
    
    # 3. Per-class F1 scores
    tp = (all_preds * all_labels).sum(dim=0)
    fp = (all_preds * (1 - all_labels)).sum(dim=0)
    fn = ((1 - all_preds) * all_labels).sum(dim=0)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # 4. Macro and Micro F1
    f1_macro = f1_per_class.mean()
    
    # Micro F1 (aggregate then compute)
    tp_total = tp.sum()
    fp_total = fp.sum()
    fn_total = fn.sum()
    precision_micro = tp_total / (tp_total + fp_total + 1e-8)
    recall_micro = tp_total / (tp_total + fn_total + 1e-8)
    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro + 1e-8)
    
    # 5. Jaccard Index (Intersection over Union)
    intersection = (all_preds * all_labels).sum(dim=1)
    union = (all_preds + all_labels - all_preds * all_labels).sum(dim=1)
    jaccard = (intersection / (union + 1e-8)).mean()

    # 6. Positive Element-wise Accuracy (Recall on positive class)
    true_positives = (all_preds * all_labels).sum()
    actual_positives = all_labels.sum()
    positive_element_wise_accuracy = true_positives / (actual_positives + 1e-8)
    
    metrics = {
        'loss': running_loss / len(val_loader),
        'exact_match_accuracy': exact_match_accuracy.item(),
        'hamming_loss': hamming_loss.item(),
        'f1_macro': f1_macro.item(),
        'f1_micro': f1_micro.item(),
        'jaccard_index': jaccard.item(),
        'element_wise_accuracy': (all_preds == all_labels).float().mean().item(),
        'positive_element_wise_accuracy': positive_element_wise_accuracy.item(),
        'all_outputs': all_outputs
    }
    
    return metrics