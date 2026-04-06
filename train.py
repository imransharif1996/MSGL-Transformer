"""
MSGL-Transformer: Training Script
------------------------------------------
Paper: MSGL-Transformer: A Multi-Scale Global-Local Transformer
       for Rodent Social Behavior Recognition
Authors: Muhammad Imran Sharif, Doina Caragea
Institution: Kansas State University

---------------------------------------------------------------
FOR CalMS21 USERS (this script):
---------------------------------------------------------------
- 4 behavior classes: Attack, Investigation, Mount, Other
- Input dimension: 28 (7 keypoints x 2 mice x 2 coordinates)
- Download: https://data.caltech.edu/records/s0vdx-0k302
- Run:
    python train.py --data_dir /path/to/calms21 --output_dir ./results

---------------------------------------------------------------
FOR RatSI USERS:
---------------------------------------------------------------
- 5 behavior classes: Approaching, Following, Moving Away,
  Social Nose Contact, Solitary
- Input dimension: 12 (6 keypoints x 2 coordinates)
- Download: https://mlorbach.gitlab.io/datasets/
- To adapt this script for RatSI, change:
    input_dim   = 12
    num_classes = 5
    class_names = ['Approaching', 'Following', 'Moving away',
                   'Social Nose Contact', 'Solitary']
  And load data using pandas from Excel files instead of JSON.
---------------------------------------------------------------
"""

import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (precision_recall_fscore_support,
                             confusion_matrix, accuracy_score,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from model import MSGLTransformer
from dataset import CalMS21Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='MSGL-Transformer on CalMS21')
    parser.add_argument('--data_dir',   type=str, required=True, help='Path to CalMS21 task1 folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--seq_len',    type=int,   default=35)
    parser.add_argument('--stride',     type=int,   default=1)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--patience',   type=int,   default=25)
    parser.add_argument('--d_model',    type=int,   default=64)
    parser.add_argument('--num_heads',  type=int,   default=4)
    parser.add_argument('--num_layers', type=int,   default=2)
    parser.add_argument('--dff',        type=int,   default=128)
    parser.add_argument('--dropout',    type=float, default=0.2)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--seed',       type=int,   default=42)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * sequences.size(0)
        correct    += outputs.max(1)[1].eq(labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs    = model(sequences)
            loss       = criterion(outputs, labels)
            total_loss += loss.item() * sequences.size(0)
            probs       = torch.softmax(outputs, dim=1)
            preds       = outputs.max(1)[1]
            correct    += preds.eq(labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    class_names = ['Attack', 'Investigation', 'Mount', 'Other']
    num_classes = 4
    input_dim   = 28

    # Load data
    train_json = os.path.join(args.data_dir, 'calms21_task1_train.json')
    test_json  = os.path.join(args.data_dir, 'calms21_task1_test.json')

    train_full = CalMS21Dataset(train_json, seq_len=args.seq_len,
                                stride=args.stride, fit_scaler=True)
    val_len    = int(0.2 * len(train_full))
    train_len  = len(train_full) - val_len
    train_dataset, val_dataset = random_split(
        train_full, [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed))

    test_dataset = CalMS21Dataset(test_json, seq_len=args.seq_len,
                                  stride=args.stride,
                                  scaler=train_full.scaler, fit_scaler=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = MSGLTransformer(input_dim=input_dim, num_classes=num_classes,
                            seq_len=args.seq_len, num_heads=args.num_heads,
                            d_model=args.d_model, num_layers=args.num_layers,
                            dff=args.dff, dropout=args.dropout).to(device)

    class_weights = train_full.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_loss    = float('inf')
    best_model_state = None
    patience_counter = 0
    history          = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f} | '
              f'Time: {time.time()-t0:.1f}s')

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print('  Model saved')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break

    # Evaluation
    model.load_state_dict(best_model_state)
    _, test_acc, preds, labels, probs = evaluate(model, test_loader, criterion, device)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)

    print(f'\n=== Test Results ===')
    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1-Score:  {f1:.4f}')

    # Save confusion matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'Confusion_matrix_test.png'), dpi=150)
    plt.close()

    # Save model
    torch.save({'model_state_dict': best_model_state, 'args': vars(args),
                'test_metrics': {'accuracy': accuracy, 'precision': precision,
                                 'recall': recall, 'f1': f1}},
               os.path.join(args.output_dir, 'best_model.pt'))

    print(f'\nAll results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
