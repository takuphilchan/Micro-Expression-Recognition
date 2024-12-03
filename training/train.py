import os
import cv2
import math
import torch
import tqdm
import gc
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as TR
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from models import SpatialAttention, ChannelAttention
from dataset_processing import VideoDataset
from configs import ablation_configurations
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc, precision_recall_curve

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_metrics_to_file(metrics, save_path):
    with open(save_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def check_grad_flow(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                print(f'{name}: {param.grad.abs().mean().item()}')
            else:
                print(f'{name}: No gradient')

def is_valid_excel_file(file_path):
    try:
        pd.read_excel(file_path, engine='openpyxl')
        return True
    except Exception as e:
        return False

def get_existing_configurations(existing_results):
    return set(zip(existing_results['test_name'], existing_results['fold']))

def load_existing_results(excel_file):
    if os.path.exists(excel_file):
        return pd.read_excel(excel_file)
    else:
        return pd.DataFrame(columns=[
            'test_name', 'fold', 'epoch', 'avg_train_loss', 'train_accuracy',
            'train_precision', 'train_recall', 'train_f1_score', 'train_balanced_accuracy',
            'train_pr_auc', 'avg_val_loss', 'val_accuracy', 'val_precision', 'val_recall',
            'val_f1_score', 'val_balanced_accuracy', 'val_pr_auc'
        ])

def save_checkpoint(state, filename):
    torch.save(state, filename)

# Early stopping mechanism
def early_stopping(epochs_without_improvement, patience):
    return epochs_without_improvement > patience

def print_metrics(epoch, train_metrics, val_metrics, avg_train_loss, avg_val_loss):
    """
    Print the training and validation metrics for the current epoch.

    Parameters:
    - epoch: Current epoch number.
    - train_metrics: Dictionary of training metrics.
    - val_metrics: Dictionary of validation metrics.
    - avg_train_loss: Average training loss for the current epoch.
    - avg_val_loss: Average validation loss for the current epoch.
    """
    print(f"Epoch {epoch + 1}")
    print("-" * 50)

    # Training metrics
    print("Training Metrics:")
    print(f"  Average Training Loss: {avg_train_loss:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall: {train_metrics['recall']:.4f}")
    print(f"  F1 Score: {train_metrics['f1']:.4f}")
    print(f"  Balanced Accuracy: {train_metrics['balanced_accuracy']:.4f}")
    print(f"  PR AUC: {train_metrics.get('pr_auc', 0.0):.4f}")

    print()

    # Validation metrics
    print("Validation Metrics:")
    print(f"  Average Validation Loss: {avg_val_loss:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  F1 Score: {val_metrics['f1']:.4f}")
    print(f"  Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
    print(f"  PR AUC: {val_metrics.get('pr_auc', 0.0):.4f}")

    print("-" * 50)
    print()

def save_to_excel(ablation_results, results_dir):
    columns = [
        'test_name', 'fold', 'epoch', 'avg_train_loss', 'train_accuracy',
        'train_precision', 'train_recall', 'train_f1_score', 'train_balanced_accuracy',
        'train_pr_auc', 'avg_val_loss', 'val_accuracy', 'val_precision', 'val_recall',
        'val_f1_score', 'val_balanced_accuracy', 'val_pr_auc'
    ]
    rows = []

    for test_name, results in ablation_results.items():
        for fold, fold_results in enumerate(results, start=1):
            for result in fold_results:
                row = [
                    test_name,
                    fold,
                    result['epoch'],
                    result['avg_train_loss'],
                    result['train_accuracy'],
                    result['train_precision'],
                    result['train_recall'],
                    result['train_f1_score'],
                    result['train_balanced_accuracy'],
                    result['train_pr_auc'],
                    result['avg_val_loss'],
                    result['val_accuracy'],
                    result['val_precision'],
                    result['val_recall'],
                    result['val_f1_score'],
                    result['val_balanced_accuracy'],
                    result['val_pr_auc']
                ]
                rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    excel_filename = os.path.join(results_dir, 'hyper_ablation_results.xlsx')

    try:
        if os.path.exists(excel_filename):
            existing_df = pd.read_excel(excel_filename)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_excel(excel_filename, index=False)
        print(f"Results saved to {excel_filename}")
    except Exception as e:
        print(f"An error occurred while saving the results: {e}")

def print_final_results(ablation_results):
    for test_name, results in ablation_results.items():
        if not results:
            print(f"No results for test: {test_name}")
            continue

        avg_training_loss = np.mean([res.get('training_loss', np.nan) for res in results])
        avg_accuracy = np.mean([res.get('accuracy', np.nan) for res in results])
        avg_precision = np.mean([res.get('precision', np.nan) for res in results])
        avg_recall = np.mean([res.get('recall', np.nan) for res in results])
        avg_f1 = np.mean([res.get('f1_score', np.nan) for res in results])
        avg_balanced_accuracy = np.mean([res.get('balanced_accuracy', np.nan) for res in results])
        avg_train_pr_auc = np.mean([res.get('train_pr_auc', np.nan) for res in results])
        avg_val_pr_auc = np.mean([res.get('val_pr_auc', np.nan) for res in results])

        print(f"Test: {test_name}")
        print(f"Average Training Loss: {avg_training_loss:.2f}")
        print(f"Average Accuracy: {avg_accuracy:.2f}")
        print(f"Average Precision: {avg_precision:.2f}")
        print(f"Average Recall: {avg_recall:.2f}")
        print(f"Average F1-score: {avg_f1:.2f}")
        print(f"Average Balanced Accuracy: {avg_balanced_accuracy:.2f}")
        print(f"Average Training PR AUC: {avg_train_pr_auc:.2f}")
        print(f"Average Validation PR AUC: {avg_val_pr_auc:.2f}")

def plot_confusion_matrix(true_labels, predictions, class_names, results_dir, test_name, fold, mode='train'):
    """
    Plots and saves the confusion matrix.

    Args:
        true_labels (list): True labels of the data.
        predictions (list): Predicted labels of the data.
        class_names (list): List of class names.
        results_dir (str): Directory to save the plot.
        test_name (str): Name of the test configuration.
        fold (int): Current fold number.
        mode (str): 'train' for training confusion matrix, 'val' for validation.
    """
    cm = confusion_matrix(true_labels, predictions, labels=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {test_name} - Fold {fold + 1} - {mode.capitalize()}')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plot_filename = f'{test_name}_fold{fold + 1}_{mode}_confusion_matrix.png'
    plot_filepath = os.path.join(results_dir, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.close()

    print(f'{mode.capitalize()} confusion matrix saved to: {plot_filepath}')

def log_metrics(writer, phase, loss, metrics, epoch):
    writer.add_scalar(f'{phase}/Loss', loss, epoch)
    for metric, value in metrics.items():
        writer.add_scalar(f'{phase}/{metric.capitalize()}', value, epoch)

def log_per_class_metrics(writer, epoch, y_true, y_pred, phase):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        writer.add_scalar(f'{phase}/Class_{i}_Precision', p, epoch)
        writer.add_scalar(f'{phase}/Class_{i}_Recall', r, epoch)
        writer.add_scalar(f'{phase}/Class_{i}_F1_Score', f, epoch)
    print(f"{phase} - Epoch {epoch} per-class metrics:")
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print(f"Class {i}: Precision: {p}, Recall: {r}, F1-Score: {f}")

def log_confusion_matrix(writer, epoch, y_true, y_pred, class_names, phase):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues')
    plt.title(f'{phase} Confusion Matrix')
    fig.colorbar(cax)

    # Set x and y ticks with labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Set the labels for the axes
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Adjust layout to ensure labels fit
    plt.tight_layout()

    # Iterate over data dimensions and create text annotations
    threshold = cax.get_array().max() / 2  # Use the maximum value to set the threshold for text color
    for (i, j), val in np.ndenumerate(cm):
        text_color = 'white' if cm[i, j] > threshold else 'black'
        ax.text(j, i, val, ha='center', va='center', color=text_color)

    # Log the confusion matrix as a figure to TensorBoard
    writer.add_figure(f'{phase}/Confusion_Matrix', fig, epoch)

    # Close the figure to avoid memory issues
    plt.close(fig)

    # Print the confusion matrix for reference
    print(f"{phase} - Epoch {epoch} Confusion Matrix:\n{cm}")

def log_roc_pr_curves(writer, epoch, y_true, y_probs, class_names, phase):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    for i, class_name in enumerate(class_names):
        # Ensure y_true is an array-like and y_probs is a 2D array
        if y_true.ndim == 1 and y_probs.ndim == 2 and y_probs.shape[1] == len(class_names):
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (class {class_name}) (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {class_name}')
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2))  # Move legend to the bottom
            plt.tight_layout()  # Adjust layout to make everything fit
            writer.add_figure(f'{phase}/ROC_Curve_{class_name}', plt.gcf(), epoch)
            plt.close()

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true == i, y_probs[:, i])
            pr_auc = auc(recall, precision)

            plt.figure()
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (class {class_name}) (area = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve for {class_name}')
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2))  # Move legend to the bottom
            plt.tight_layout()  # Adjust layout to make everything fit
            writer.add_figure(f'{phase}/PR_Curve_{class_name}', plt.gcf(), epoch)
            plt.close()
        else:
            print(f"Invalid input dimensions for class {class_name}: y_true.ndim = {y_true.ndim}, y_probs.ndim = {y_probs.ndim}")

def compute_metrics(y_true, y_pred, y_probs=None):
    """
    Compute metrics for multi-class classification.

    Parameters:
    - y_true: Array-like of true labels.
    - y_pred: Array-like of predicted labels.
    - y_probs: Array-like of predicted probabilities (for ROC AUC and PR AUC).

    Returns:
    - metrics: Dictionary of computed metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    if y_probs is not None:
        y_probs = np.array(y_probs)
        num_classes = len(np.unique(y_true))

        # Ensure y_probs is a 2D array
        if y_probs.ndim == 1:
            raise ValueError("y_probs should be a 2D array with shape (num_samples, num_classes)")

        # Adjust the shape of y_probs if necessary
        if y_probs.shape[1] != num_classes:
            if y_probs.shape[1] > num_classes:
                y_probs = y_probs[:, :num_classes]  # Truncate extra columns
            elif y_probs.shape[1] < num_classes:
                padding = np.zeros((y_probs.shape[0], num_classes - y_probs.shape[1]))
                y_probs = np.hstack((y_probs, padding))

        # Normalize y_probs to ensure they sum to 1 across classes for each sample
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

        # Compute ROC AUC
        metrics['roc_auc'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')

        # Compute Precision-Recall AUC
        pr_auc = 0
        for i in range(num_classes):
            true_binary = (y_true == i).astype(int)
            pred_binary = y_probs[:, i]
            precision, recall, _ = precision_recall_curve(true_binary, pred_binary)
            pr_auc += auc(recall, precision)
        pr_auc /= num_classes
        metrics['pr_auc'] = pr_auc

    return metrics

def train_one_fold(train_loader, val_loader, model, criterion, optimizer, scheduler, num_epochs, device, writer, fold, patience, results_dir, dataset, checkpoint_dir, test_name,accumulation_steps):
    best_metric = -float('inf')  # Initialize with a very low value for F1 score
    epochs_without_improvement = 0
    fold_result = []

    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        y_true_train = []
        y_pred_train = []
        y_probs_train = []

        optimizer.zero_grad()

        # Training Phase
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Training") as pbar:
            for i, (inputs, padding_mask, labels) in enumerate(train_loader):
                inputs, padding_mask, labels = inputs.to(device, non_blocking=True), padding_mask.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                with autocast():
                    outputs = model(inputs, padding_mask)
                    loss = criterion(outputs, labels)
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                running_train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                y_true_train.extend(labels.cpu().numpy())
                y_pred_train.extend(preds.cpu().numpy())
                y_probs_train.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())

                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})
                pbar.refresh()

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_metrics = compute_metrics(y_true_train, y_pred_train, np.array(y_probs_train))

        model.eval()
        running_val_loss = 0.0
        y_true_val = []
        y_pred_val = []
        y_probs_val = []

        # Validation Phase
        with torch.no_grad(), tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Validation") as pbar:
            for inputs, padding_mask, labels in val_loader:
                inputs, padding_mask, labels = inputs.to(device, non_blocking=True), padding_mask.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                outputs = model(inputs, padding_mask)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())
                y_probs_val.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())

                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})
                pbar.refresh()

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_metrics = compute_metrics(y_true_val, y_pred_val, np.array(y_probs_val))
        avg_val_f1 = val_metrics['f1']  # Extract the F1 score for early stopping

        # Log Metrics
        log_metrics(writer, 'Training', avg_train_loss, train_metrics, epoch)
        log_metrics(writer, 'Validation', avg_val_loss, val_metrics, epoch)
        log_per_class_metrics(writer, epoch, y_true_train, y_pred_train, phase='Train')
        log_per_class_metrics(writer, epoch, y_true_val, y_pred_val, phase='Validation')
        log_confusion_matrix(writer, epoch, y_true_train, y_pred_train, class_names=[class_name for class_name, _ in sorted(dataset.class_to_idx.items(), key=lambda x: x[1])], phase='Train')
        log_confusion_matrix(writer, epoch, y_true_val, y_pred_val, class_names=[class_name for class_name, _ in sorted(dataset.class_to_idx.items(), key=lambda x: x[1])], phase='Validation')
        log_roc_pr_curves(writer, epoch, y_true_train, np.array(y_probs_train), class_names=[class_name for class_name, _ in sorted(dataset.class_to_idx.items(), key=lambda x: x[1])], phase='Train')
        log_roc_pr_curves(writer, epoch, y_true_val, np.array(y_probs_val), class_names=[class_name for class_name, _ in sorted(dataset.class_to_idx.items(), key=lambda x: x[1])], phase='Validation')

        print(f"Epoch {epoch+1} Metrics:")
        print(f"  Training - Accuracy: {train_metrics['accuracy']:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1 Score: {train_metrics['f1']:.4f}, Balanced Accuracy: {train_metrics['balanced_accuracy']:.4f}, PR AUC: {train_metrics['pr_auc']:.4f}")
        print(f"  Validation - Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1 Score: {val_metrics['f1']:.4f}, Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}, PR AUC: {val_metrics['pr_auc']:.4f}")

        metrics = {
            'epoch': epoch,
            'avg_train_loss': avg_train_loss,
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1_score': train_metrics['f1'],
            'train_balanced_accuracy': train_metrics['balanced_accuracy'],
            'train_pr_auc': train_metrics['pr_auc'],
            'avg_val_loss': avg_val_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1_score': val_metrics['f1'],
            'val_balanced_accuracy': val_metrics['balanced_accuracy'],
            'val_pr_auc': val_metrics['pr_auc']
        }

        fold_result.append(metrics)

        # Step the scheduler based on the current epoch
        scheduler.step()

        # Save best model
        if avg_val_f1 > best_metric:  # Compare using F1 score
            best_metric = avg_val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'{test_name}_fold_{fold}.pth'))
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement > patience:
            print("Early stopping due to no improvement")
            break

    return fold_result

def reinitialize_weights(module):
    if isinstance(module, nn.Linear):
        # Linear layers are initialized using Kaiming uniform by default
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, ChannelAttention):
        nn.init.kaiming_uniform_(module.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(module.fc2.weight, a=math.sqrt(5))
    elif isinstance(module, SpatialAttention):
        # Initialize SpatialAttention
        nn.init.kaiming_uniform_(module.conv1.weight, a=math.sqrt(5))
    # Add more initializations as needed for other layer types

# Function to group subjects by class and count their samples
def group_subjects_by_class(dataset):
    class_subject_counts = defaultdict(lambda: defaultdict(int))
    for idx, label in enumerate(dataset.labels):
        subject = dataset.subjects[idx]
        class_subject_counts[label][subject] += 1
    return class_subject_counts


def allocate_subjects_unique(class_subject_counts, train_ratio=0.8, max_spno_val=2):
    train_subjects, val_subjects = set(), set()
    train_class_counts, val_class_counts = defaultdict(int), defaultdict(int)
    total_samples_per_class = {label: sum(subject_counts.values()) for label, subject_counts in class_subject_counts.items()}
    train_samples_target_per_class = {label: int(total * train_ratio) for label, total in total_samples_per_class.items()}

    spno_subjects_per_class = defaultdict(int)  # Track 'spNO' subjects per class for validation
    spno_subjects_to_train = defaultdict(set)  # Track 'spNO' subjects to be considered for training

    def add_subjects_to_set(subject_set, other_set, target_class_counts, source_class_counts, is_validation=False):
        nonlocal spno_subjects_per_class, spno_subjects_to_train  # Use the outer scope variables

        for label in source_class_counts.keys():
            sorted_subjects = sorted(source_class_counts[label].items(), key=lambda x: x[1], reverse=True)
            for subject, count in sorted_subjects:
                if subject in other_set:
                    continue

                # For validation set, limit 'spNO' subjects per class
                if is_validation and "spNO" in subject:
                    if spno_subjects_per_class[label] >= max_spno_val:
                        # Track 'spNO' subjects to be added to training set
                        spno_subjects_to_train[label].add(subject)
                        continue
                    else:
                        spno_subjects_per_class[label] += 1  # Increment the 'spNO' subject count for this class in validation

                # Normal allocation logic for training/validation
                if subject_set == train_subjects:
                    if target_class_counts[label] + count <= train_samples_target_per_class[label]:
                        subject_set.add(subject)
                        target_class_counts[label] += count
                    else:
                        break
                else:
                    if target_class_counts[label] + count <= total_samples_per_class[label] - train_samples_target_per_class[label]:
                        subject_set.add(subject)
                        target_class_counts[label] += count
                    else:
                        break

    # Allocate subjects to training set first
    print("Allocating subjects to training set...")
    add_subjects_to_set(train_subjects, val_subjects, train_class_counts, class_subject_counts)
    print(f"Training subjects: {train_subjects}")
    print(f"Training class counts: {dict(train_class_counts)}")

    # Allocate subjects to validation set
    print("Allocating subjects to validation set...")
    add_subjects_to_set(val_subjects, train_subjects, val_class_counts, class_subject_counts, is_validation=True)
    print(f"Validation subjects: {val_subjects}")
    print(f"Validation class counts: {dict(val_class_counts)}")

    # Ensure 'spNO' subjects in validation not used are added to training
    for label, subjects in spno_subjects_to_train.items():
        for subject in subjects:
            if subject not in train_subjects and subject not in val_subjects:
                train_subjects.add(subject)
                # Update train class counts
                subject_count = class_subject_counts[label][subject]
                train_class_counts[label] += subject_count

    # Ensure that the subjects in each set are unique
    assert not (train_subjects & val_subjects), "Subjects overlap between training and validation sets"

    return train_subjects, val_subjects, train_class_counts, val_class_counts

# Function to sample a specific number of samples from each class
def sample_from_each_class(indices, labels, num_samples_per_class):
    class_indices = defaultdict(list)
    for idx in indices:
        label = labels[idx]
        class_indices[label].append(idx)

    sampled_indices = []
    for label, indices_list in class_indices.items():
        if len(indices_list) < num_samples_per_class:
            raise ValueError(f"Not enough samples in class {label} to sample {num_samples_per_class} samples.")

        sampled_indices.extend(random.sample(indices_list, num_samples_per_class))

    return sampled_indices

# Function to print subjects for a given set of indices
def print_subjects_for_indices(indices, dataset):
    subjects = [dataset.subjects[i] for i in indices]
    subject_counts = defaultdict(int)
    for subject in subjects:
        subject_counts[subject] += 1

    print("Subjects and their counts:")
    for subject, count in subject_counts.items():
        print(f"  Subject {subject}: {count} videos")

# Function to print subjects and classes for a given set of indices
def print_subjects_by_class(indices, dataset):
    subjects_by_class = defaultdict(set)
    for i in indices:
        label = dataset.labels[i]
        subject = dataset.subjects[i]
        subjects_by_class[label].add(subject)

    for label, subjects in subjects_by_class.items():
        print(f"Class {label}:")
        for subject in sorted(subjects):
            print(f"  Subject {subject}")


# Ensure consistent seeding for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights_and_sampler(labels):
    """Compute class weights and sampler for the imbalanced dataset."""
    label_encoder = LabelEncoder()
    labels_numeric = label_encoder.fit_transform(labels)

    # Compute class sample counts
    class_sample_counts = np.bincount(labels_numeric)
    total_samples = len(labels_numeric)
    num_classes = len(class_sample_counts)

    # Compute class weights
    # Adjust class weights to be inversely proportional to class frequencies
    class_weights = total_samples / (num_classes * class_sample_counts)

    # Normalize class weights if needed
    class_weights = class_weights / np.max(class_weights)  # Normalization to [0, 1]

    # Assign weights to each sample
    sample_weights = class_weights[labels_numeric]

    # Create a weighted sampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Convert class weights to tensor for use in loss functions
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    return class_weights_tensor, sampler, label_encoder


# Print the video count per subject for each class
def print_videos_per_subject_per_class(dataset):
    class_subjects = defaultdict(lambda: defaultdict(int))
    for idx, (video_file, label) in enumerate(zip(dataset.video_files, dataset.labels)):
        subject = dataset.extract_subject_from_file(os.path.basename(video_file))
        class_index = dataset.class_to_idx.get(label, None)
        if class_index is not None:
            class_subjects[class_index][subject] += 1

    # Print subjects and video counts per class
    print("Subjects per class and their video counts:")
    for cls in sorted(class_subjects.keys()):
        print(f"Class {cls}:")
        for subject, count in class_subjects[cls].items():
            print(f"  Subject {subject}: {count} videos")
        total_videos = sum(class_subjects[cls].values())
        print(f"  Total videos for Class {cls}: {total_videos}")

def print_class_distribution(set_name, indices, labels):
    class_labels = [labels[i] for i in indices]
    distribution = compute_class_distribution(class_labels)
    print(f"{set_name} class distribution: {distribution}")

def compute_class_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))

# Example of calculating class distributions for a fold
def print_fold_distributions(train_indices, val_indices, dataset):
    train_labels = [dataset.labels[idx] for idx in train_indices]
    val_labels = [dataset.labels[idx] for idx in val_indices]

    train_distribution = compute_class_distribution(train_labels)
    val_distribution = compute_class_distribution(val_labels)

    print(f"Training set class distribution: {train_distribution}")
    print(f"Validation set class distribution: {val_distribution}")

class ConsistentRandomBrightness:
    def __init__(self, min_brightness=0.95, max_brightness=1.05):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.brightness_factor = None  # This will store the random brightness factor for a given sequence

    def __call__(self, video_frames):
        # If the brightness factor has not been set for the current sequence, generate one
        if self.brightness_factor is None:
            self.brightness_factor = random.uniform(self.min_brightness, self.max_brightness)
        # Apply the same brightness factor to all frames
        return [TR.adjust_brightness(frame, self.brightness_factor) for frame in video_frames]

    def reset(self):
        # Call this to reset the brightness factor between sequences
        self.brightness_factor = None

def seeded_transform(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    consistent_brightness = ConsistentRandomBrightness(0.95, 1.05)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(contrast=0.1, saturation=0.1, hue=0.2),
        transforms.ToTensor(),
    ]), consistent_brightness

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights (1D tensor)

    def forward(self, inputs, targets):
        # Cross-entropy loss (per-sample loss)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # print(f"Cross-Entropy Loss (ce_loss): {ce_loss}")

        # Probabilities from cross-entropy
        probs = torch.exp(-ce_loss)  # Equivalent to probabilities of the correct class
        # print(f"Probabilities for target class (pt): {probs}")

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - probs) ** self.gamma
        # print(f"Focal Weight: {focal_weight}")

        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            # Gather alpha values for each target class
            alpha_t = self.alpha.gather(0, targets)
            # print(f"Alpha (class weights) for each target: {alpha_t}")
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # print(f"Focal Loss before reduction: {focal_loss}")

        # Return the mean focal loss across the batch
        final_loss = focal_loss.mean()
        # print(f"Final Focal Loss (mean): {final_loss}")

        return final_loss

def get_optimizer_and_scheduler(model, train_loader, pretrained_lr, non_pretrained_lr, min_lr, weight_decay, total_epochs):
    def extract_parameters(extractor):
        return {
            'pretrained': [p for n, p in extractor.pretrained_model.named_parameters()],
            'spatial_attention': [p for n, p in extractor.spatial_attention.named_parameters()]
        }

    params = {}
    if hasattr(model.feature_extractor, 'pretrained_model') and hasattr(model.feature_extractor, 'spatial_attention'):
        params.update(extract_parameters(model.feature_extractor))

    if model.use_channel_attention and hasattr(model, 'channel_attention'):
        params['channel_attention'] = [p for n, p in model.channel_attention.named_parameters()]

    if hasattr(model, 'classifier'):
        params['classifier'] = [p for n, p in model.classifier.named_parameters()]

    # Define parameter groups with their specific learning rates (global weight decay applies to all groups)
    param_groups = [
        {'params': params.get('pretrained', []), 'lr': pretrained_lr},  # Pretrained layers
        {'params': params.get('spatial_attention', []), 'lr': non_pretrained_lr}, # Non-pretrained layers
        {'params': params.get('channel_attention', []), 'lr': non_pretrained_lr},
        {'params': params.get('classifier', []), 'lr': non_pretrained_lr}
    ]

    # Check for duplicate parameters
    all_params = sum(params.values(), [])
    if len(set(all_params)) != len(all_params):
        raise ValueError("Some parameters appear in more than one parameter group")

    # Create optimizer with global weight decay applied uniformly to all parameter groups
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    # Automatically calculate total steps based on DataLoader
    num_batches_per_epoch = len(train_loader)  # Number of batches per epoch
    total_steps = num_batches_per_epoch * total_epochs  # Total steps for all epochs

    # Adjust the max learning rates to reduce instability
    max_lrs = [
        pretrained_lr * 1.5,    # Conservative for pretrained layers
        non_pretrained_lr * 1.5, # More aggressive for non-pretrained layers
        non_pretrained_lr * 1.5,
        non_pretrained_lr * 1.5
    ]
    # OneCycleLR scheduler with separate max learning rates for each parameter group
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
          optimizer,
          max_lr=max_lrs,
          total_steps=total_steps,
          pct_start=0.3,            # Start decay earlier
          anneal_strategy='cos',  # Switch to linear annealing for smoother decay
          final_div_factor=20        # Reduce learning rate more at the end
      )

    return optimizer, scheduler

def train_model():
    seed = 42
    set_seed(seed)

    # Parameters
    NUM_CLASSES = 3
    BATCH_SIZE = 4
    VAL_BATCH_SIZE = 4
    NUM_EPOCHS = 15
    PRETRAINED_LR = 2e-4        # Increase learning rate
    BASE_LR = 1e-3             # Increase learning rate
    MIN_LR = 1e-6
    PATIENCE = NUM_EPOCHS
    GAMMA_LOSS = 1.0         # Lower gamma for focal loss, or try CrossEntropyLoss
    ACCUMULATION_STEPS = 2      # Reduce gradient accumulation to update more frequently
    WEIGHT_DECAY = 1e-4  # Weight decay

    # File paths
    path = "/content/drive/MyDrive/COMPOSITE_CASMEII"
    dataset_name = "BEST_COMPOSITE_3_CLASS"
    checkpoint_dir = os.path.join(path, "ablation_ern_3_final", "checkpoints")
    results_dir = os.path.join(path, "ablation_ern_3_final", "results")
    log_dir = os.path.join(results_dir, "tensorboard_logs")
    results_file = os.path.join(results_dir, 'hyper_ablation_results.xlsx')
    existing_results = load_existing_results(results_file)
    existing_configurations = get_existing_configurations(existing_results)

    # Helper function to create directories
    def create_directories():
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    create_directories()

    # Load dataset
    dataset_folder = os.path.join(path, dataset_name)
    transform = seeded_transform(seed)
    dataset = VideoDataset(dataset_folder, transform=transform, seed=seed)
    print(f"Total videos in dataset: {len(dataset)}")

    # Print parameters
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Batch size (train): {BATCH_SIZE}")
    print(f"Batch size (validation): {VAL_BATCH_SIZE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Base Learning rate: {BASE_LR}")
    print(f"Pretrained Learning rate: {PRETRAINED_LR}")
    print(f"Minimum learning rate: {MIN_LR}")
    print(f"Early stopping patience (epochs): {PATIENCE}")
    print(f"Gamma value for focal loss: {GAMMA_LOSS}")
    print(f"Gradient accumulation steps: {ACCUMULATION_STEPS}")

    # Prepare Stratified K-Fold
    labels = dataset.labels
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)

    def print_fold_distributions():
        class_subject_counts = group_subjects_by_class(dataset)
        for fold_num, (train_indices, val_indices) in enumerate(skf.split(np.arange(len(labels)), labels), 1):
            train_subjects, val_subjects, _, _ = allocate_subjects_unique(class_subject_counts, train_ratio=0.85)
            filtered_train_indices = [i for i in train_indices if dataset.subjects[i] in train_subjects]
            filtered_val_indices = [i for i in val_indices if dataset.subjects[i] in val_subjects]
            print(f"\nFold {fold_num} Distribution:")
            print_class_distribution("Training set", filtered_train_indices, labels)
            print_class_distribution("Validation set", filtered_val_indices, labels)

    print_fold_distributions()

    ablation_tests = ablation_configurations()
    ablation_results = {name: [] for name in ablation_tests.keys()}

    for config_name, config in ablation_tests.items():
        print(f"\nUsing configuration: {config_name}")

        for fold_num, (train_indices, val_indices) in enumerate(skf.split(np.arange(len(labels)), labels), 1):
            print(f"\nProcessing Fold {fold_num} with configuration {config_name}...")

            # Check if the configuration and fold already exist
            if (config_name, fold_num) in existing_configurations:
                print(f"Skipping Fold {fold_num} with configuration {config_name} as it already exists.")
                continue

            class_subject_counts = group_subjects_by_class(dataset)
            train_subjects, val_subjects, _, _ = allocate_subjects_unique(class_subject_counts, train_ratio=0.85)
            filtered_train_indices = [i for i in train_indices if dataset.subjects[i] in train_subjects]
            filtered_val_indices = [i for i in val_indices if dataset.subjects[i] in val_subjects]

            # Print subjects and classes
            print_subjects_by_class(filtered_train_indices, dataset)
            print_subjects_by_class(filtered_val_indices, dataset)
            print_class_distribution("Training set", filtered_train_indices, labels)
            print_class_distribution("Validation set", filtered_val_indices, labels)

            train_labels = [dataset.labels[idx] for idx in filtered_train_indices]
            class_weights, train_sampler, _ = compute_class_weights_and_sampler(train_labels)
            print("Class weights:", class_weights)

            val_labels = [dataset.labels[idx] for idx in filtered_val_indices]
            val_class_distribution = compute_class_distribution(val_labels)
            print(f"Validation set class distribution: {val_class_distribution}")

            # DataLoaders
            train_dataset = Subset(dataset, filtered_train_indices)
            val_dataset = Subset(dataset, filtered_val_indices)

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                sampler=train_sampler,
                num_workers=1,
                pin_memory=True,
                collate_fn=VideoDataset.collate_wrapper_with_mask
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=VAL_BATCH_SIZE,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                collate_fn=VideoDataset.collate_wrapper_with_mask
            )

            # TensorBoard writer
            fold_log_dir = os.path.join(log_dir, config_name, f"fold_{fold_num}")
            os.makedirs(fold_log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=fold_log_dir)

            # Initialize and train the model
            model = config['model'].to(device)

            if hasattr(model.feature_extractor, 'spatial_attention') and model.feature_extractor.spatial_attention:
                model.feature_extractor.spatial_attention.apply(reinitialize_weights)
            if hasattr(model, 'channel_attention') and model.channel_attention:
                model.channel_attention.apply(reinitialize_weights)
            if hasattr(model, 'classifier') and model.classifier:
                model.classifier.apply(reinitialize_weights)

            # Call the function to get the optimizer and scheduler
            optimizer, scheduler = get_optimizer_and_scheduler(
                model,
                train_loader,
                PRETRAINED_LR,
                BASE_LR,
                MIN_LR,
                WEIGHT_DECAY,
                NUM_EPOCHS
            )
            class_weights_alpha = torch.tensor([0.7, 1.3, 1.1], dtype=torch.float)
            criterion = FocalLoss(alpha=class_weights_alpha.to(device), gamma=GAMMA_LOSS).to(device)

            fold_result = train_one_fold(
                train_loader, val_loader, model, criterion, optimizer, scheduler,
                NUM_EPOCHS, device, writer, fold=fold_num, patience=PATIENCE,
                results_dir=results_dir, dataset=dataset, checkpoint_dir=checkpoint_dir,
                test_name=config_name, accumulation_steps=ACCUMULATION_STEPS
            )

            ablation_results[config_name].append(fold_result)
            print(f"Fold {fold_num} with configuration {config_name} completed. Results:")
            print(fold_result)
            writer.close()

            cleanup(model, optimizer, scheduler, train_loader, val_loader)

        print(f"Completed all folds with configuration {config_name}.")
        save_to_excel(ablation_results, results_dir)

    print("All folds and configurations have been processed and results saved.")

def cleanup(model, optimizer, scheduler, train_loader, val_loader):
    # Move model to CPU to free GPU memory
    # model.cpu()
    del model
    del optimizer
    del scheduler
    del train_loader
    del val_loader
    gc.collect()  # Force garbage collection
    torch.cuda.empty_cache()

# Print model parameters to check for shared weights
# def print_model_params(model, fold):
#     print(f"Model parameters for fold {fold}:")
#     for param in model.parameters():
#         print(param.data)

def print_model_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: mean={param.data.mean()}, std={param.data.std()}, min={param.data.min()}, max={param.data.max()}")

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
    train_model()