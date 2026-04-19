import argparse
import csv
import datetime
import json
import pathlib
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageEnhance
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_device() -> Tuple[torch.device, str]:
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        name = torch.cuda.get_device_name(0)
        print(f"[GPU] CUDA device detected: {name}")
        return dev, name

    try:
        import torch_directml  # type: ignore

        dev = torch_directml.device()
        print("[GPU] DirectML device detected")
        return dev, "DirectML"
    except Exception:
        print("[WARN] No GPU backend found - falling back to CPU")
        return torch.device('cpu'), "CPU"


class BrightnessScale:
    def __init__(self, low: float = 0.8, high: float = 1.2):
        self.low = low
        self.high = high

    def __call__(self, img):
        factor = float(np.random.uniform(self.low, self.high))
        return ImageEnhance.Brightness(img).enhance(factor)


class EmotionLensCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 7),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ResultsLoggerCallback:
    def __init__(self, filepath='models/training_results.txt', gpu_name='CPU', total_target_epochs=100):
        self.filepath = pathlib.Path(filepath)
        self.gpu_name = gpu_name
        self.total_target_epochs = total_target_epochs
        self.start_time = None
        self.best_val_acc = 0.0
        self.best_val_loss = 999.0
        self.total_epochs = 0

    def on_train_begin(self):
        self.start_time = datetime.datetime.now()
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if self.filepath.exists():
            self.filepath.unlink()
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("  EmotionLens CNN - Training Results Log\n")
            f.write("=" * 60 + "\n")
            f.write(f"  Started   : {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  GPU       : {self.gpu_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"{'Epoch':<8}{'Train Acc':<14}{'Val Acc':<14}{'Train Loss':<14}{'Val Loss':<14}{'LR':<12}\n")
            f.write("-" * 76 + "\n")
            f.flush()

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], lr: float):
        acc = metrics.get('accuracy', 0.0)
        val_acc = metrics.get('val_accuracy', 0.0)
        loss = metrics.get('loss', 0.0)
        val_loss = metrics.get('val_loss', 0.0)
        self.total_epochs = epoch + 1
        self.best_val_acc = max(self.best_val_acc, val_acc)
        self.best_val_loss = min(self.best_val_loss, val_loss)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"{epoch+1:<8}{acc:<14.4f}{val_acc:<14.4f}{loss:<14.4f}{val_loss:<14.4f}{lr:<12.6f}\n")
            f.flush()

    def on_train_end(self, test_loss: float, test_acc: float):
        end_time = datetime.datetime.now()
        duration = (end_time - self.start_time).seconds // 60
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("  FINAL TRAINING SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"  Best Val Accuracy   : {self.best_val_acc:.4f}  ({self.best_val_acc*100:.2f}%)\n")
            f.write(f"  Best Val Loss       : {self.best_val_loss:.4f}\n")
            f.write(f"  Test Accuracy       : {test_acc:.4f}  ({test_acc*100:.2f}%)\n")
            f.write(f"  Test Loss           : {test_loss:.4f}\n")
            f.write(f"  Total Epochs Run    : {self.total_epochs}\n")
            f.write(f"  Early Stopped       : {'Yes' if self.total_epochs < self.total_target_epochs else 'No'}\n")
            f.write(f"  Training Duration   : {duration} minutes\n")
            f.write(f"  GPU Used            : {self.gpu_name}\n")
            f.write(f"  Completed At        : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
            f.flush()


class LiveMetricsCallback:
    def __init__(self, filepath='models/live_metrics.json', total_epochs=100):
        self.filepath = pathlib.Path(filepath)
        self.total_epochs = total_epochs
        self.history = {
            'epoch': [], 'accuracy': [], 'val_accuracy': [],
            'loss': [], 'val_loss': [], 'lr': []
        }

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], lr: float):
        self.history['epoch'].append(epoch + 1)
        self.history['accuracy'].append(round(metrics.get('accuracy', 0.0), 4))
        self.history['val_accuracy'].append(round(metrics.get('val_accuracy', 0.0), 4))
        self.history['loss'].append(round(metrics.get('loss', 0.0), 4))
        self.history['val_loss'].append(round(metrics.get('val_loss', 0.0), 4))
        self.history['lr'].append(round(lr, 8))

        payload = {
            'current_epoch': epoch + 1,
            'total_epochs': self.total_epochs,
            'history': self.history,
            'status': 'training'
        }
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(payload, f)

    def on_train_end(self):
        if self.filepath.exists():
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {
                'current_epoch': 0,
                'total_epochs': self.total_epochs,
                'history': self.history,
            }
        data['status'] = 'complete'
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += int(labels.size(0))
    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: List[int] = []
    all_true: List[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_true.extend(labels.numpy().tolist())
    return np.array(all_true), np.array(all_preds)


def save_history(history: Dict[str, List[float]], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


def save_curves(history: Dict[str, List[float]], plots_dir: pathlib.Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.get('accuracy', []), label='Train Accuracy', linewidth=2)
    plt.plot(history.get('val_accuracy', []), label='Validation Accuracy', linewidth=2)
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'accuracy_curve.png', dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.get('loss', []), label='Train Loss', linewidth=2)
    plt.plot(history.get('val_loss', []), label='Validation Loss', linewidth=2)
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'loss_curve.png', dpi=140)
    plt.close()


def save_confusion_and_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], plots_dir: pathlib.Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(plots_dir / 'confusion_matrix.png', dpi=140)
    plt.close()

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(plots_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train EmotionLens CNN on FER-2013 split folders (PyTorch)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--dataset', type=str, default='Dataset')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device, gpu_name = get_device()

    project_root = pathlib.Path(__file__).resolve().parent
    dataset_dir = (project_root / args.dataset).resolve()
    models_dir = project_root / 'models'
    plots_dir = models_dir / 'plots'
    best_model_path = models_dir / 'best_model.pt'
    training_csv_path = models_dir / 'training_log.csv'
    live_metrics_path = models_dir / 'live_metrics.json'

    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f'Dataset path does not exist: {dataset_dir}')

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.Lambda(lambda x: BrightnessScale(0.8, 1.2)(x))], p=0.8),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(dataset_dir / 'train', transform=train_transform)
    test_dataset = datasets.ImageFolder(dataset_dir / 'test', transform=test_transform)

    class_map = train_dataset.class_to_idx
    print(f"[INFO] Class index mapping: {class_map}")
    print(f"[INFO] Training sample count: {len(train_dataset)}")
    print(f"[INFO] Test sample count: {len(test_dataset)}")

    if len(class_map) != 7:
        raise ValueError(f'Expected 7 classes, found {len(class_map)} classes: {class_map}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    model = EmotionLensCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    with open(training_csv_path, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['epoch', 'accuracy', 'val_accuracy', 'loss', 'val_loss', 'lr'])

    with open(live_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'current_epoch': 0,
                'total_epochs': args.epochs,
                'history': {
                    'epoch': [], 'accuracy': [], 'val_accuracy': [],
                    'loss': [], 'val_loss': [], 'lr': []
                },
                'status': 'training'
            },
            f,
        )

    results_logger = ResultsLoggerCallback(
        filepath=str(models_dir / 'training_results.txt'),
        gpu_name=gpu_name,
        total_target_epochs=args.epochs,
    )
    live_metrics = LiveMetricsCallback(filepath=str(live_metrics_path), total_epochs=args.epochs)
    results_logger.on_train_begin()

    history: Dict[str, List[float]] = {
        'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [], 'lr': []
    }

    best_val_acc = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += int((preds == labels).sum().item())
            running_total += int(labels.size(0))

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]['lr'])

        metrics = {
            'accuracy': train_acc,
            'val_accuracy': val_acc,
            'loss': train_loss,
            'val_loss': val_loss,
        }

        history['accuracy'].append(round(train_acc, 6))
        history['val_accuracy'].append(round(val_acc, 6))
        history['loss'].append(round(train_loss, 6))
        history['val_loss'].append(round(val_loss, 6))
        history['lr'].append(round(current_lr, 8))

        with open(training_csv_path, 'a', newline='', encoding='utf-8') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow([epoch + 1, train_acc, val_acc, train_loss, val_loss, current_lr])

        results_logger.on_epoch_end(epoch, metrics, current_lr)
        live_metrics.on_epoch_end(epoch, metrics, current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            best_state = {
                'state_dict': model.state_dict(),
                'class_to_idx': class_map,
                'epoch': epoch + 1,
                'val_accuracy': val_acc,
            }
            torch.save(best_state, best_model_path)
            print(f"[CHECKPOINT] Saved best model at epoch {epoch + 1} with val_acc={val_acc:.4f}")
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={current_lr:.6f}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"[EARLY STOP] No val_accuracy improvement for {args.patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state['state_dict'])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'[RESULT] Test Loss: {test_loss:.4f}')
    print(f'[RESULT] Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)')

    save_history(history, models_dir / 'history.json')
    save_curves(history, plots_dir)

    ordered_classes = [label for label, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    y_true, y_pred = predict_loader(model, test_loader, device)
    save_confusion_and_report(y_true, y_pred, ordered_classes, plots_dir)

    results_logger.on_train_end(test_loss=test_loss, test_acc=test_acc)
    live_metrics.on_train_end()


if __name__ == '__main__':
    main()
