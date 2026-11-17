"""Train a Bangla handwritten character classifier from a subset of BanglaLekha-Isolated.

Uses pretrained Hugging Face vision backbones with transfer learning, logs at least
two training runs to MLflow, and saves the best-performing model for whole-word
inference in app.py.
"""

import argparse
import json
import os
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_to_square(image: Image.Image, fill: int = 0) -> Image.Image:
    """Pad a grayscale or RGB image to a square while preserving aspect ratio."""
    width, height = image.size
    max_dim = max(width, height)
    pad_w = (max_dim - width) // 2
    pad_h = (max_dim - height) // 2
    padding = (pad_w, pad_h, max_dim - width - pad_w, max_dim - height - pad_h)
    return ImageOps.expand(image, padding, fill=fill)


def load_dataset_subset(
    data_dir: Path,
    labels_map: dict[str, str],
    samples_per_class: int,
    seed: int,
) -> tuple[list[Path], list[int]]:
    """Sample up to `samples_per_class` images from each class folder (1..84)."""
    rng = random.Random(seed)
    image_paths: list[Path] = []
    targets: list[int] = []

    for class_id_str in sorted(labels_map.keys(), key=int):
        class_dir = data_dir / class_id_str
        if not class_dir.is_dir():
            continue
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not files:
            continue
        chosen = rng.sample(files, min(samples_per_class, len(files)))
        label_idx = int(class_id_str) - 1  # 0-indexed for PyTorch CrossEntropy
        for path in chosen:
            image_paths.append(path)
            targets.append(label_idx)

    if not image_paths:
        raise FileNotFoundError(f"No images found under {data_dir}. Check --data-dir path.")
    return image_paths, targets


class BanglaCharDataset(Dataset):
    """Dataset of isolated Bangla character images preprocessed for Hugging Face models."""

    def __init__(
        self,
        image_paths: list[Path],
        targets: list[int],
        processor: AutoImageProcessor,
        augment: bool = False,
    ) -> None:
        self.image_paths = image_paths
        self.targets = targets
        self.processor = processor
        self.augment_transform = (
            transforms.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.9, 1.1))
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("L")
        img = pad_to_square(img, fill=0).convert("RGB")
        if self.augment_transform is not None:
            img = self.augment_transform(img)
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, self.targets[idx]


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate loss and accuracy on a DataLoader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def run_training(
    run_name: str,
    model_name: str,
    freeze_backbone: bool,
    augment: bool,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    train_paths: list[Path],
    train_targets: list[int],
    val_paths: list[Path],
    val_targets: list[int],
    labels_map: dict[str, str],
    device: torch.device,
    best_model_dir: Path,
    best_overall_acc: float,
) -> float:
    """Execute one MLflow-tracked training run and return updated best validation accuracy."""
    id2label = {int(k) - 1: v for k, v in labels_map.items()}
    label2id = {v: k for k, v in id2label.items()}

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels_map),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    model.to(device)

    train_ds = BanglaCharDataset(train_paths, train_targets, processor, augment=augment)
    val_ds = BanglaCharDataset(val_paths, val_targets, processor, augment=False)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_name": model_name,
                "freeze_backbone": freeze_backbone,
                "augment": augment,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "num_classes": len(labels_map),
                "device": device.type,
            }
        )

        best_run_acc = 0.0
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            for pixel_values, labels in train_loader:
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(pixel_values=pixel_values)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                seen += labels.size(0)

            scheduler.step()
            train_loss = running_loss / max(seen, 1)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                step=epoch,
            )
            print(
                f"[{run_name}] Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )

            if val_acc > best_run_acc:
                best_run_acc = val_acc
                if val_acc > best_overall_acc:
                    best_overall_acc = val_acc
                    best_model_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(best_model_dir)
                    processor.save_pretrained(best_model_dir)

        mlflow.log_metric("best_val_accuracy", best_run_acc)
        if best_model_dir.exists():
            mlflow.log_artifacts(str(best_model_dir), artifact_path="hf_model")

    return best_overall_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Bangla character classifier with MLflow.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../BanglaLekha-Isolated/Images"),
        help="Path to directory containing class folders 1..84.",
    )
    parser.add_argument("--labels-path", type=Path, default=Path("labels.json"))
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("models/hf_model"))
    parser.add_argument("--mlflow-dir", type=Path, default=Path("artifacts/mlflow"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    with open(args.labels_path, encoding="utf-8") as f:
        labels_map: dict[str, str] = json.load(f)

    image_paths, targets = load_dataset_subset(
        args.data_dir, labels_map, args.samples_per_class, args.seed
    )
    train_paths, val_paths, train_targets, val_targets = train_test_split(
        image_paths, targets, test_size=0.2, stratify=targets, random_state=args.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Train: {len(train_paths)} | Val: {len(val_paths)}")

    args.mlflow_dir.mkdir(parents=True, exist_ok=True)
    db_path = (args.mlflow_dir / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("bangla-handwritten-recognition")

    configs = [
        {
            "run_name": "mobilenet_v2_head_baseline",
            "model_name": "google/mobilenet_v2_1.0_224",
            "freeze_backbone": True,
            "augment": False,
            "learning_rate": 1e-3,
            "epochs": max(3, args.epochs - 1),
        },
        {
            "run_name": "resnet18_finetuned_aug",
            "model_name": "microsoft/resnet-18",
            "freeze_backbone": False,
            "augment": True,
            "learning_rate": 3e-4,
            "epochs": args.epochs,
        },
    ]

    best_acc = 0.0
    for cfg in configs:
        best_acc = run_training(
            run_name=cfg["run_name"],
            model_name=cfg["model_name"],
            freeze_backbone=cfg["freeze_backbone"],
            augment=cfg["augment"],
            learning_rate=cfg["learning_rate"],
            epochs=cfg["epochs"],
            batch_size=args.batch_size,
            train_paths=train_paths,
            train_targets=train_targets,
            val_paths=val_paths,
            val_targets=val_targets,
            labels_map=labels_map,
            device=device,
            best_model_dir=args.output_dir,
            best_overall_acc=best_acc,
        )

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    print(f"Best model saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
