import os
import json
import urllib.request
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment, ToPILImage, TrivialAugmentWide

import wandb

###############################################################################
# Model Definition
###############################################################################
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 100)  # 100 classes for CIFAR-100
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

###############################################################################
# Data Augmentation Utilities
###############################################################################
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size, _, H, W = x.size()
    rand_index = torch.randperm(batch_size).to(x.device)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - int(W * np.sqrt(1 - lam)) // 2, 0, W)
    bby1 = np.clip(cy - int(H * np.sqrt(1 - lam)) // 2, 0, H)
    bbx2 = np.clip(cx + int(W * np.sqrt(1 - lam)) // 2, 0, W)
    bby2 = np.clip(cy + int(H * np.sqrt(1 - lam)) // 2, 0, H)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
    y_a, y_b = y, y[rand_index]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x, y_a, y_b, lam

###############################################################################
# Learning Rate Adjustment Helper
###############################################################################
def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr):
    lr = base_lr * float(epoch + 1) / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

###############################################################################
# Training and Validation Functions
###############################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # Apply Mixup or CutMix if enabled
        if CONFIG["use_cutmix"]:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=CONFIG["cutmix_alpha"])
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        elif CONFIG["use_mixup"]:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=CONFIG["mixup_alpha"])
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})
    return running_loss / len(trainloader), 100. * correct / total

def validate(model, valloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})
    return running_loss / len(valloader), 100. * correct / total

###############################################################################
# Test-Time Augmentation (TTA) Functions
###############################################################################
def test_time_augmentation(model, image, num_augmentations=10, device='cuda'):
    model.eval()
    tta_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # Convert image tensor to PIL image if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        image = ToPILImage()(image)
    all_preds = []
    with torch.no_grad():
        orig_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        orig_tensor = orig_transform(image).unsqueeze(0).to(device)
        orig_output = model(orig_tensor)
        all_preds.append(F.softmax(orig_output, dim=1))
        for _ in range(num_augmentations - 1):
            aug_tensor = tta_transforms(image).unsqueeze(0).to(device)
            aug_output = model(aug_tensor)
            all_preds.append(F.softmax(aug_output, dim=1))
    all_preds = torch.cat(all_preds, dim=0)
    return torch.mean(all_preds, dim=0, keepdim=True)

def evaluate_with_tta(model, dataloader, device, num_augmentations=10):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    progress_bar = tqdm(dataloader, desc="[TTA Evaluation]", leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            batch_predictions = []
            for i in range(inputs.size(0)):
                image = inputs[i]
                tta_probs = test_time_augmentation(model, image, num_augmentations, device)
                batch_predictions.append(tta_probs)
            batch_probs = torch.cat(batch_predictions, dim=0)
            _, predicted = batch_probs.max(1)
            labels = labels.to(device)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy().tolist())
            progress_bar.set_postfix({"acc": 100. * correct / total})
    accuracy = 100. * correct / total
    return accuracy, all_predictions

def eval_ood_with_tta(model, CONFIG, num_augmentations=10):
    device = CONFIG["device"]
    model.eval()
    ood_datasets = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 
                    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost',
                    'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    all_predictions = {}
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    for dataset_name in ood_datasets:
        print(f"Processing {dataset_name}...")
        ood_dir = os.path.join(CONFIG["ood_dir"], dataset_name)
        ood_dataset = torchvision.datasets.ImageFolder(root=ood_dir, transform=transform_test)
        ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=CONFIG["batch_size"],
                                                   shuffle=False, num_workers=CONFIG["num_workers"])
        _, predictions = evaluate_with_tta(model, ood_loader, device, num_augmentations=num_augmentations)
        all_predictions[dataset_name] = predictions
    return all_predictions

###############################################################################
# Main Training Loop (Reverted to single phase, similar to v1)
###############################################################################
def main():
    CONFIG = {
        "model": "ResNet18_pretrained",
        "batch_size": 256,
        "learning_rate": 0.001,
        "backbone_lr": 0.0001,
        "epochs": 40, # Increased epochs, adjust as needed
        "warmup_epochs": 5,
        "num_workers": 4,
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge-improved", # Keep project name or change
        "seed": 42,
        "use_mixup": True,
        "mixup_alpha": 0.2,
        "use_cutmix": True,
        "cutmix_alpha": 1.0,
        "label_smoothing": 0.1,
        "weight_decay": 5e-4,
        # TTA parameters (can be enabled for evaluation)
        "use_tta": False,
        "tta_num_augmentations": 10,
        # Removed distillation parameters
    }
    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    # Seed everything for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    # Potentially add cuda seeding if using CUDA

    # Define transforms using TrivialAugmentWide
    # from torchvision.transforms import RandAugment # Keep commented or remove
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        TrivialAugmentWide(), # Changed from RandAugment
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # --- Data Loading ---
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                              download=True, transform=transform_train)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"],
                                              shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"],
                                            shuffle=False, num_workers=CONFIG["num_workers"])
    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False,
                                             download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"],
                                             shuffle=False, num_workers=CONFIG["num_workers"])

    # --- Model Initialization ---
    print("\n--- Initializing Model ---")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 100)
    )
    model = model.to(CONFIG["device"])
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    # --- Optimizer and Loss ---
    # Use AdamW instead of Adam
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    cosine_scheduler = None

    # --- WandB Setup ---
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG, name="Baseline_Improved") # Single run name
    wandb.watch(model, log="all", log_freq=100)
    best_val_acc = 0.0

    # --- Training Loop ---
    print("--- Starting Training ---")
    for epoch in range(CONFIG["epochs"]):
        # Learning Rate Adjustment
        if epoch < CONFIG["warmup_epochs"]:
            current_lr = adjust_learning_rate(optimizer, epoch, CONFIG["warmup_epochs"], CONFIG["learning_rate"])
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}: Warmup phase, classifier lr: {current_lr:.6f}")
        elif epoch == CONFIG["warmup_epochs"]:
            backbone_params = [param for name, param in model.named_parameters() if "fc" not in name]
            for param in backbone_params:
                param.requires_grad = True
            # Add backbone params to optimizer (AdamW example)
            # optimizer.add_param_group({'params': backbone_params, 'lr': CONFIG["backbone_lr"]})
            optimizer.add_param_group({'params': backbone_params, 'lr': CONFIG["backbone_lr"]})
            remaining_epochs = CONFIG["epochs"] - epoch
            # Consider OneCycleLR or CosineAnnealingWarmRestarts here instead of plain Cosine
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}: Unfreezing backbone; added {len(backbone_params)} parameters with lr {CONFIG['backbone_lr']}")
        elif cosine_scheduler: # Check if scheduler exists before stepping
            cosine_scheduler.step()

        # Training Step
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        current_classifier_lr = optimizer.param_groups[0]["lr"]
        current_backbone_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR (cls/bb): {current_classifier_lr:.6f}/{current_backbone_lr:.6f}")

        # Logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "classifier_lr": current_classifier_lr,
            "backbone_lr": current_backbone_lr
        })

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"*** New best validation accuracy: {best_val_acc:.2f}%. Saving model to best_model.pth ***")
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

    wandb.finish()
    print(f"--- Training Complete. Best validation accuracy: {best_val_acc:.2f}% ---")

    # --- Final Evaluation ---
    print("\n--- Starting Final Evaluation ---")
    # Load the best model saved during training
    print(f"Loading best model from best_model.pth for evaluation...")
    # Re-initialize model structure before loading state_dict
    model = resnet18(weights=None) # Don't load default weights again
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 100)
    )
    model.load_state_dict(torch.load("best_model.pth"))
    model = model.to(CONFIG["device"])
    model.eval()

    # Import evaluation scripts
    import eval_cifar100
    import eval_ood_pretrained
    from eval_ood_pretrained import create_ood_df

    if CONFIG["use_tta"]:
        print(f"Evaluating with Test Time Augmentation (TTA)...")
        # Note: evaluate_with_tta needs the model object directly
        tta_accuracy, _ = evaluate_with_tta(model, testloader, CONFIG["device"],
                                             num_augmentations=CONFIG["tta_num_augmentations"])
        print(f"TTA CIFAR-100 Test Accuracy: {tta_accuracy:.2f}%")
        # Evaluate OOD with TTA
        print("Evaluating on OOD data with TTA...")
        all_predictions_ood_tta = eval_ood_with_tta(model, CONFIG, num_augmentations=CONFIG["tta_num_augmentations"])
        submission_df_ood_tta = create_ood_df(all_predictions_ood_tta)
        submission_df_ood_tta.to_csv("submission_ood_tta.csv", index=False)
        print("submission_ood_tta.csv created successfully.")
    else:
        print(f"Evaluating standard accuracy using best_model.pth")
        # Use the modified evaluation function that takes model_path (or just pass the loaded model)
        predictions_cifar, standard_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"], model_path="best_model.pth")
        print(f"CIFAR-100 Test Accuracy: {standard_accuracy:.2f}%")

        print("Evaluating on OOD data...")
        # Use the modified evaluation function that takes model_path (or just pass the loaded model)
        all_predictions_ood = eval_ood_pretrained.evaluate_ood_test(model, CONFIG, model_path="best_model.pth")
        submission_df_ood = create_ood_df(all_predictions_ood)
        submission_df_ood.to_csv("submission_ood.csv", index=False)
        print("submission_ood.csv created successfully.")



if __name__ == '__main__':
    main()
