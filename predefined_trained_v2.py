import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import wandb
import json
import copy
from torch.cuda.amp import autocast, GradScaler

################################################################################
# Feature Pyramid Network Implementation
################################################################################
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
        
    def forward(self, x):
        # x is a list of features from different layers
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, x)]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode='nearest'
            )
        
        # Apply 3x3 conv on each merged feature
        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        
        return outs

################################################################################
# Enhanced ResNet18 with FPN
################################################################################
class EnhancedResNet18(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(EnhancedResNet18, self).__init__()
        # Load pretrained ResNet18
        self.resnet = resnet18(pretrained=pretrained)
        
        # Remove the original FC layer
        self.resnet.fc = nn.Identity()
        
        # Define the FPN layers
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256, 512],  # ResNet18 channel sizes
            out_channels=256
        )
        
        # Improved classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # For progressive unfreezing, we need to access the layers
        self.layer_groups = [
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        ]
        
    def forward(self, x):
        # Extract features at different levels of ResNet
        feat0 = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        x = self.resnet.maxpool(feat0)
        
        feat1 = self.resnet.layer1(x)
        feat2 = self.resnet.layer2(feat1)
        feat3 = self.resnet.layer3(feat2)
        feat4 = self.resnet.layer4(feat3)
        
        # Apply FPN
        fpn_features = self.fpn([feat1, feat2, feat3, feat4])
        
        # Use the highest level feature for classification
        out = self.classifier(fpn_features[-1])
        return out

################################################################################
# L2SP Regularization (regularization for transfer learning)
################################################################################
class L2SP:
    def __init__(self, model, pretrained_model, alpha=0.1, beta=0.01):
        self.model = model
        self.pretrained_model = pretrained_model
        self.alpha = alpha  # weight for L2-SP regularization
        self.beta = beta    # weight for L2 regularization
        
    def __call__(self):
        # L2-SP regularization: penalize deviation from pretrained weights
        l2_sp_reg = 0
        l2_reg = 0
        
        for (name, param), (_, param_pretrained) in zip(
            self.model.named_parameters(), self.pretrained_model.named_parameters()
        ):
            # Skip batch norm parameters and classifier
            if 'bn' not in name and 'classifier' not in name and param.requires_grad:
                l2_sp_reg += torch.sum((param - param_pretrained) ** 2)
            
            # Regular L2 regularization for all parameters
            if param.requires_grad:
                l2_reg += torch.sum(param ** 2)
        
        return self.alpha * l2_sp_reg + self.beta * l2_reg

################################################################################
# Mixup and CutMix functions (keeping from original code)
################################################################################
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size, _, H, W = x.size()
    rand_index = torch.randperm(batch_size).to(x.device)
    # Define the patch size
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - int(W * np.sqrt(1 - lam)) // 2, 0, W)
    bby1 = np.clip(cy - int(H * np.sqrt(1 - lam)) // 2, 0, H)
    bbx2 = np.clip(cx + int(W * np.sqrt(1 - lam)) // 2, 0, W)
    bby2 = np.clip(cy + int(H * np.sqrt(1 - lam)) // 2, 0, H)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
    y_a, y_b = y, y[rand_index]
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x, y_a, y_b, lam

################################################################################
# Consistency Regularization
################################################################################
def consistency_loss(model, x, temperature=1.0, alpha=0.5):
    # Create two different augmented views
    # We'll use simple random cropping and flipping for this example
    batch_size = x.size(0)
    
    # Create random augmentations
    transform1 = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip()
    ])
    
    transform2 = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip()
    ])
    
    # Apply augmentations
    x1 = transform1(x)
    x2 = transform2(x)
    
    # Get model predictions
    with torch.no_grad():
        logits1 = model(x1) / temperature
        probs1 = F.softmax(logits1, dim=1)
    
    # Get predictions for the second view
    logits2 = model(x2) / temperature
    
    # Calculate consistency loss (KL divergence)
    loss = F.kl_div(
        F.log_softmax(logits2, dim=1),
        probs1,
        reduction='batchmean'
    )
    
    return alpha * loss

################################################################################
# Enhanced Training Function with Gradient Accumulation
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG, l2sp_reg=None, scaler=None):
    """Train one epoch with gradient accumulation and consistency regularization."""
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Gradient accumulation steps
    accumulation_steps = CONFIG.get("accumulation_steps", 1)
    
    # Reset gradients initially
    optimizer.zero_grad()

    # Put the trainloader iterator in a tqdm so it can print progress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Mixed precision training
        with autocast(enabled=CONFIG.get("use_mixed_precision", False)):
            # --- Handle data augmentation ---
            if CONFIG["use_cutmix"] and np.random.random() < 0.5:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=CONFIG["cutmix_alpha"])
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            elif CONFIG["use_mixup"] and np.random.random() < 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=CONFIG["mixup_alpha"])
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Add consistency regularization loss if enabled
            if CONFIG.get("use_consistency", False):
                cons_loss = consistency_loss(model, inputs, 
                                           temperature=CONFIG.get("consistency_temp", 1.0),
                                           alpha=CONFIG.get("consistency_alpha", 0.5))
                loss += cons_loss
            
            # Add L2SP regularization if provided
            if l2sp_reg is not None:
                loss += l2sp_reg()
            
            # Scale the loss for gradient accumulation
            loss = loss / accumulation_steps
        
        # Backward pass with mixed precision
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Only step and zero grad after accumulation or at the end of loader
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(trainloader):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)

        # For accuracy calculation, we need to handle mixup/cutmix
        if CONFIG["use_cutmix"] or CONFIG["use_mixup"]:
            # For mixup/cutmix, we'll use original labels for simplicity
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        else:
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

################################################################################
# Test Time Augmentation Function
################################################################################
def test_with_tta(model, testloader, criterion, device, num_augments=5):
    """Evaluate model with test-time augmentation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Define TTA transforms
    tta_transforms = [
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        transforms.Compose([
            transforms.Resize(256), 
            transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
    ]
    
    with torch.no_grad():
        progress_bar = tqdm(testloader, desc="[Test with TTA]", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            total += labels.size(0)
            
            # Original prediction
            outputs = model(inputs)
            
            # Add TTA predictions
            for i in range(min(num_augments, len(tta_transforms))):
                transform = tta_transforms[i]
                # Apply transform to each image individually
                tta_outputs = []
                for j in range(inputs.size(0)):
                    # Convert tensor to PIL and back
                    img = transforms.ToPILImage()(inputs[j].cpu())
                    aug_img = transform(img).unsqueeze(0).to(device)
                    tta_outputs.append(model(aug_img))
                
                # Combine TTA outputs
                tta_output = torch.cat(tta_outputs, dim=0)
                outputs += tta_output
            
            # Average predictions
            outputs /= (num_augments + 1)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({"loss": running_loss / (progress_bar.n + 1), 
                                     "acc": 100. * correct / total})
    
    test_loss = running_loss / len(testloader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

################################################################################
# Self-distillation Function
################################################################################
def self_distillation(teacher_model, student_model, trainloader, optimizer, CONFIG):
    """Apply self-distillation from teacher to student model."""
    device = CONFIG["device"]
    student_model.train()
    teacher_model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Temperature for knowledge distillation
    T = CONFIG.get("distillation_temp", 4.0)
    # Alpha for balancing distillation and CE loss
    alpha = CONFIG.get("distillation_alpha", 0.9)
    
    # Hard loss
    criterion_ce = nn.CrossEntropyLoss()
    # Soft loss
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    
    progress_bar = tqdm(trainloader, desc="[Self-Distillation]", leave=False)
    
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Get teacher predictions (no grad)
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
            teacher_probs = F.softmax(teacher_outputs / T, dim=1)
        
        # Get student predictions
        student_outputs = student_model(inputs)
        
        # Compute hard loss (student prediction vs ground truth)
        hard_loss = criterion_ce(student_outputs, labels)
        
        # Compute soft loss (student prediction vs teacher prediction)
        soft_loss = criterion_kl(
            F.log_softmax(student_outputs / T, dim=1),
            teacher_probs
        ) * (T * T)
        
        # Total loss
        loss = (1 - alpha) * hard_loss + alpha * soft_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = student_outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({
            "loss": running_loss / (i + 1), 
            "acc": 100. * correct / total
        })
    
    distill_loss = running_loss / len(trainloader)
    distill_acc = 100. * correct / total
    return distill_loss, distill_acc

################################################################################
# Main function with all improvements
################################################################################
def main():
    ############################################################################
    #    Configuration Dictionary with New Parameters
    ############################################################################
    CONFIG = {
        "model": "EnhancedResNet18_FPN",
        "batch_size": 128,  # Reduced to account for more complex model
        "learning_rate": 0.001,
        "backbone_lr": 0.0001,
        "epochs": 40,
        "warmup_epochs": 5,
        "num_workers": 4,
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge-improved",
        "seed": 42,
        
        # Mixup/CutMix Params
        "use_mixup": True,
        "mixup_alpha": 0.2,
        "use_cutmix": True,
        "cutmix_alpha": 1.0,
        
        # NEW PARAMETERS
        # Progressive unfreezing config
        "progressive_unfreezing": True,
        "unfreeze_schedule": [10, 15, 20, 25],  # Epochs at which to unfreeze layer groups
        
        # FPN config
        "use_fpn": True,
        
        # Gradient accumulation
        "accumulation_steps": 4,  # Accumulate gradients over 4 batches
        
        # L2SP regularization
        "use_l2sp": True,
        "l2sp_alpha": 0.1,
        "l2sp_beta": 0.01,
        
        # Consistency regularization
        "use_consistency": True,
        "consistency_alpha": 0.5,
        "consistency_temp": 1.0,
        
        # Mixed precision training
        "use_mixed_precision": True,
        
        # Cosine annealing with restarts
        "use_sgdr": True,
        "sgdr_T_0": 10,  # Initial restart interval
        "sgdr_T_mult": 2,  # Multiply factor for subsequent restarts
        "sgdr_eta_min": 1e-6,  # Minimum learning rate
        
        # Test-time augmentation
        "use_tta": True,
        "tta_num_augments": 5,
        
        # Self-distillation
        "use_self_distillation": True,
        "distillation_temp": 4.0,
        "distillation_alpha": 0.9,
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)
    
    # Set random seed for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG["seed"])

    ############################################################################
    #    Enhanced Data Transformations with AugMix
    ############################################################################
    from torchvision.transforms import AutoAugment, AugMix, RandAugment, InterpolationMode

    # Training transforms with AugMix
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        AugMix(),  # AugMix for better robustness
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Validation and test transforms (NO augmentation)
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    ############################################################################
    #    Data Loading (same as original)
    ############################################################################
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)

    # Split train into train and validation (80/20 split)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    # Create data loaders for training and validation
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], 
                                            shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"], 
                                           shuffle=False, num_workers=CONFIG["num_workers"])

    # Load CIFAR-100 test set with test transforms (no augmentation)
    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False,
                                          download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"],
                                           shuffle=False, num_workers=CONFIG["num_workers"])

    ############################################################################
    #    Instantiate enhanced model with FPN
    ############################################################################
    if CONFIG["use_fpn"]:
        model = EnhancedResNet18(num_classes=100, pretrained=True)
    else:
        # Fallback to original approach if FPN is disabled
        model = resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 100)
        )
    
    # Save a copy of the pretrained model for L2SP regularization
    pretrained_model = copy.deepcopy(model)
    pretrained_model.eval()
    for param in pretrained_model.parameters():
        param.requires_grad = False
        
    # Move model to device
    model = model.to(CONFIG["device"])
    pretrained_model = pretrained_model.to(CONFIG["device"])
    
    # Initialize L2SP regularization if enabled
    l2sp_reg = L2SP(model, pretrained_model, 
                   alpha=CONFIG["l2sp_alpha"], 
                   beta=CONFIG["l2sp_beta"]) if CONFIG["use_l2sp"] else None

    print("\nModel summary:")
    print(f"{model}\n")

    ############################################################################
    #    Loss Function, Optimizer and Discriminative Learning Rates
    ############################################################################
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Increased from 0.05
    
    # Create parameter groups for discriminative learning rates
    if isinstance(model, EnhancedResNet18) and CONFIG["progressive_unfreezing"]:
        # Initially, only train the classifier
        for param in model.parameters():
            param.requires_grad = False
        
        # Enable gradients for the classifier
        for param in model.classifier.parameters():
            param.requires_grad = True
            
        # Set up optimizer with only classifier parameters
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(), 
            lr=CONFIG["learning_rate"],
            weight_decay=1e-4
        )
    else:
        # If not using progressive unfreezing, use simplified approach
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=1e-4
        )
    
    # Initialize mixed precision scaler if enabled
    scaler = GradScaler() if CONFIG["use_mixed_precision"] and torch.cuda.is_available() else None
    
    # Initialize cosine annealing with restarts if enabled
    if CONFIG["use_sgdr"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=CONFIG["sgdr_T_0"],
            T_mult=CONFIG["sgdr_T_mult"],
            eta_min=CONFIG["sgdr_eta_min"]
        )
    else:
        # Fallback to regular cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=CONFIG["epochs"],
            eta_min=1e-6
        )

    # Initialize wandb
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)

    ############################################################################
    #    Training Loop with Progressive Unfreezing
    ############################################################################
    best_val_acc = 0.0
    best_model_state = None
    snapshot_models = []  # For snapshot ensembling
    
    # Define layer groups for progressive unfreezing (if using EnhancedResNet18)
    if isinstance(model, EnhancedResNet18) and CONFIG["progressive_unfreezing"]:
        layer_groups = [
            model.resnet.layer1,
            model.resnet.layer2,
            model.resnet.layer3,
            model.resnet.layer4
        ]
    else:
        # Fallback for regular ResNet18
        layer_groups = []
        if hasattr(model, 'layer1'):
            layer_groups.extend([model.layer1, model.layer2, model.layer3, model.layer4])

    for epoch in range(CONFIG["epochs"]):
        # Progressive unfreezing based on schedule
        if CONFIG["progressive_unfreezing"] and epoch in CONFIG["unfreeze_schedule"] and layer_groups:
            # Unfreeze the next layer group
            layer_idx = CONFIG["unfreeze_schedule"].index(epoch)
            if layer_idx < len(layer_groups):
                layer_to_unfreeze = layer_groups[layer_idx]
                print(f"Epoch {epoch+1}: Unfreezing layer group {layer_idx+1}")
                
                # Unfreeze the layer
                for param in layer_to_unfreeze.parameters():
                    param.requires_grad = True
                
                # Add the unfrozen parameters to the optimizer with a lower learning rate
                # The deeper the layer, the lower the learning rate
                lr_factor = 0.1 ** (len(layer_groups) - layer_idx)
                optimizer.add_param_group({
                    'params': layer_to_unfreeze.parameters(),
                    'lr': CONFIG["backbone_lr"] * lr_factor
                })
        
        # Train one epoch
        train_loss, train_acc = train(
            epoch, model, trainloader, optimizer, criterion, CONFIG, 
            l2sp_reg=l2sp_reg, scaler=scaler
        )
        
        # Validate
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")
        
        # Save snapshot for ensembling at SGDR restart points
        if CONFIG["use_sgdr"] and (epoch + 1) % CONFIG["sgdr_T_0"] == 0:
            print(f"Saving model snapshot at epoch {epoch+1}")
            snapshot_models.append(copy.deepcopy(model.state_dict()))
            torch.save(model.state_dict(), f"snapshot_model_epoch_{epoch+1}.pth")
    
    # Load the best model for final evaluation
    model.load_state_dict(best_model_state)
    
    ############################################################################
    #    Self-Distillation Phase (if enabled)
    ############################################################################
    if CONFIG["use_self_distillation"]:
        teacher_model = resnet18(pretrained=True)
        num_ftrs = teacher_model.fc.in_features
        teacher_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 100)
        )
        teacher_model = teacher_model.to(CONFIG["device"])
        # Freeze teacher model
        for param in teacher_model.parameters():
            param.requires_grad = False

        # We will implement a manual warmup for the classifier.
        def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr):
            """Linearly increase lr for the classifier during warmup."""
            lr = base_lr * float(epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return lr

        # Use cosine annealing for epochs after warmup.
        cosine_scheduler = None  # Will be reinitialized after warmup.

        for epoch in range(CONFIG["epochs"]):
            if epoch < CONFIG["warmup_epochs"]:
                # Warmup phase: adjust lr for fc parameters only.
                current_lr = adjust_learning_rate(optimizer, epoch, CONFIG["warmup_epochs"], CONFIG["learning_rate"])
                print(f"Epoch {epoch+1}: Warmup phase, classifier lr: {current_lr:.6f}")
            elif epoch == CONFIG["warmup_epochs"]:
                # End of warmup: unfreeze backbone and add them to the optimizer with a lower lr.
                backbone_params = [param for name, param in model.named_parameters() if "fc" not in name]
                for param in backbone_params:
                    param.requires_grad = True
                optimizer.add_param_group({'params': backbone_params, 'lr': CONFIG["backbone_lr"]})
                # Reinitialize cosine annealing scheduler for the remaining epochs.
                remaining_epochs = CONFIG["epochs"] - epoch
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
                print(f"Epoch {epoch+1}: Unfreezing backbone; added {len(backbone_params)} parameters with lr {CONFIG['backbone_lr']}")
            else:
                # Use cosine annealing scheduler to update learning rates.
                cosine_scheduler.step()

            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

            for i, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])

                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                if CONFIG["use_self_distillation"] and teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = teacher_model(inputs)
                    
                    # Combine hard and soft labels
                    hard_loss = criterion(outputs, labels)
                    soft_loss = F.kl_div(
                        F.log_softmax(outputs / CONFIG["temperature"], dim=1),
                        F.softmax(teacher_outputs / CONFIG["temperature"], dim=1),
                        reduction='batchmean'
                    ) * (CONFIG["temperature"] ** 2)
                    
                    loss = (1 - CONFIG["alpha"]) * hard_loss + CONFIG["alpha"] * soft_loss
                else:
                    # Regular training without distillation
                    if CONFIG["use_mixup"]:
                        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=CONFIG["mixup_alpha"])
                        outputs = model(inputs)
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    else:
                        loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

            train_loss = running_loss / len(trainloader)
            train_acc = 100. * correct / total

            # Validation phase
            val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            # Log metrics
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"]
            })

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_model.pth")
                wandb.save("best_model.pth")
                
                # Update teacher model if using self-distillation
                if CONFIG["use_self_distillation"]:
                    teacher_model.load_state_dict(model.state_dict())
        
    wandb.finish()

    ############################################################################
    # Evaluation
    ############################################################################
    import eval_cifar100
    import eval_ood_pretrained

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood_pretrained.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood_pretrained.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()