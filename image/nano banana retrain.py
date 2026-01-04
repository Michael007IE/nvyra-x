#!/usr/bin/env python3
"""
RETRAIN NANO DETECTOR - FIX HIGH FPR ISSUE
==========================================
Retrains Nano Banana Pro detector with more diverse real images to fix 47% FPR.

PROBLEM: Original Nano had 47% FPR on cars/photos (overfitted to artistic images)
SOLUTION: Add more diverse real images (cars, photos, objects, scenes)

New real image distribution:
- 20 Cars (photographic)
- 20 Food (photographic)  
- 20 Nature/Objects (photographic)
- 20 WikiArt (artistic - keep some)
- 10 Portraits (photographic)
- 10 Pokemon (sprites)

Total: 100 real images (vs 50 before) - DOUBLED!
Focus: More photographic variety to reduce FPR

Version: 2.0 - Nano Fix
Date: 2026-01-02
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("🔧 NANO DETECTOR RETRAINING - FIX HIGH FPR")
print("="*100)
print("\n🎯 Goal: Reduce FPR from 47% to <10%")
print("\n📊 Strategy:")
print("   1. DOUBLE real images (50 → 100)")
print("   2. Add diverse photographic content (cars, food, nature)")
print("   3. Keep advanced training techniques")
print("   4. Validate on same datasets that failed\n")

# Install dependencies
print("📦 Installing dependencies...")
os.system("pip install -q datasets transformers pillow scikit-learn matplotlib torch torchvision")
print("✅ Dependencies installed!\n")

os.environ['HF_TOKEN'] = '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'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import ViTModel, ViTImageProcessor, get_cosine_schedule_with_warmup
from datasets import load_dataset

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for Nano retraining"""
    
    # Paths
    OUTPUT_DIR = Path("/home/zeus/nano_detector_retrained")
    MODEL_DIR = OUTPUT_DIR / "nano_detector"
    LOGS_DIR = OUTPUT_DIR / "logs"
    VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
    
    # Device
    RANDOM_SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    PIN_MEMORY = True
    NUM_EPOCHS = 15
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.1
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    LABEL_SMOOTHING = 0.1
    MIXED_PRECISION = True
    USE_BFLOAT16 = True
    EARLY_STOPPING_PATIENCE = 5
    
    # Advanced techniques
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    MIXUP_PROB = 0.5
    USE_TTA = True
    
    # Model architecture
    BASE_MODEL = "WinKawaks/vit-small-patch16-224"
    HIDDEN_SIZE = 384
    DROPOUT_1 = 0.3
    DROPOUT_2 = 0.4
    
    # Augmentation
    AUG_HORIZONTAL_FLIP = 0.5
    AUG_ROTATION_DEGREES = 15
    AUG_ROTATION_PROB = 0.3
    AUG_COLOR_JITTER_PROB = 0.4
    AUG_BRIGHTNESS = 0.1
    AUG_CONTRAST = 0.1
    AUG_SATURATION = 0.1
    AUG_CROP_PROB = 0.2
    AUG_CROP_SCALE = (0.8, 1.0)
    AUG_BLUR_PROB = 0.1
    
    # Data splits
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # NEW: More diverse real images!
    SAMPLES = {
        'nano': 200,       # All Nano images
        'flux': 100,       # FLUX negatives
        'sdxl': 100,       # SDXL negatives
        'seedream': 50,    # Other AI
        'imagegbt': 50,    # Other AI
        'cars': 20,        # NEW: Photographic cars
        'food': 20,        # NEW: Photographic food
        'pokemon': 20,     # Sprites
        'wikiart': 20,     # Artistic (keep some)
        'portraits': 10,   # Photographic portraits
    }

# Create directories
for dir_path in [Config.OUTPUT_DIR, Config.MODEL_DIR, Config.LOGS_DIR, Config.VISUALIZATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Set seeds
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(Config.RANDOM_SEED)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✅ TF32 enabled")

# ============================================================================
# ADVANCED AUGMENTATION
# ============================================================================

class AdvancedAugmentation:
    """Advanced image augmentation for training"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply random augmentations"""
        
        # Horizontal flip
        if random.random() < self.config.AUG_HORIZONTAL_FLIP:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Rotation
        if random.random() < self.config.AUG_ROTATION_PROB:
            angle = random.uniform(-self.config.AUG_ROTATION_DEGREES, 
                                  self.config.AUG_ROTATION_DEGREES)
            image = image.rotate(angle, resample=Image.BICUBIC, fillcolor=(128, 128, 128))
        
        # Color jitter
        if random.random() < self.config.AUG_COLOR_JITTER_PROB:
            # Brightness
            if random.random() < 0.5:
                enhancer = ImageEnhance.Brightness(image)
                factor = 1.0 + random.uniform(-self.config.AUG_BRIGHTNESS, 
                                             self.config.AUG_BRIGHTNESS)
                image = enhancer.enhance(factor)
            
            # Contrast
            if random.random() < 0.5:
                enhancer = ImageEnhance.Contrast(image)
                factor = 1.0 + random.uniform(-self.config.AUG_CONTRAST, 
                                             self.config.AUG_CONTRAST)
                image = enhancer.enhance(factor)
            
            # Saturation
            if random.random() < 0.5:
                enhancer = ImageEnhance.Color(image)
                factor = 1.0 + random.uniform(-self.config.AUG_SATURATION, 
                                             self.config.AUG_SATURATION)
                image = enhancer.enhance(factor)
        
        # Random crop and resize
        if random.random() < self.config.AUG_CROP_PROB:
            w, h = image.size
            scale = random.uniform(*self.config.AUG_CROP_SCALE)
            new_w, new_h = int(w * scale), int(h * scale)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            image = image.crop((left, top, left + new_w, top + new_h))
            image = image.resize((w, h), Image.BICUBIC)
        
        # Gaussian blur
        if random.random() < self.config.AUG_BLUR_PROB:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
        
        return image

# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal loss for handling hard examples"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# MIXUP
# ============================================================================

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixed loss for mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class DetectorModel(nn.Module):
    """ViT-Small based detector with advanced configuration"""
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.vit = ViTModel.from_pretrained(config.BASE_MODEL)
        
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, 192),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_1),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_2),
            nn.Linear(96, 2)
        )
    
    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        return logits

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class TrainingLogger:
    """Comprehensive logging system"""
    
    def __init__(self, log_dir: Path, model_name: str):
        self.log_dir = log_dir
        self.model_name = model_name
        self.log_file = log_dir / f"{model_name}_training.log"
        
        with open(self.log_file, 'w') as f:
            f.write(f"="*80 + "\n")
            f.write(f"Training Log for {model_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"="*80 + "\n\n")
    
    def log(self, message: str, print_console: bool = True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        if print_console:
            print(message)

class MetricsTracker:
    """Track and visualize training metrics"""
    
    def __init__(self, model_name: str, save_dir: Path):
        self.model_name = model_name
        self.save_dir = save_dir
        self.metrics = defaultdict(list)
    
    def add_metric(self, metric_name: str, value: float):
        self.metrics[metric_name].append(value)
    
    def add_metrics(self, metrics_dict: Dict):
        for name, value in metrics_dict.items():
            self.add_metric(name, value)
    
    def save_metrics(self, save_path: Path = None):
        if save_path is None:
            save_path = self.save_dir / f"{self.model_name}_metrics.json"
        
        with open(save_path, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)

# ============================================================================
# PYTORCH DATASET
# ============================================================================

class DetectorDataset(Dataset):
    """PyTorch dataset with advanced augmentation"""
    
    def __init__(self, images: List[Image.Image], labels: List[int], 
                 processor: ViTImageProcessor, config: Config, augment: bool = False):
        self.images = images
        self.labels = labels
        self.processor = processor
        self.config = config
        self.augment = augment
        
        if augment:
            self.augmentation = AdvancedAugmentation(config)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].copy()
        label = self.labels[idx]
        
        if self.augment:
            image = self.augmentation(image)
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ============================================================================
# DATA SPLITTING
# ============================================================================

def stratified_split(images: List, labels: List, sources: List,
                    train_ratio: float, val_ratio: float, test_ratio: float,
                    random_seed: int = 42) -> Tuple:
    """Stratified split by source to maintain distribution"""
    
    np.random.seed(random_seed)
    
    source_groups = defaultdict(list)
    for i, source in enumerate(sources):
        source_groups[source].append(i)
    
    train_indices, val_indices, test_indices = [], [], []
    
    print("\n📊 Stratified Split by Source:")
    for source, indices in source_groups.items():
        np.random.shuffle(indices)
        
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train+n_val])
        test_indices.extend(indices[n_train+n_val:])
        
        print(f"   {source:15s}: {n_train:3d} train | {len(indices[n_train:n_train+n_val]):2d} val | {len(indices[n_train+n_val:]):2d} test")
    
    print(f"\n   Total:")
    print(f"      Train: {len(train_indices)}")
    print(f"      Val:   {len(val_indices)}")
    print(f"      Test:  {len(test_indices)}")
    
    train_imgs = [images[i] for i in train_indices]
    train_lbls = [labels[i] for i in train_indices]
    
    val_imgs = [images[i] for i in val_indices]
    val_lbls = [labels[i] for i in val_indices]
    
    test_imgs = [images[i] for i in test_indices]
    test_lbls = [labels[i] for i in test_indices]
    
    return (train_imgs, train_lbls), (val_imgs, val_lbls), (test_imgs, test_lbls)

def create_augmented_copies(images: List[Image.Image], num_copies: int, 
                           augmentation: AdvancedAugmentation) -> List[Image.Image]:
    """Create augmented copies of images"""
    augmented = []
    for img in images[:num_copies]:
        augmented.append(augmentation(img.copy()))
    return augmented

# ============================================================================
# TRAINER
# ============================================================================

class AdvancedTrainer:
    """Advanced training pipeline"""
    
    def __init__(self, model_name: str, train_loader, val_loader, test_loader, 
                 save_dir: Path, config: Config):
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_dir = save_dir
        self.config = config
        
        self.logger = TrainingLogger(config.LOGS_DIR, model_name)
        self.metrics_tracker = MetricsTracker(model_name, save_dir)
        
        self.model = DetectorModel(config).to(config.DEVICE)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.log(f"Model initialized: {model_name}")
        self.logger.log(f"Total parameters: {total_params/1e6:.1f}M")
        
        print(f"\n📊 Model Parameters: {total_params/1e6:.1f}M (ViT-Small)")
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        total_steps = len(train_loader) * config.NUM_EPOCHS
        warmup_steps = int(total_steps * config.WARMUP_RATIO)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
        
        if config.USE_FOCAL_LOSS:
            self.criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
            print(f"Loss: Focal Loss (alpha={config.FOCAL_ALPHA}, gamma={config.FOCAL_GAMMA})")
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        
        self.scaler = GradScaler() if config.MIXED_PRECISION else None
        
        self.metrics = defaultdict(list)
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            pixel_values = batch['pixel_values'].to(self.config.DEVICE)
            labels = batch['labels'].to(self.config.DEVICE)
            
            if self.config.USE_MIXUP and random.random() < self.config.MIXUP_PROB:
                pixel_values, labels_a, labels_b, lam = mixup_data(
                    pixel_values, labels, self.config.MIXUP_ALPHA
                )
                
                if self.config.MIXED_PRECISION:
                    dtype = torch.bfloat16 if self.config.USE_BFLOAT16 else torch.float16
                    with autocast(dtype=dtype):
                        logits = self.model(pixel_values)
                        loss = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(pixel_values)
                    loss = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_a.cpu().numpy())
                
            else:
                if self.config.MIXED_PRECISION:
                    dtype = torch.bfloat16 if self.config.USE_BFLOAT16 else torch.float16
                    with autocast(dtype=dtype):
                        logits = self.model(pixel_values)
                        loss = self.criterion(logits, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(pixel_values)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                acc = accuracy_score(all_labels, all_preds)
                print(f"   [{self.model_name}] Epoch {epoch+1} | Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, loader, use_tta: bool = False):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                pixel_values = batch['pixel_values'].to(self.config.DEVICE)
                labels = batch['labels'].to(self.config.DEVICE)
                
                logits = self.model(pixel_values)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr
        }
    
    def train(self):
        print(f"\n{'='*100}")
        print(f"🚀 TRAINING {self.model_name.upper()} (ADVANCED)")
        print(f"{'='*100}\n")
        
        training_start = time.time()
        
        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start = time.time()
            
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} started")
            
            train_loss = self.train_epoch(epoch)
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            val_metrics = self.evaluate(self.val_loader, use_tta=False)
            
            epoch_time = time.time() - epoch_start
            
            print(f"\n📊 Epoch {epoch+1}/{self.config.NUM_EPOCHS} ({epoch_time:.1f}s):")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_metrics['loss']:.4f}")
            print(f"   Val Acc:    {val_metrics['accuracy']:.4f}")
            print(f"   Val F1:     {val_metrics['f1']:.4f}")
            print(f"   Val FPR:    {val_metrics['fpr']:.4f}")
            
            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_loss'].append(val_metrics['loss'])
            self.metrics['val_accuracy'].append(val_metrics['accuracy'])
            self.metrics['val_f1'].append(val_metrics['f1'])
            self.metrics['val_fpr'].append(val_metrics['fpr'])
            
            self.metrics_tracker.add_metrics({
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'val_fpr': val_metrics['fpr']
            })
            
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                
                torch.save(self.model.state_dict(), self.save_dir / "best_model.pt")
                print(f"   🏆 New best! Accuracy: {self.best_val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
            
            print()
        
        training_time = time.time() - training_start
        
        print(f"\n{'='*100}")
        print(f"🧪 FINAL TEST EVALUATION")
        print(f"{'='*100}\n")
        
        self.model.load_state_dict(torch.load(self.save_dir / "best_model.pt"))
        test_metrics = self.evaluate(self.test_loader, use_tta=self.config.USE_TTA)
        
        print(f"📊 Test Results:")
        print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall:    {test_metrics['recall']:.4f}")
        print(f"   F1 Score:  {test_metrics['f1']:.4f}")
        print(f"   FPR:       {test_metrics['fpr']:.4f}")
        
        with open(self.save_dir / "test_results.json", 'w') as f:
            json.dump({
                'accuracy': float(test_metrics['accuracy']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'f1': float(test_metrics['f1']),
                'fpr': float(test_metrics['fpr']),
                'training_time_minutes': training_time / 60
            }, f, indent=2)
        
        self.metrics_tracker.save_metrics()
        
        print(f"\n⏱️  Training time: {training_time/60:.1f} minutes")
        
        return test_metrics

# ============================================================================
# ENHANCED DATASET DOWNLOADER
# ============================================================================

class EnhancedDatasetDownloader:
    """Downloads images with better error handling"""
    
    def __init__(self):
        self.min_size = 512
    
    def download_dataset(self, name: str, split: str, num_samples: int, 
                        skip: int = 0, dataset_key: str = None) -> Tuple[List[Image.Image], str]:
        """Download images using streaming"""
        
        print(f"\n{'='*80}")
        print(f"[{dataset_key.upper() if dataset_key else 'DATASET'}]")
        print(f"{'='*80}")
        print(f"📥 Downloading: {name}")
        print(f"   Samples needed: {num_samples}")
        
        try:
            dataset = load_dataset(
                name, 
                split=split,
                streaming=True,
                token=os.environ.get('HF_TOKEN')
            )
            print(f"   ✅ Streaming mode enabled")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return [], dataset_key or name
        
        images = []
        checked = 0
        max_checks = num_samples * 5
        
        print(f"   Extracting {num_samples} images...")
        
        for item in dataset:
            if checked < skip:
                checked += 1
                continue
            
            if checked >= skip + max_checks:
                break
            
            try:
                img = None
                for field in ['image', 'img', 'picture', 'photo']:
                    if field in item and isinstance(item[field], Image.Image):
                        img = item[field]
                        break
                
                if img:
                    w, h = img.size
                    if w >= self.min_size and h >= self.min_size:
                        img_resized = img.resize((1024, 1024), Image.BICUBIC)
                        images.append(img_resized.convert('RGB'))
                        
                        if len(images) >= num_samples:
                            break
                
                checked += 1
                    
            except Exception as e:
                continue
        
        print(f"   ✅ Extracted {len(images)} images")
        
        gc.collect()
        
        return images, dataset_key or name
    
    def download_all_for_nano(self) -> Dict:
        """Download all datasets for Nano retraining"""
        
        print("\n" + "="*100)
        print("📦 DOWNLOADING DATASETS - NANO RETRAINING")
        print("="*100)
        print("\n⚡ NEW STRATEGY: 100 diverse real images (vs 50 before)")
        print("   Focus: More photographic variety to reduce FPR\n")
        
        all_datasets = {}
        start_time = time.time()
        
        # AI Generated
        print("🤖 AI-GENERATED DATASETS:")
        
        datasets_to_download = [
            ('ash12321/flux-1-dev-generated-10k', 'train', Config.SAMPLES['flux'], 0, 'flux'),
            ('ash12321/sdxl-generated-10k', 'train', Config.SAMPLES['sdxl'], 0, 'sdxl'),
            ('ash12321/nano-banana-pro-generated-1k', 'train', Config.SAMPLES['nano'], 0, 'nano'),
            ('ash12321/seedream-4.5-generated-2k', 'train', Config.SAMPLES['seedream'], 0, 'seedream'),
            ('ash12321/imagegbt-1.5-generated-1k', 'train', Config.SAMPLES['imagegbt'], 0, 'imagegbt'),
        ]
        
        for name, split, num, skip, key in datasets_to_download:
            images, _ = self.download_dataset(name, split, num, skip, key)
            all_datasets[key] = (images, key)
        
        # Real Images
        print("\n📷 REAL IMAGE DATASETS (DIVERSE):")
        
        real_datasets = [
            ('tanganke/stanford_cars', 'train', Config.SAMPLES['cars'], 0, 'cars'),
            ('food101', 'train', Config.SAMPLES['food'], 0, 'food'),
            ('huggan/pokemon', 'train', Config.SAMPLES['pokemon'], 0, 'pokemon'),
            ('huggan/wikiart', 'train', Config.SAMPLES['wikiart'], 5000, 'wikiart'),
            ('prithivMLmods/Realistic-Face-Portrait-1024px', 'train', Config.SAMPLES['portraits'], 0, 'portraits'),
        ]
        
        for name, split, num, skip, key in real_datasets:
            images, _ = self.download_dataset(name, split, num, skip, key)
            all_datasets[key] = (images, key)
        
        total_time = time.time() - start_time
        
        print("\n" + "="*100)
        print("✅ DOWNLOAD COMPLETE!")
        print("="*100)
        print(f"\n⏱️  Total time: {total_time/60:.1f} minutes")
        
        print("\n📊 Downloaded:")
        total_images = 0
        for key, (imgs, _) in all_datasets.items():
            count = len(imgs)
            total_images += count
            print(f"   {key:15s}: {count:,} images")
        
        print(f"\n   Total: {total_images:,} images")
        
        return all_datasets

# ============================================================================
# BUILD NANO DATASET - V2 WITH DIVERSE REAL IMAGES
# ============================================================================

def build_nano_dataset_v2(all_datasets: Dict, config: Config) -> Tuple:
    """Build Nano dataset with 100 diverse real images"""
    
    print("\n" + "="*100)
    print("🔨 BUILDING NANO DETECTOR DATASET V2 (640 IMAGES)")
    print("="*100)
    print("Strategy: 40% positive / 60% negative with DIVERSE real images")
    
    nano_imgs, _ = all_datasets['nano']
    flux_imgs, _ = all_datasets['flux']
    sdxl_imgs, _ = all_datasets['sdxl']
    seedream_imgs, _ = all_datasets['seedream']
    gbt_imgs, _ = all_datasets['imagegbt']
    
    # NEW: Diverse real images
    cars_imgs, _ = all_datasets['cars']
    food_imgs, _ = all_datasets['food']
    pokemon_imgs, _ = all_datasets['pokemon']
    wikiart_imgs, _ = all_datasets['wikiart']
    portraits_imgs, _ = all_datasets['portraits']
    
    augmentation = AdvancedAugmentation(config)
    
    images = []
    labels = []
    sources = []
    
    # Positive: 240 (200 real + 40 augmented)
    num_nano = min(200, len(nano_imgs))
    print(f"✅ Nano (positive): {num_nano} real + 40 augmented = {num_nano + 40} images (37.5%)")
    images.extend(nano_imgs[0:num_nano])
    labels.extend([1] * num_nano)
    sources.extend(['nano'] * num_nano)
    
    augmented = create_augmented_copies(nano_imgs[0:num_nano], 40, augmentation)
    images.extend(augmented)
    labels.extend([1] * len(augmented))
    sources.extend(['nano_aug'] * len(augmented))
    
    # Negative: 400 total (62.5%)
    # FLUX: 100
    num_flux = min(100, len(flux_imgs))
    print(f"✅ FLUX (negative): {num_flux} images")
    images.extend(flux_imgs[0:num_flux])
    labels.extend([0] * num_flux)
    sources.extend(['flux'] * num_flux)
    
    # SDXL: 100
    num_sdxl = min(100, len(sdxl_imgs))
    print(f"✅ SDXL (negative): {num_sdxl} images")
    images.extend(sdxl_imgs[0:num_sdxl])
    labels.extend([0] * num_sdxl)
    sources.extend(['sdxl'] * num_sdxl)
    
    # Other AI: 100
    num_seedream = min(50, len(seedream_imgs))
    num_gbt = min(50, len(gbt_imgs))
    print(f"✅ Other AI (negative): {num_seedream + num_gbt} images (SeeDream + ImageGBT)")
    images.extend(seedream_imgs[0:num_seedream])
    images.extend(gbt_imgs[0:num_gbt])
    labels.extend([0] * (num_seedream + num_gbt))
    sources.extend(['other_ai'] * (num_seedream + num_gbt))
    
    # REAL: 100 images (DOUBLED!) with DIVERSE sources
    real_sources = []
    
    # Cars (photographic)
    num_cars = min(20, len(cars_imgs))
    print(f"✅ REAL - Cars (photographic): {num_cars} images ← NEW!")
    images.extend(cars_imgs[0:num_cars])
    labels.extend([0] * num_cars)
    sources.extend(['real'] * num_cars)
    real_sources.append(('cars', num_cars))
    
    # Food (photographic)
    num_food = min(20, len(food_imgs))
    print(f"✅ REAL - Food (photographic): {num_food} images ← NEW!")
    images.extend(food_imgs[0:num_food])
    labels.extend([0] * num_food)
    sources.extend(['real'] * num_food)
    real_sources.append(('food', num_food))
    
    # Pokemon (sprites)
    num_pokemon = min(20, len(pokemon_imgs))
    print(f"✅ REAL - Pokemon (sprites): {num_pokemon} images")
    images.extend(pokemon_imgs[0:num_pokemon])
    labels.extend([0] * num_pokemon)
    sources.extend(['real'] * num_pokemon)
    real_sources.append(('pokemon', num_pokemon))
    
    # WikiArt (artistic - keep some for balance)
    num_wikiart = min(20, len(wikiart_imgs))
    print(f"✅ REAL - WikiArt (artistic): {num_wikiart} images")
    images.extend(wikiart_imgs[0:num_wikiart])
    labels.extend([0] * num_wikiart)
    sources.extend(['real'] * num_wikiart)
    real_sources.append(('wikiart', num_wikiart))
    
    # Portraits (photographic)
    num_portraits = min(10, len(portraits_imgs))
    print(f"✅ REAL - Portraits (photographic): {num_portraits} images")
    images.extend(portraits_imgs[0:num_portraits])
    labels.extend([0] * num_portraits)
    sources.extend(['real'] * num_portraits)
    real_sources.append(('portraits', num_portraits))
    
    total_real = sum(count for _, count in real_sources)
    print(f"\n📊 Total REAL images: {total_real} (vs 50 before) ← 2x MORE!")
    
    total = len(images)
    pos = sum(labels)
    neg = total - pos
    
    print(f"\n📊 Total: {total} ({pos} pos / {neg} neg)")
    print(f"Balance: {pos/total*100:.1f}% vs {neg/total*100:.1f}%")
    
    return images, labels, sources

# ============================================================================
# QUICK VALIDATOR
# ============================================================================

def quick_validate(model, processor, device):
    """Quick validation on cars and pokemon to check FPR"""
    
    print("\n" + "="*100)
    print("🧪 QUICK VALIDATION - CARS & POKEMON")
    print("="*100)
    
    # Download test datasets
    downloader = EnhancedDatasetDownloader()
    
    print("\n📥 Loading Cars dataset (failed before)...")
    cars_imgs, _ = downloader.download_dataset('tanganke/stanford_cars', 'test', 50, 0, 'cars')
    
    print("\n📥 Loading Pokemon dataset (passed before)...")
    pokemon_imgs, _ = downloader.download_dataset('huggan/pokemon', 'train', 50, 100, 'pokemon')
    
    if len(cars_imgs) == 0 or len(pokemon_imgs) == 0:
        print("⚠️  Could not load validation datasets")
        return
    
    # Test on cars
    model.eval()
    predictions = []
    
    print("\n🚗 Testing on Cars...")
    for img in cars_imgs:
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(inputs['pixel_values'])
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            predictions.append(pred)
    
    cars_correct = sum(1 for p in predictions if p == 0)  # Should be 0 (not AI)
    cars_fpr = (len(predictions) - cars_correct) / len(predictions) * 100
    
    print(f"   Cars: {cars_correct}/{len(predictions)} correct")
    print(f"   FPR: {cars_fpr:.1f}% (was 94% before!)")
    
    # Test on pokemon
    predictions = []
    
    print("\n🎮 Testing on Pokemon...")
    for img in pokemon_imgs:
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(inputs['pixel_values'])
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            predictions.append(pred)
    
    pokemon_correct = sum(1 for p in predictions if p == 0)
    pokemon_fpr = (len(predictions) - pokemon_correct) / len(predictions) * 100
    
    print(f"   Pokemon: {pokemon_correct}/{len(predictions)} correct")
    print(f"   FPR: {pokemon_fpr:.1f}%")
    
    # Overall
    total_correct = cars_correct + pokemon_correct
    total_images = len(cars_imgs) + len(pokemon_imgs)
    overall_fpr = (total_images - total_correct) / total_images * 100
    
    print(f"\n📊 Overall Real Image Performance:")
    print(f"   Correct: {total_correct}/{total_images} ({total_correct/total_images*100:.1f}%)")
    print(f"   FPR: {overall_fpr:.1f}% (vs 47% before)")
    
    if overall_fpr < 10:
        print("   ✅ EXCELLENT! FPR < 10%")
    elif overall_fpr < 20:
        print("   ✅ GOOD! FPR < 20%")
    else:
        print("   ⚠️  Still high FPR, may need more training")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Retrain Nano detector"""
    
    print("\n" + "="*100)
    print("🎯 INITIALIZING NANO RETRAINING")
    print("="*100)
    
    if torch.cuda.is_available():
        print(f"\n💻 Hardware:")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Download datasets
    downloader = EnhancedDatasetDownloader()
    all_datasets = downloader.download_all_for_nano()
    
    # Build dataset
    images, labels, sources = build_nano_dataset_v2(all_datasets, Config)
    
    # Load processor
    print(f"\n💿 Loading image processor...")
    processor = ViTImageProcessor.from_pretrained(Config.BASE_MODEL)
    print(f"✅ Processor loaded!")
    
    # Split data
    print("\n📊 Nano Detector Splits:")
    
    nano_train, nano_val, nano_test = stratified_split(
        images, labels, sources,
        Config.TRAIN_RATIO, Config.VAL_RATIO, Config.TEST_RATIO, Config.RANDOM_SEED
    )
    
    # Create datasets
    nano_train_dataset = DetectorDataset(nano_train[0], nano_train[1], processor, Config, augment=True)
    nano_val_dataset = DetectorDataset(nano_val[0], nano_val[1], processor, Config, augment=False)
    nano_test_dataset = DetectorDataset(nano_test[0], nano_test[1], processor, Config, augment=False)
    
    # Create dataloaders
    nano_train_loader = DataLoader(nano_train_dataset, batch_size=Config.BATCH_SIZE, 
                                    shuffle=True, num_workers=Config.NUM_WORKERS, 
                                    pin_memory=Config.PIN_MEMORY)
    nano_val_loader = DataLoader(nano_val_dataset, batch_size=Config.BATCH_SIZE, 
                                  shuffle=False, num_workers=Config.NUM_WORKERS, 
                                  pin_memory=Config.PIN_MEMORY)
    nano_test_loader = DataLoader(nano_test_dataset, batch_size=Config.BATCH_SIZE, 
                                   shuffle=False)
    
    # Train
    print("\n" + "="*100)
    print("🚀 TRAINING NANO DETECTOR V2")
    print("="*100)
    
    trainer = AdvancedTrainer("nano_detector_v2", nano_train_loader, nano_val_loader,
                             nano_test_loader, Config.MODEL_DIR, Config)
    results = trainer.train()
    
    print("\n" + "="*100)
    print("🧪 QUICK VALIDATION ON PROBLEM DATASETS")
    print("="*100)
    
    # Load best model
    model = DetectorModel(Config)
    model.load_state_dict(torch.load(Config.MODEL_DIR / "best_model.pt"))
    model.to(Config.DEVICE)
    
    # Quick validation
    quick_validate(model, processor, Config.DEVICE)
    
    print("\n" + "="*100)
    print("✅ NANO RETRAINING COMPLETE!")
    print("="*100)
    
    print(f"\n📊 Test Results:")
    print(f"   Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"   Precision: {results['precision']*100:.2f}%")
    print(f"   Recall:    {results['recall']*100:.2f}%")
    print(f"   F1 Score:  {results['f1']*100:.2f}%")
    print(f"   FPR:       {results['fpr']*100:.2f}%")
    
    print(f"\n💾 Model saved: {Config.MODEL_DIR / 'best_model.pt'}")
    
    print("\n📝 Next steps:")
    print("   1. Run full validation: python validate_small_models.py")
    print("   2. If FPR < 10%: Upload to HuggingFace")
    print("   3. If FPR still high: Add more diverse real images")

if __name__ == "__main__":
    main()
