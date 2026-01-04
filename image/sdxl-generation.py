#!/usr/bin/env python3
"""
SDXL AI Image Detector - Complete Production Pipeline
======================================================

A comprehensive pipeline for:
1. Generating 10,000 SDXL images with diverse prompts
2. Creating stratified train/val/test splits
3. Training a ResNet-50 binary classifier
4. Testing with comprehensive metrics
5. Uploading to HuggingFace

Author: Claude & User
Version: 1.0
Speed Optimized: 10 inference steps (~6-7 sec/image on L4 GPU)
Total Time: ~17-19 hours on L4 GPU
"""

# ============================================
# IMPORTS - CAREFUL ORDER TO AVOID CIRCULAR IMPORTS
# ============================================

# Import torch FIRST and fully initialize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.set_grad_enabled(True)  # Force full initialization

# Core imports
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Metrics and visualization
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix, 
    classification_report,
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
from collections import defaultdict, Counter
import json
import shutil
import warnings
import sys
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
import gc

# HuggingFace
from huggingface_hub import HfApi, create_repo
from datasets import Dataset as HFDataset, Image as HFImage

# Diffusers - IMPORT LAST after torch is fully loaded
from diffusers import StableDiffusionXLPipeline

# Configure
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION CLASS
# ============================================

class Config:
    """
    Centralized configuration for the entire pipeline.
    All hyperparameters, paths, and settings in one place.
    """
    
    # ========== Paths ==========
    REAL_IMAGES_DIR = Path("./real_images_dataset")
    OUTPUT_DIR = Path("./detector_outputs/sdxl")
    GENERATED_DIR = OUTPUT_DIR / "generated"
    DATASET_DIR = OUTPUT_DIR / "dataset"
    MODEL_DIR = OUTPUT_DIR / "model"
    RESULTS_DIR = OUTPUT_DIR / "results"
    LOGS_DIR = OUTPUT_DIR / "logs"
    PLOTS_DIR = OUTPUT_DIR / "plots"
    
    # ========== HuggingFace Settings ==========
    HF_USERNAME = "ash12321"
    HF_TOKEN = "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"
    HF_REPO_NAME = "sdxl-detector-resnet50"
    HF_DATASET_NAME = "sdxl-generated-10k"
    
    # ========== Generation Settings ==========
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    NUM_IMAGES = 10000
    NUM_INFERENCE_STEPS = 10  # Optimized for speed (6-7 sec/image)
    GUIDANCE_SCALE = 7.0
    IMAGE_SIZE = 1024
    JPEG_QUALITY = 95
    
    # ========== Training Settings ==========
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.0001
    TRAIN_SIZE = 256
    GRADIENT_CLIP = 1.0
    
    # ========== Data Split Settings ==========
    RANDOM_SEED = 42
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # ========== Early Stopping ==========
    PATIENCE = 5
    MIN_DELTA = 0.001
    
    # ========== Device Settings ==========
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # ========== Augmentation Settings ==========
    FLIP_PROB = 0.5
    ROTATION_DEGREES = 15
    COLOR_JITTER_BRIGHTNESS = 0.2
    COLOR_JITTER_CONTRAST = 0.2
    COLOR_JITTER_SATURATION = 0.2
    COLOR_JITTER_HUE = 0.1
    TRANSLATE = 0.1
    
    # ========== Logging Settings ==========
    LOG_INTERVAL = 100  # Log every N images
    CHECKPOINT_INTERVAL = 1000  # Save checkpoint every N images
    
    @classmethod
    def print_config(cls):
        """Print all configuration settings"""
        logger.info("\n" + "="*70)
        logger.info("CONFIGURATION")
        logger.info("="*70)
        logger.info(f"Device: {cls.DEVICE}")
        logger.info(f"Model: {cls.MODEL_ID}")
        logger.info(f"Images to generate: {cls.NUM_IMAGES:,}")
        logger.info(f"Inference steps: {cls.NUM_INFERENCE_STEPS}")
        logger.info(f"Image size: {cls.IMAGE_SIZE}×{cls.IMAGE_SIZE}")
        logger.info(f"Batch size: {cls.BATCH_SIZE}")
        logger.info(f"Epochs: {cls.NUM_EPOCHS}")
        logger.info(f"Learning rate: {cls.LEARNING_RATE}")
        logger.info(f"Random seed: {cls.RANDOM_SEED}")
        logger.info("="*70 + "\n")

# ============================================
# PROMPT GENERATION SYSTEM
# ============================================

class PromptGenerator:
    """
    Advanced prompt generation system with diverse templates
    and randomized components for maximum variety.
    """
    
    # Prompt templates - organized by category
    PEOPLE_TEMPLATES = [
        "a portrait of {subject} {action}, {style}",
        "photo of {subject} {location}, {quality}",
        "{subject} {action}, professional photography",
        "headshot of {subject}, {style}",
        "candid photo of {subject} {action}, {quality}",
        "{subject} with {expression}, {style}",
        "full body shot of {subject}, {quality}",
        "{subject} in {setting}, {style}",
    ]
    
    LANDSCAPE_TEMPLATES = [
        "a beautiful {landscape} during {time}, {style}",
        "{landscape} with {weather}, {quality}",
        "scenic view of {landscape}, {style}",
        "panoramic photo of {landscape}, {quality}",
        "{landscape} at {time}, {style}",
        "nature photography of {landscape}, {quality}",
        "aerial view of {landscape}, {style}",
        "{landscape} landscape, {quality}",
    ]
    
    OBJECT_TEMPLATES = [
        "a {object} on {surface}, {style}",
        "close-up of {object}, {quality}",
        "{object} in {setting}, professional photo",
        "product photo of {object}, {style}",
        "macro shot of {object}, {quality}",
        "{object} with {lighting}, {style}",
        "studio photo of {object}, {quality}",
    ]
    
    ANIMAL_TEMPLATES = [
        "a {animal} {action}, {style}",
        "{animal} in {habitat}, {quality}",
        "wildlife photo of {animal}, {style}",
        "close-up of {animal}, {quality}",
        "{animal} portrait, {style}",
        "{animal} in natural habitat, {quality}",
    ]
    
    FOOD_TEMPLATES = [
        "a plate of {food}, {style}",
        "{food} on {surface}, food photography",
        "delicious {food}, {quality}",
        "gourmet {food}, {style}",
        "{food} with {lighting}, food photography",
        "professional photo of {food}, {quality}",
    ]
    
    ARCHITECTURE_TEMPLATES = [
        "{building} at {time}, {style}",
        "architectural photo of {building}, {quality}",
        "{building} with {feature}, {style}",
        "exterior of {building}, {quality}",
        "{building} facade, {style}",
        "modern {building}, {quality}",
    ]
    
    # Component lists
    SUBJECTS = [
        "a person", "a woman", "a man", "a child",
        "an elderly person", "a professional", "an athlete",
        "a student", "a couple", "friends", "a family"
    ]
    
    ACTIONS = [
        "smiling", "walking", "sitting", "standing",
        "reading", "working", "relaxing", "laughing",
        "thinking", "talking", "looking away", "posing"
    ]
    
    EXPRESSIONS = [
        "a happy expression", "a serious look", "a friendly smile",
        "a confident pose", "a thoughtful gaze", "a joyful expression"
    ]
    
    LANDSCAPES = [
        "mountains", "ocean", "forest", "desert", "lake",
        "valley", "countryside", "beach", "canyon",
        "waterfall", "hills", "river", "meadow", "cliffs"
    ]
    
    TIMES = [
        "sunset", "sunrise", "golden hour", "blue hour",
        "night", "daytime", "dusk", "dawn", "afternoon",
        "morning", "twilight", "midday"
    ]
    
    WEATHER = [
        "clear sky", "clouds", "dramatic sky", "fog",
        "misty conditions", "sunny weather", "overcast sky",
        "storm clouds", "cloudy sky", "perfect weather"
    ]
    
    OBJECTS = [
        "coffee cup", "book", "flower", "watch", "camera",
        "laptop", "phone", "keys", "glasses", "bottle",
        "plant", "vase", "pen", "notebook", "headphones"
    ]
    
    SURFACES = [
        "wooden table", "marble counter", "glass surface",
        "fabric", "concrete", "stone", "metal surface",
        "leather", "white background", "dark background"
    ]
    
    ANIMALS = [
        "cat", "dog", "bird", "deer", "fox", "horse",
        "rabbit", "squirrel", "lion", "elephant", "tiger",
        "bear", "wolf", "eagle", "owl", "dolphin"
    ]
    
    HABITATS = [
        "forest", "garden", "field", "park", "wilderness",
        "savanna", "jungle", "meadow", "mountains",
        "plains", "wetlands", "desert"
    ]
    
    FOODS = [
        "pasta", "salad", "burger", "sushi", "pizza",
        "cake", "fruit", "vegetables", "steak", "soup",
        "dessert", "sandwich", "noodles", "rice", "bread"
    ]
    
    BUILDINGS = [
        "modern house", "skyscraper", "church", "bridge",
        "tower", "castle", "museum", "library", "station",
        "temple", "palace", "office building", "apartment"
    ]
    
    FEATURES = [
        "glass windows", "stone walls", "wooden doors",
        "metal structure", "gardens", "columns", "arches",
        "balconies", "fountains", "stairs", "entrance"
    ]
    
    STYLES = [
        "4k photo", "professional photography", "high resolution",
        "detailed", "cinematic", "realistic", "sharp focus",
        "masterpiece", "8k quality", "award winning", "stunning"
    ]
    
    QUALITIES = [
        "professional photo", "high quality", "4k", "detailed",
        "sharp focus", "beautiful lighting", "vibrant colors",
        "stunning", "perfect composition", "professional quality"
    ]
    
    SETTINGS = [
        "studio", "outdoor", "indoor", "natural light",
        "dramatic lighting", "soft lighting", "bright",
        "moody", "atmospheric", "well-lit", "ambient"
    ]
    
    LOCATIONS = [
        "in a park", "at the beach", "in the city", "in nature",
        "at home", "in studio", "on the street", "in the mountains",
        "by the water", "in the countryside", "downtown"
    ]
    
    LIGHTING = [
        "soft lighting", "dramatic lighting", "natural light",
        "golden hour light", "studio lighting", "ambient light",
        "backlight", "side lighting", "rim lighting", "diffused light"
    ]
    
    @classmethod
    def get_all_templates(cls) -> List[str]:
        """Get all templates combined"""
        return (
            cls.PEOPLE_TEMPLATES +
            cls.LANDSCAPE_TEMPLATES +
            cls.OBJECT_TEMPLATES +
            cls.ANIMAL_TEMPLATES +
            cls.FOOD_TEMPLATES +
            cls.ARCHITECTURE_TEMPLATES
        )
    
    @classmethod
    def generate(cls) -> str:
        """Generate a random diverse prompt"""
        template = random.choice(cls.get_all_templates())
        
        prompt = template.format(
            subject=random.choice(cls.SUBJECTS),
            action=random.choice(cls.ACTIONS),
            expression=random.choice(cls.EXPRESSIONS),
            landscape=random.choice(cls.LANDSCAPES),
            time=random.choice(cls.TIMES),
            weather=random.choice(cls.WEATHER),
            object=random.choice(cls.OBJECTS),
            surface=random.choice(cls.SURFACES),
            animal=random.choice(cls.ANIMALS),
            habitat=random.choice(cls.HABITATS),
            food=random.choice(cls.FOODS),
            building=random.choice(cls.BUILDINGS),
            feature=random.choice(cls.FEATURES),
            style=random.choice(cls.STYLES),
            quality=random.choice(cls.QUALITIES),
            setting=random.choice(cls.SETTINGS),
            location=random.choice(cls.LOCATIONS),
            lighting=random.choice(cls.LIGHTING)
        )
        
        return prompt
    
    @classmethod
    def generate_batch(cls, n: int) -> List[str]:
        """Generate multiple prompts"""
        return [cls.generate() for _ in range(n)]

# ============================================
# IMAGE GENERATION
# ============================================

class ImageGenerator:
    """
    SDXL image generator with progress tracking,
    error handling, and resume capability.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.pipe = None
        self.stats = {
            'generated': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None,
            'prompts_used': [],
            'generation_times': []
        }
    
    def load_model(self):
        """Load SDXL pipeline with optimizations"""
        logger.info("Loading SDXL model...")
        
        try:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.config.MODEL_ID,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            
            self.pipe.to(self.config.DEVICE)
            
            # Enable memory optimizations
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_single_image(self, index: int) -> bool:
        """Generate a single image with error handling"""
        output_path = self.config.GENERATED_DIR / f"sdxl_{index:06d}.jpg"
        
        # Skip if already exists
        if output_path.exists():
            self.stats['skipped'] += 1
            return True
        
        try:
            start_time = time.time()
            
            # Generate prompt
            prompt = PromptGenerator.generate()
            self.stats['prompts_used'].append(prompt)
            
            # Random seed for diversity
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(self.config.DEVICE).manual_seed(seed)
            
            # Generate image
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=self.config.NUM_INFERENCE_STEPS,
                guidance_scale=self.config.GUIDANCE_SCALE,
                height=self.config.IMAGE_SIZE,
                width=self.config.IMAGE_SIZE,
                generator=generator
            ).images[0]
            
            # Save with high quality
            image.save(output_path, "JPEG", quality=self.config.JPEG_QUALITY)
            
            # Track time
            elapsed = time.time() - start_time
            self.stats['generation_times'].append(elapsed)
            
            self.stats['generated'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate image {index}: {e}")
            self.stats['failed'] += 1
            return False
    
    def generate_batch(self, start_idx: int = 0):
        """Generate all images with progress tracking"""
        logger.info(f"Starting generation from image {start_idx}")
        
        # Create output directory
        self.config.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        
        self.stats['start_time'] = datetime.now()
        
        # Load model
        self.load_model()
        
        # Generate images
        for i in tqdm(range(start_idx, self.config.NUM_IMAGES), desc="Generating SDXL images"):
            success = self.generate_single_image(i)
            
            # Periodic cleanup
            if (i + 1) % self.config.LOG_INTERVAL == 0:
                self.log_progress(i + 1)
                gc.collect()
                torch.cuda.empty_cache()
            
            # Stop if too many failures
            if self.stats['failed'] > 50:
                logger.error("Too many failures, stopping generation")
                break
        
        self.stats['end_time'] = datetime.now()
        
        # Cleanup
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()
        
        # Final statistics
        self.log_final_stats()
        self.save_generation_report()
    
    def log_progress(self, current: int):
        """Log generation progress"""
        elapsed = datetime.now() - self.stats['start_time']
        avg_time = np.mean(self.stats['generation_times'][-100:]) if self.stats['generation_times'] else 0
        remaining = (self.config.NUM_IMAGES - current) * avg_time
        
        logger.info(f"\nProgress: {current}/{self.config.NUM_IMAGES} ({current/self.config.NUM_IMAGES*100:.1f}%)")
        logger.info(f"Elapsed: {elapsed}")
        logger.info(f"Avg time/image: {avg_time:.2f}s")
        logger.info(f"Est. remaining: {timedelta(seconds=int(remaining))}")
        logger.info(f"Generated: {self.stats['generated']:,}, Failed: {self.stats['failed']}, Skipped: {self.stats['skipped']}")
    
    def log_final_stats(self):
        """Log final generation statistics"""
        elapsed = self.stats['end_time'] - self.stats['start_time']
        avg_time = np.mean(self.stats['generation_times']) if self.stats['generation_times'] else 0
        
        logger.info("\n" + "="*70)
        logger.info("GENERATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Total time: {elapsed}")
        logger.info(f"Generated: {self.stats['generated']:,} images")
        logger.info(f"Failed: {self.stats['failed']} images")
        logger.info(f"Skipped: {self.stats['skipped']} images")
        logger.info(f"Avg time/image: {avg_time:.2f}s")
        logger.info(f"Output: {self.config.GENERATED_DIR}")
        logger.info("="*70 + "\n")
    
    def save_generation_report(self):
        """Save detailed generation report"""
        report = {
            'generated': self.stats['generated'],
            'failed': self.stats['failed'],
            'skipped': self.stats['skipped'],
            'start_time': self.stats['start_time'].isoformat(),
            'end_time': self.stats['end_time'].isoformat(),
            'total_duration': str(self.stats['end_time'] - self.stats['start_time']),
            'avg_time_per_image': float(np.mean(self.stats['generation_times'])) if self.stats['generation_times'] else 0,
            'config': {
                'model': self.config.MODEL_ID,
                'num_images': self.config.NUM_IMAGES,
                'inference_steps': self.config.NUM_INFERENCE_STEPS,
                'guidance_scale': self.config.GUIDANCE_SCALE,
                'image_size': self.config.IMAGE_SIZE
            }
        }
        
        self.config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.config.LOGS_DIR / 'generation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generation report saved to {self.config.LOGS_DIR / 'generation_report.json'}")

# ============================================
# HUGGINGFACE DATASET UPLOAD
# ============================================

class DatasetUploader:
    """Upload generated images to HuggingFace"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_dataset_card(self, num_images: int) -> str:
        """Create dataset card markdown"""
        return f"""---
license: mit
task_categories:
- image-classification
tags:
- ai-generated
- sdxl
- stable-diffusion-xl
size_categories:
- 10K<n<100K
---

# SDXL Generated Images Dataset ({num_images:,} images)

This dataset contains {num_images:,} AI-generated images created with Stable Diffusion XL for training an AI image detector.

## Dataset Details

- **Model**: Stable Diffusion XL Base 1.0
- **Total Images**: {num_images:,}
- **Resolution**: {self.config.IMAGE_SIZE}×{self.config.IMAGE_SIZE} pixels
- **Format**: JPEG (quality {self.config.JPEG_QUALITY})
- **Inference Steps**: {self.config.NUM_INFERENCE_STEPS}
- **Guidance Scale**: {self.config.GUIDANCE_SCALE}
- **Random Seeds**: Unique per image for maximum diversity
- **Generation Date**: {datetime.now().strftime('%Y-%m-%d')}

## Prompt Diversity

Images generated with diverse prompts covering:
- Portraits and people (various poses, expressions, settings)
- Landscapes (mountains, oceans, forests, etc.)
- Objects and still life
- Animals in natural habitats
- Food photography
- Architecture and buildings
- Various lighting conditions and styles

## Usage

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("{self.config.HF_USERNAME}/{self.config.HF_DATASET_NAME}")

# Access images
for example in dataset["train"]:
    image = example["image"]
    filename = example["filename"]
```

## License

MIT License - Free for research and commercial use.

## Related

- Detector Model: [{self.config.HF_USERNAME}/{self.config.HF_REPO_NAME}](https://huggingface.co/{self.config.HF_USERNAME}/{self.config.HF_REPO_NAME})
"""
    
    def upload(self):
        """Upload dataset to HuggingFace"""
        logger.info("="*70)
        logger.info("UPLOADING DATASET TO HUGGINGFACE")
        logger.info("="*70)
        
        try:
            api = HfApi(token=self.config.HF_TOKEN)
            repo_id = f"{self.config.HF_USERNAME}/{self.config.HF_DATASET_NAME}"
            
            # Create repo
            logger.info(f"Creating repository: {repo_id}")
            create_repo(
                repo_id=repo_id,
                token=self.config.HF_TOKEN,
                repo_type="dataset",
                private=False,
                exist_ok=True
            )
            
            # Collect images
            images = sorted(list(self.config.GENERATED_DIR.glob("*.jpg")))
            logger.info(f"Found {len(images):,} images to upload")
            
            if len(images) == 0:
                logger.warning("No images to upload")
                return
            
            # Create dataset
            logger.info("Creating HuggingFace dataset...")
            dataset_dict = {
                "image": [str(p) for p in images],
                "filename": [p.name for p in images],
                "index": list(range(len(images)))
            }
            
            dataset = HFDataset.from_dict(dataset_dict)
            dataset = dataset.cast_column("image", HFImage())
            
            # Upload
            logger.info("Uploading to HuggingFace (this may take 10-15 minutes)...")
            dataset.push_to_hub(
                repo_id,
                token=self.config.HF_TOKEN,
                commit_message=f"Upload {len(images):,} SDXL generated images"
            )
            
            # Upload dataset card
            logger.info("Uploading dataset card...")
            dataset_card = self.create_dataset_card(len(images))
            api.upload_file(
                path_or_fileobj=dataset_card.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=self.config.HF_TOKEN
            )
            
            logger.info("="*70)
            logger.info("✅ DATASET UPLOADED SUCCESSFULLY")
            logger.info("="*70)
            logger.info(f"🤗 https://huggingface.co/datasets/{repo_id}")
            logger.info("="*70 + "\n")
            
        except Exception as e:
            logger.error(f"Dataset upload failed: {e}")
            logger.info(f"Images are still saved locally at: {self.config.GENERATED_DIR}")

# ============================================
# DATASET SPLITS
# ============================================

class DatasetSplitter:
    """Create stratified train/val/test splits"""
    
    def __init__(self, config: Config):
        self.config = config
        self.stats = {}
    
    def create_splits(self):
        """Create stratified splits ensuring each dataset is represented"""
        logger.info("="*70)
        logger.info("CREATING STRATIFIED DATASET SPLITS")
        logger.info("="*70)
        logger.info(f"Random seed: {self.config.RANDOM_SEED}")
        logger.info(f"Split ratios: Train={self.config.TRAIN_RATIO}, Val={self.config.VAL_RATIO}, Test={self.config.TEST_RATIO}")
        
        # Set seeds
        random.seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
        
        self.config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        
        # Collect real images by dataset
        logger.info("\nCollecting real images by dataset...")
        real_datasets = defaultdict(list)
        
        for dataset_dir in self.config.REAL_IMAGES_DIR.iterdir():
            if dataset_dir.is_dir():
                images = list(dataset_dir.glob("*.jpg"))
                if images:
                    real_datasets[dataset_dir.name] = images
                    logger.info(f"  {dataset_dir.name}: {len(images):,} images")
        
        # Collect fake images
        logger.info("\nCollecting SDXL images...")
        fake_images = list(self.config.GENERATED_DIR.glob("*.jpg"))
        logger.info(f"  sdxl: {len(fake_images):,} images")
        
        # Calculate targets
        total_real = sum(len(imgs) for imgs in real_datasets.values())
        target_real = min(self.config.NUM_IMAGES, total_real)
        
        logger.info(f"\nDataset statistics:")
        logger.info(f"  Total real available: {total_real:,}")
        logger.info(f"  Target real images: {target_real:,}")
        logger.info(f"  Total fake images: {len(fake_images):,}")
        
        # Create splits
        train_data, val_data, test_data = [], [], []
        
        logger.info("\nCreating stratified splits for real images...")
        
        # Split real images stratified by dataset
        for dataset_name, images in real_datasets.items():
            random.shuffle(images)
            
            # Proportional sampling
            proportion = len(images) / total_real
            num_from_dataset = min(int(target_real * proportion), len(images))
            selected = images[:num_from_dataset]
            
            # Split this dataset
            n = len(selected)
            train_size = int(n * self.config.TRAIN_RATIO)
            val_size = int(n * self.config.VAL_RATIO)
            
            train_imgs = selected[:train_size]
            val_imgs = selected[train_size:train_size + val_size]
            test_imgs = selected[train_size + val_size:]
            
            # Add to splits
            for img in train_imgs:
                train_data.append({
                    'image_path': str(img),
                    'label': 0,
                    'label_name': 'real',
                    'source': dataset_name
                })
            
            for img in val_imgs:
                val_data.append({
                    'image_path': str(img),
                    'label': 0,
                    'label_name': 'real',
                    'source': dataset_name
                })
            
            for img in test_imgs:
                test_data.append({
                    'image_path': str(img),
                    'label': 0,
                    'label_name': 'real',
                    'source': dataset_name
                })
            
            logger.info(f"  {dataset_name}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
        
        # Split fake images
        logger.info("\nShuffling and splitting fake images...")
        random.shuffle(fake_images)
        
        n_fake = len(fake_images)
        fake_train_size = int(n_fake * self.config.TRAIN_RATIO)
        fake_val_size = int(n_fake * self.config.VAL_RATIO)
        
        fake_train = fake_images[:fake_train_size]
        fake_val = fake_images[fake_train_size:fake_train_size + fake_val_size]
        fake_test = fake_images[fake_train_size + fake_val_size:]
        
        for img in fake_train:
            train_data.append({'image_path': str(img), 'label': 1, 'label_name': 'fake', 'source': 'sdxl'})
        for img in fake_val:
            val_data.append({'image_path': str(img), 'label': 1, 'label_name': 'fake', 'source': 'sdxl'})
        for img in fake_test:
            test_data.append({'image_path': str(img), 'label': 1, 'label_name': 'fake', 'source': 'sdxl'})
        
        logger.info(f"  sdxl: train={len(fake_train)}, val={len(fake_val)}, test={len(fake_test)}")
        
        # Final shuffle
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        # Save to CSV
        pd.DataFrame(train_data).to_csv(self.config.DATASET_DIR / "train.csv", index=False)
        pd.DataFrame(val_data).to_csv(self.config.DATASET_DIR / "val.csv", index=False)
        pd.DataFrame(test_data).to_csv(self.config.DATASET_DIR / "test.csv", index=False)
        
        # Log final statistics
        total = len(train_data) + len(val_data) + len(test_data)
        
        logger.info("\n" + "="*70)
        logger.info("SPLITS CREATED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"Train: {len(train_data):,} images ({len(train_data)/total*100:.1f}%)")
        logger.info(f"  - Real: {sum(1 for x in train_data if x['label']==0):,}")
        logger.info(f"  - Fake: {sum(1 for x in train_data if x['label']==1):,}")
        logger.info(f"Val:   {len(val_data):,} images ({len(val_data)/total*100:.1f}%)")
        logger.info(f"  - Real: {sum(1 for x in val_data if x['label']==0):,}")
        logger.info(f"  - Fake: {sum(1 for x in val_data if x['label']==1):,}")
        logger.info(f"Test:  {len(test_data):,} images ({len(test_data)/total*100:.1f}%)")
        logger.info(f"  - Real: {sum(1 for x in test_data if x['label']==0):,}")
        logger.info(f"  - Fake: {sum(1 for x in test_data if x['label']==1):,}")
        logger.info(f"\nSaved to: {self.config.DATASET_DIR}")
        logger.info("="*70 + "\n")
        
        # Save split statistics
        self.stats = {
            'train': {'total': len(train_data), 'real': sum(1 for x in train_data if x['label']==0), 'fake': sum(1 for x in train_data if x['label']==1)},
            'val': {'total': len(val_data), 'real': sum(1 for x in val_data if x['label']==0), 'fake': sum(1 for x in val_data if x['label']==1)},
            'test': {'total': len(test_data), 'real': sum(1 for x in test_data if x['label']==0), 'fake': sum(1 for x in test_data if x['label']==1)}
        }
        
        self.config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.config.LOGS_DIR / 'split_statistics.json', 'w') as f:
            json.dump(self.stats, f, indent=2)

# ============================================
# DATASET CLASS
# ============================================

class ImageDataset(Dataset):
    """
    PyTorch Dataset for loading real and fake images
    with proper error handling and transformations.
    """
    
    def __init__(self, csv_path: Path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.failed_loads = 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        try:
            image = Image.open(row['image_path']).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load {row['image_path']}: {e}")
            # Return black image as fallback
            image = Image.new('RGB', (256, 256), color='black')
            self.failed_loads += 1
        
        label = row['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_failed_count(self):
        """Return number of failed image loads"""
        return self.failed_loads

# ============================================
# TRAINER CLASS
# ============================================

class Trainer:
    """
    Complete training system with:
    - Data augmentation
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Metrics tracking
    - Visualization
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.history = []
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.epochs_no_improve = 0
    
    def setup_model(self):
        """Initialize model"""
        logger.info("Setting up ResNet-50 model...")
        
        # Set seeds
        torch.manual_seed(self.config.RANDOM_SEED)
        torch.cuda.manual_seed(self.config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        
        # Load pretrained ResNet-50
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        
        # Modify final layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)
        )
        
        self.model = self.model.to(self.config.DEVICE)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # LR scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Model loaded:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    def setup_data(self):
        """Setup data loaders with augmentation"""
        logger.info("Setting up data loaders...")
        
        # Training augmentation
        train_transform = transforms.Compose([
            transforms.Resize((self.config.TRAIN_SIZE, self.config.TRAIN_SIZE)),
            transforms.RandomHorizontalFlip(p=self.config.FLIP_PROB),
            transforms.RandomRotation(self.config.ROTATION_DEGREES),
            transforms.ColorJitter(
                brightness=self.config.COLOR_JITTER_BRIGHTNESS,
                contrast=self.config.COLOR_JITTER_CONTRAST,
                saturation=self.config.COLOR_JITTER_SATURATION,
                hue=self.config.COLOR_JITTER_HUE
            ),
            transforms.RandomAffine(degrees=0, translate=(self.config.TRANSLATE, self.config.TRANSLATE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Validation transform (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((self.config.TRAIN_SIZE, self.config.TRAIN_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = ImageDataset(self.config.DATASET_DIR / "train.csv", train_transform)
        val_dataset = ImageDataset(self.config.DATASET_DIR / "val.csv", val_transform)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        logger.info(f"✅ Data loaders ready:")
        logger.info(f"  Train: {len(train_dataset):,} images ({len(self.train_loader)} batches)")
        logger.info(f"  Val:   {len(val_dataset):,} images ({len(self.val_loader)} batches)")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.config.DEVICE), labels.to(self.config.DEVICE)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.GRADIENT_CLIP)
            
            self.optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(self.train_loader)
        
        return avg_train_loss, train_acc
    
    def validate(self, epoch: int) -> Tuple[float, float, float, float]:
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Val]  ")
            
            for images, labels in pbar:
                images = images.to(self.config.DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels.to(self.config.DEVICE))
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.numpy())
                
                # Update progress bar
                current_acc = accuracy_score(val_labels, val_preds) * 100
                pbar.set_postfix({'acc': f'{current_acc:.2f}%'})
        
        # Calculate metrics
        val_acc = accuracy_score(val_labels, val_preds) * 100
        val_prec = precision_score(val_labels, val_preds, zero_division=0) * 100
        val_rec = recall_score(val_labels, val_preds, zero_division=0) * 100
        val_f1 = f1_score(val_labels, val_preds, zero_division=0) * 100
        avg_val_loss = val_loss / len(self.val_loader)
        
        return avg_val_loss, val_acc, val_prec, val_rec, val_f1
    
    def train(self):
        """Full training loop"""
        logger.info("="*70)
        logger.info("TRAINING DETECTOR")
        logger.info("="*70)
        
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        self.setup_model()
        self.setup_data()
        
        logger.info("\n🚀 Starting training...\n")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Check for improvement
            improved = False
            if val_acc > self.best_val_acc + self.config.MIN_DELTA:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                self.epochs_no_improve = 0
                improved = True
                
                # Save best checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'train_acc': train_acc,
                    'config': {
                        'model': 'resnet50',
                        'train_size': self.config.TRAIN_SIZE,
                        'random_seed': self.config.RANDOM_SEED
                    }
                }, self.config.MODEL_DIR / "best.pth")
            else:
                self.epochs_no_improve += 1
            
            # Save history
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_precision': val_prec,
                'val_recall': val_rec,
                'val_f1': val_f1,
                'lr': current_lr
            })
            
            # Log epoch results
            logger.info(f"\n{'='*70}")
            logger.info(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} Summary")
            logger.info(f"{'='*70}")
            logger.info(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            logger.info(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            logger.info(f"       Precision={val_prec:.2f}%, Recall={val_rec:.2f}%, F1={val_f1:.2f}%")
            logger.info(f"Best:  Acc={self.best_val_acc:.2f}%, F1={self.best_val_f1:.2f}%")
            logger.info(f"LR:    {current_lr:.6f}")
            if improved:
                logger.info("✅ Model improved and saved!")
            logger.info(f"{'='*70}\n")
            
            # Early stopping
            if self.epochs_no_improve >= self.config.PATIENCE:
                logger.info(f"⚠️  Early stopping triggered (no improvement for {self.config.PATIENCE} epochs)")
                break
        
        # Save training history
        pd.DataFrame(self.history).to_csv(self.config.MODEL_DIR / "training_history.csv", index=False)
        
        # Plot training curves
        self.plot_training_history()
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        logger.info(f"Best validation F1 score: {self.best_val_f1:.2f}%")
        logger.info(f"Model saved to: {self.config.MODEL_DIR / 'best.pth'}")
        logger.info("="*70 + "\n")
    
    def plot_training_history(self):
        """Plot and save training curves"""
        self.config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.history)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', linewidth=2)
        axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(df['epoch'], df['train_acc'], label='Train Acc', marker='o', linewidth=2)
        axes[0, 1].plot(df['epoch'], df['val_acc'], label='Val Acc', marker='o', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Validation metrics
        axes[1, 0].plot(df['epoch'], df['val_f1'], label='F1', marker='o', linewidth=2, color='green')
        axes[1, 0].plot(df['epoch'], df['val_precision'], label='Precision', marker='s', linewidth=2, color='blue')
        axes[1, 0].plot(df['epoch'], df['val_recall'], label='Recall', marker='^', linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Score (%)', fontsize=12)
        axes[1, 0].set_title('Validation Metrics', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(df['epoch'], df['lr'], marker='o', linewidth=2, color='red')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.PLOTS_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to: {self.config.PLOTS_DIR / 'training_history.png'}")

# ============================================
# TESTER CLASS
# ============================================

class ModelTester:
    """
    Comprehensive model testing with:
    - Multiple metrics
    - Confusion matrix
    - ROC curve
    - Precision-Recall curve
    - Per-class analysis
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.results = {}
    
    def load_best_model(self):
        """Load the best checkpoint"""
        logger.info("Loading best model checkpoint...")
        
        self.model = models.resnet50()
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)
        )
        
        checkpoint = torch.load(self.config.MODEL_DIR / "best.pth", map_location=self.config.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        
        logger.info(f"✅ Model loaded (best val acc: {checkpoint.get('val_acc', 0):.2f}%)")
    
    def test(self):
        """Run comprehensive testing"""
        logger.info("="*70)
        logger.info("TESTING MODEL")
        logger.info("="*70)
        
        self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        
        self.load_best_model()
        
        # Setup test data
        test_transform = transforms.Compose([
            transforms.Resize((self.config.TRAIN_SIZE, self.config.TRAIN_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        test_dataset = ImageDataset(self.config.DATASET_DIR / "test.csv", test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS
        )
        
        logger.info(f"Test set: {len(test_dataset):,} images\n")
        
        # Run inference
        logger.info("Running inference on test set...")
        test_preds = []
        test_labels = []
        test_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(self.config.DEVICE)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.numpy())
                test_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake
        
        # Calculate metrics
        self.calculate_all_metrics(test_labels, test_preds, test_probs)
        
        # Create visualizations
        self.plot_confusion_matrix(test_labels, test_preds)
        self.plot_roc_curve(test_labels, test_probs)
        self.plot_precision_recall_curve(test_labels, test_probs)
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def calculate_all_metrics(self, labels, preds, probs):
        """Calculate comprehensive metrics"""
        self.results = {
            'accuracy': float(accuracy_score(labels, preds) * 100),
            'precision': float(precision_score(labels, preds, zero_division=0) * 100),
            'recall': float(recall_score(labels, preds, zero_division=0) * 100),
            'f1': float(f1_score(labels, preds, zero_division=0) * 100),
            'auc_roc': float(roc_auc_score(labels, probs)),
            'average_precision': float(average_precision_score(labels, probs)),
            'confusion_matrix': confusion_matrix(labels, preds).tolist(),
            'classification_report': classification_report(labels, preds, target_names=['Real', 'Fake'])
        }
    
    def plot_confusion_matrix(self, labels, preds):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16}
        )
        plt.ylabel('Actual', fontsize=14)
        plt.xlabel('Predicted', fontsize=14)
        plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.config.PLOTS_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, labels, probs):
        """Plot and save ROC curve"""
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=3, color='blue')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.PLOTS_DIR / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, labels, probs):
        """Plot and save precision-recall curve"""
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.4f})', linewidth=3, color='green')
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.PLOTS_DIR / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save test results to JSON"""
        with open(self.config.RESULTS_DIR / 'test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to: {self.config.RESULTS_DIR / 'test_results.json'}")
    
    def print_summary(self):
        """Print test results summary"""
        logger.info("\n" + "="*70)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*70)
        logger.info(f"Accuracy:           {self.results['accuracy']:.2f}%")
        logger.info(f"Precision:          {self.results['precision']:.2f}%")
        logger.info(f"Recall:             {self.results['recall']:.2f}%")
        logger.info(f"F1 Score:           {self.results['f1']:.2f}%")
        logger.info(f"AUC-ROC:            {self.results['auc_roc']:.4f}")
        logger.info(f"Average Precision:  {self.results['average_precision']:.4f}")
        logger.info("="*70)
        logger.info("\nClassification Report:")
        logger.info("="*70)
        print(self.results['classification_report'])
        logger.info("="*70 + "\n")

# ============================================
# MODEL UPLOADER
# ============================================

class ModelUploader:
    """Upload trained model to HuggingFace"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_model_card(self) -> str:
        """Create model card README"""
        return f"""---
license: mit
tags:
- image-classification
- ai-detection
- sdxl
- deepfake-detection
library_name: pytorch
---

# SDXL Detector - ResNet50

Binary classifier for detecting AI-generated images from Stable Diffusion XL.

## Model Details

- **Architecture**: ResNet-50 (ImageNet pretrained)
- **Task**: Binary classification (Real vs Fake)
- **Training Data**: 10,000 real + 10,000 SDXL images
- **Input Size**: 256×256 RGB
- **Classes**: Real (0), Fake (1)

## Performance

See `test_results.json` for detailed metrics.

## Usage

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load model
model = models.resnet50()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(2048, 2)
)

checkpoint = torch.load('pytorch_model.bin')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('test.jpg').convert('RGB')
image = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(image)
    probs = torch.softmax(output, dim=1)
    pred = output.argmax(1).item()

print(f"Prediction: {{'Fake' if pred == 1 else 'Real'}}")
print(f"Confidence: {{probs[0][pred].item()*100:.2f}}%")
```

## Files

- `pytorch_model.bin`: Model weights
- `config.json`: Configuration
- `training_history.csv`: Training metrics
- `test_results.json`: Test results
- `*.png`: Visualizations

## Training

- Epochs: {self.config.NUM_EPOCHS}
- Batch Size: {self.config.BATCH_SIZE}
- Learning Rate: {self.config.LEARNING_RATE}
- Optimizer: AdamW
- Early Stopping: Patience {self.config.PATIENCE}

## Dataset

Generated images: [{self.config.HF_USERNAME}/{self.config.HF_DATASET_NAME}](https://huggingface.co/datasets/{self.config.HF_USERNAME}/{self.config.HF_DATASET_NAME})
"""
    
    def upload(self):
        """Upload model and artifacts to HuggingFace"""
        logger.info("="*70)
        logger.info("UPLOADING MODEL TO HUGGINGFACE")
        logger.info("="*70)
        
        try:
            api = HfApi(token=self.config.HF_TOKEN)
            repo_id = f"{self.config.HF_USERNAME}/{self.config.HF_REPO_NAME}"
            
            # Create repo
            logger.info(f"Creating repository: {repo_id}")
            create_repo(repo_id=repo_id, token=self.config.HF_TOKEN, private=False, exist_ok=True)
            
            # Prepare upload directory
            upload_dir = self.config.OUTPUT_DIR / "hf_upload"
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            logger.info("Preparing files...")
            shutil.copy(self.config.MODEL_DIR / "best.pth", upload_dir / "pytorch_model.bin")
            shutil.copy(self.config.MODEL_DIR / "training_history.csv", upload_dir / "training_history.csv")
            shutil.copy(self.config.RESULTS_DIR / "test_results.json", upload_dir / "test_results.json")
            
            # Copy plots if they exist
            for plot_name in ['training_history.png', 'confusion_matrix.png', 'roc_curve.png', 'precision_recall_curve.png']:
                plot_path = self.config.PLOTS_DIR / plot_name
                if plot_path.exists():
                    shutil.copy(plot_path, upload_dir / plot_name)
            
            # Create config
            config_dict = {
                "model_type": "resnet50",
                "task": "image-classification",
                "num_classes": 2,
                "class_names": ["real", "fake"],
                "image_size": self.config.TRAIN_SIZE,
                "generator_model": "stable-diffusion-xl-base-1.0",
                "random_seed": self.config.RANDOM_SEED
            }
            
            with open(upload_dir / "config.json", 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            # Create README
            with open(upload_dir / "README.md", 'w') as f:
                f.write(self.create_model_card())
            
            # Upload files
            logger.info("Uploading files to HuggingFace...")
            for file_path in upload_dir.iterdir():
                if file_path.is_file():
                    logger.info(f"  Uploading {file_path.name}...")
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=file_path.name,
                        repo_id=repo_id,
                        token=self.config.HF_TOKEN
                    )
            
            logger.info("\n" + "="*70)
            logger.info("✅ MODEL UPLOADED SUCCESSFULLY")
            logger.info("="*70)
            logger.info(f"🤗 https://huggingface.co/{repo_id}")
            logger.info("="*70 + "\n")
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")

# ============================================
# MAIN PIPELINE
# ============================================

def main():
    """
    Run the complete SDXL detector pipeline from start to finish.
    """
    print("\n" + "="*70)
    print("🚀 SDXL AI IMAGE DETECTOR - COMPLETE PIPELINE")
    print("="*70)
    print("\nThis pipeline will:")
    print("  1. Generate 10,000 SDXL images with diverse prompts")
    print("  2. Upload images to HuggingFace dataset")
    print("  3. Create stratified train/val/test splits")
    print("  4. Train ResNet-50 binary classifier")
    print("  5. Test model with comprehensive metrics")
    print("  6. Upload model to HuggingFace")
    print("\nEstimated time on L4 GPU:")
    print("  - Generation: ~17-19 hours (10 steps, ~6-7 sec/image)")
    print("  - Training: ~8-10 hours")
    print("  - Total: ~25-29 hours")
    print("\n" + "="*70)
    
    # Print configuration
    Config.print_config()
    
    # Confirm start
    response = input("▶️  Start pipeline? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Pipeline cancelled")
        return
    
    # Initialize
    start_time = datetime.now()
    config = Config()
    
    try:
        # Step 1: Generate images
        logger.info("\n🎨 STEP 1: IMAGE GENERATION")
        existing = list(config.GENERATED_DIR.glob("*.jpg")) if config.GENERATED_DIR.exists() else []
        generator = ImageGenerator(config)
        generator.generate_batch(start_idx=len(existing))
        
        # Step 2: Upload dataset
        logger.info("\n📤 STEP 2: DATASET UPLOAD")
        uploader = DatasetUploader(config)
        uploader.upload()
        
        # Step 3: Create splits
        logger.info("\n📊 STEP 3: DATASET SPLITS")
        splitter = DatasetSplitter(config)
        splitter.create_splits()
        
        # Step 4: Train model
        logger.info("\n🎓 STEP 4: MODEL TRAINING")
        trainer = Trainer(config)
        trainer.train()
        
        # Step 5: Test model
        logger.info("\n🧪 STEP 5: MODEL TESTING")
        tester = ModelTester(config)
        tester.test()
        
        # Step 6: Upload model
        logger.info("\n📤 STEP 6: MODEL UPLOAD")
        model_uploader = ModelUploader(config)
        model_uploader.upload()
        
        # Pipeline complete
        elapsed = datetime.now() - start_time
        
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETE!")
        print("="*70)
        print(f"⏱️  Total time: {elapsed}")
        print(f"📁 Output directory: {config.OUTPUT_DIR}")
        print(f"🤗 Model: https://huggingface.co/{config.HF_USERNAME}/{config.HF_REPO_NAME}")
        print(f"🤗 Dataset: https://huggingface.co/datasets/{config.HF_USERNAME}/{config.HF_DATASET_NAME}")
        print("="*70)
        print("\n✅ All artifacts saved successfully!")
        print(f"   - Generated images: {config.GENERATED_DIR}")
        print(f"   - Model weights: {config.MODEL_DIR}")
        print(f"   - Test results: {config.RESULTS_DIR}")
        print(f"   - Visualizations: {config.PLOTS_DIR}")
        print(f"   - Logs: {config.LOGS_DIR}")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Pipeline interrupted by user")
        logger.info("Progress has been saved and can be resumed")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        logger.info("\nPartial results may have been saved to:")
        logger.info(f"  {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
