#!/usr/bin/env python3
"""
Combines 5 specialized detectors into a unified AI detection system.

Models:
1. FLUX Detector (99.81% accuracy)
2. SDXL Detector (99.81% accuracy)
3. Nano Banana Pro Detector (95-97% accuracy)
4. SeeDream 4.5 Detector (98.4% accuracy)
5. ImageGBT 1.5 Detector (100% test accuracy)

Coverage: ~80% of known AI image generators

How it works:
1. Each model independently evaluates the image
2. Returns confidence score for its generator
3. Ensemble decides:
   - If ANY model confidence > threshold → AI-generated
   - Generator = model with highest confidence
   - If ALL confidences low → REAL image

Version: 1.0 - Production Grade
Date: 2026-01-02
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
from huggingface_hub import hf_hub_download

os.environ['HF_TOKEN'] = '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'

print("Custom Deepfake Detection Platform")
print("\n Loading deefake detection system...")
print(" Covers: FLUX, SDXL, Nano, SeeDream, ImageGBT (~80% of market)\n")

class AIGenerator(Enum):
    """Supported AI generators"""
    FLUX = "FLUX 1.0 Dev"
    SDXL = "Stable Diffusion XL"
    NANO = "Nano Banana Pro"
    SEEDREAM = "SeeDream 4.5"
    IMAGEGBT = "ImageGBT 1.5"
    REAL = "Real Image (Not AI)"
    UNKNOWN = "Unknown AI Generator"

@dataclass
class ModelPrediction:
    generator: AIGenerator
    confidence: float
    is_ai: bool
    model_name: str

@dataclass
class EnsembleResult:
    is_ai_generated: bool
    generator: AIGenerator
    confidence: float
    certainty_level: str 
    all_predictions: Dict[str, ModelPrediction]
    reasoning: str
    detection_time: float

class FluxSDXLDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(192, 2)
        )
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits

class SmallModelDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("WinKawaks/vit-small-patch16-224")
        self.classifier = nn.Sequential(
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(96, 2)
        )
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits
       
# Specalised Detection Wrapper
class SpecializedDetector:
    def __init__(self, generator: AIGenerator, model_class, 
                 repo_id: str, device: str = "cuda"):
        self.generator = generator
        self.device = device
        print(f"   Loading {generator.value} detector...")
        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename="pytorch_model.bin",
                token=os.environ.get('HF_TOKEN')
            )
            self.model = model_class()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.to(device)
            self.model.eval()
            
            # Load appropriate processor
            if model_class == FluxSDXLDetector:
                self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            else:
                self.processor = ViTImageProcessor.from_pretrained("WinKawaks/vit-small-patch16-224")
            
            print(f" Yes {generator.value} ready!")
            
        except Exception as e:
            print(f" No, failed to load {generator.value}: {e}")
            raise
    
    def predict(self, image: Image.Image) -> ModelPrediction:
        """Predict if image is from this generator"""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(inputs['pixel_values'])
            probs = torch.nn.functional.softmax(logits, dim=1)
            confidence = probs[0][1].item()
            is_ai = confidence > 0.5
        
        return ModelPrediction(
            generator=self.generator,
            confidence=confidence,
            is_ai=is_ai,
            model_name=self.generator.value
        )
class EnsembleAIDetector:
    """Unified ensemble detector combining all 5 models"""
    
    def __init__(self, device: str = None, 
                 confidence_threshold: float = 0.7,
                 high_confidence_threshold: float = 0.9):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        
        print(f"\n🔧 Initializing Ensemble Detector...")
        print(f"   Device: {self.device}")
        print(f"   Confidence Threshold: {confidence_threshold:.0%}")
        print(f"   High Confidence Threshold: {high_confidence_threshold:.0%}")
        
        # Load all 5 detectors
        print(f"\n Loading 5 specialized detectors:")
        
        self.detectors = {
            'flux': SpecializedDetector(
                AIGenerator.FLUX,
                FluxSDXLDetector,
                "ash12321/flux-detector-final",
                self.device
            ),
            'sdxl': SpecializedDetector(
                AIGenerator.SDXL,
                FluxSDXLDetector,
                "ash12321/sdxl-detector-final",
                self.device
            ),
            'nano': SpecializedDetector(
                AIGenerator.NANO,
                SmallModelDetector,
                "ash12321/nano-banana-pro-detector-final",
                self.device
            ),
            'seedream': SpecializedDetector(
                AIGenerator.SEEDREAM,
                SmallModelDetector,
                "ash12321/seedream-4.5-detector-final",
                self.device
            ),
            'imagegbt': SpecializedDetector(
                AIGenerator.IMAGEGBT,
                SmallModelDetector,
                "ash12321/imagegbt-1.5-detector-final",
                self.device
            ),
        }
        
        print(f"\n Models Loaded")
    
    def _get_certainty_level(self, confidence: float) -> str:
        """Map confidence to certainty level"""
        if confidence >= 0.95:
            return "Very High"
        elif confidence >= 0.85:
            return "High"
        elif confidence >= 0.70:
            return "Medium"
        else:
            return "Low"
    
    def _build_reasoning(self, predictions: Dict[str, ModelPrediction], 
                        final_generator: AIGenerator, final_confidence: float) -> str:
        """Build human-readable reasoning"""
        
        if final_generator == AIGenerator.REAL:
            high_conf_models = [p.model_name for p in predictions.values() 
                              if not p.is_ai and p.confidence > 0.5]
            
            if len(high_conf_models) == 5:
                return f"All 5 detectors agree this is a real image (unanimous verdict)."
            else:
                max_ai_conf = max(p.confidence for p in predictions.values())
                return f"No detector strongly identified this as AI-generated. Max AI confidence: {max_ai_conf:.1%}. Likely real."
        else:
            detecting_model = final_generator.value
            other_detections = [p.model_name for p in predictions.values() 
                               if p.is_ai and p.generator != final_generator]
            if other_detections:
                return f"{detecting_model} detector identified this with {final_confidence:.1%} confidence. Also flagged by: {', '.join(other_detections)}."
            else:
                return f"{detecting_model} detector identified this with {final_confidence:.1%} confidence. Other detectors found no match."
    
    def detect(self, image: Image.Image, return_all_scores: bool = False) -> EnsembleResult:
        """
        Detect if image is AI-generated and identify the generator.
        
        Args:
            image: PIL Image to analyze
            return_all_scores: If True, includes all model predictions in result
        
        Returns:
            EnsembleResult with detection verdict and details
        """
        
        start_time = time.time()
        predictions = {}
        for name, detector in self.detectors.items():
            predictions[name] = detector.predict(image)
        max_prediction = max(predictions.values(), key=lambda p: p.confidence)
        if max_prediction.confidence >= self.confidence_threshold:
            is_ai = True
            generator = max_prediction.generator
            confidence = max_prediction.confidence
        else:
            is_ai = False
            generator = AIGenerator.REAL
            confidence = 1.0 - max_prediction.confidence
        certainty_level = self._get_certainty_level(confidence)
        reasoning = self._build_reasoning(predictions, generator, confidence)
        detection_time = time.time() - start_time
        
        result = EnsembleResult(
            is_ai_generated=is_ai,
            generator=generator,
            confidence=confidence,
            certainty_level=certainty_level,
            all_predictions=predictions if return_all_scores else {},
            reasoning=reasoning,
            detection_time=detection_time
        )
        
        return result
    
    def detect_batch(self, images: List[Image.Image], 
                    show_progress: bool = True) -> List[EnsembleResult]:
        """Detect multiple images"""
        
        results = []
        for i, image in enumerate(images):
            if show_progress and (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(images)} images...")
            result = self.detect(image)
            results.append(result)
        return results
    
    def print_result(self, result: EnsembleResult):
        print("Detection Result")
        
        if result.is_ai_generated:
            print(f"\n This image appears to be generated by Artificial Intelligence (AI).")
            print(f" Generator: {result.generator.value}")
            print(f" Confidence: {result.confidence:.1%}")
            print(f" Certainty: {result.certainty_level}")
        else:
            print(f"\n This image appears to be real. ")
            print(f" Confidence: {result.confidence:.1%} (that it's real)")
            print(f" Certainty: {result.certainty_level}")
        
        print(f"\n💭 Reasoning: {result.reasoning}")
        print(f"⏱️  Detection Time: {result.detection_time*1000:.1f}ms")
        
        if result.all_predictions:
            print(f"\n📋 Individual Model Scores:")
            for name, pred in result.all_predictions.items():
                status = "yes" if pred.is_ai else "no"
                print(f"   {status} {pred.model_name:20s}: {pred.confidence:.1%}")
def detect_image_file(image_path: str, detector: EnsembleAIDetector = None) -> EnsembleResult:
    if detector is None:
        detector = EnsembleAIDetector()
    image = Image.open(image_path).convert('RGB')
    result = detector.detect(image, return_all_scores=True)
    return result
def detect_image_url(url: str, detector: EnsembleAIDetector = None) -> EnsembleResult:
    import requests
    from io import BytesIO
    if detector is None:
        detector = EnsembleAIDetector()
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    result = detector.detect(image, return_all_scores=True)
    return result
def batch_detect_folder(folder_path: str, detector: EnsembleAIDetector = None) -> Dict:    
    if detector is None:
        detector = EnsembleAIDetector()
    folder = Path(folder_path)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        image_files.extend(folder.glob(ext))
    print(f"\n Processing {len(image_files)} images from {folder_path}...")
    results = {}
    
    for img_path in image_files:
        try:
            image = Image.open(img_path).convert('RGB')
            result = detector.detect(image)
            results[img_path.name] = result
        except Exception as e:
            print(f" Failed to process {img_path.name}: {e}")
    
    return results
   
# Staistics and Reporting
def generate_report(results: List[EnsembleResult]) -> Dict:
    """Generate statistics report from batch results"""
    
    total = len(results)
    ai_generated = sum(1 for r in results if r.is_ai_generated)
    real_images = total - ai_generated
    generator_counts = {}
    for result in results:
        if result.is_ai_generated:
            gen = result.generator.value
            generator_counts[gen] = generator_counts.get(gen, 0) + 1
    avg_confidence = sum(r.confidence for r in results) / total if total > 0 else 0
    high_confidence = sum(1 for r in results if r.confidence >= 0.9)
    
    report = {
        'total_images': total,
        'ai_generated': ai_generated,
        'real_images': real_images,
        'ai_percentage': ai_generated / total * 100 if total > 0 else 0,
        'generator_breakdown': generator_counts,
        'average_confidence': avg_confidence,
        'high_confidence_detections': high_confidence,
        'avg_detection_time_ms': sum(r.detection_time for r in results) / total * 1000 if total > 0 else 0
    }
    
    return report

def print_report(report: Dict):
    print(" Detection Report")
    
    print(f"\n Summary:")
    print(f"  Total Images: {report['total_images']}")
    print(f"  AI-Generated: {report['ai_generated']} ({report['ai_percentage']:.1f}%)")
    print(f"  Real Images:  {report['real_images']} ({100-report['ai_percentage']:.1f}%)")
    
    if report['generator_breakdown']:
        print(f"\n AI Generator Breakdown:")
        for gen, count in sorted(report['generator_breakdown'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {gen:20s}: {count:3d} images")
    
    print(f"\n Confidence:")
    print(f"   Average: {report['average_confidence']:.1%}")
    print(f"   High Confidence (>90%): {report['high_confidence_detections']}")
    print(f"\n Performance:")
    print(f"   Avg Detection Time: {report['avg_detection_time_ms']:.1f}ms per image")

# Basic Testing

def main():
   print("Example Usage:")
    detector = EnsembleAIDetector(
        confidence_threshold=0.7,  
        high_confidence_threshold=0.9  
    )   
    print("\n  1. Detect single image from file:")
    print("   ```python")
    print("   result = detector.detect(Image.open('image.jpg'))")
    print("   detector.print_result(result)")
    print("   ```")
    
    print("\n 2. Detect from URL:")
    print("   ```python")
    print("   result = detect_image_url('https://example.com/image.jpg')")
    print("   ```")
    
    print("\n 3️. Batch detect folder:")
    print("   ```python")
    print("   results = batch_detect_folder('/path/to/images/')")
    print("   report = generate_report(list(results.values()))")
    print("   print_report(report)")
    print("   ```")
    
    print("\n 4. Get all model scores:")
    print("   ```python")
    print("   result = detector.detect(image, return_all_scores=True)")
    print("   for name, pred in result.all_predictions.items():")
    print("       print(f'{name}: {pred.confidence:.1%}')")
    print("   ```")
    
    print("\n 5. Adjust confidence threshold:")
    print("   ```python")
    print("   # More strict (fewer false positives)")
    print("   detector = EnsembleAIDetector(confidence_threshold=0.85)")
    print("   ")
    print("   # More lenient (catch more AI images)")
    print("   detector = EnsembleAIDetector(confidence_threshold=0.60)")
    print("   ```")
    
    print("API Reference")
    
    print("\nEnsembleResult fields:")
    print("   - is_ai_generated: bool")
    print("   - generator: AIGenerator enum")
    print("   - confidence: float (0-1)")
    print("   - certainty_level: str ('Very High', 'High', 'Medium', 'Low')")
    print("   - all_predictions: Dict[str, ModelPrediction]")
    print("   - reasoning: str")
    print("   - detection_time: float (seconds)")
    
    print("\nModelPrediction fields:")
    print("   - generator: AIGenerator")
    print("   - confidence: float (0-1)")
    print("   - is_ai: bool")
    print("   - model_name: str")
    
    print("Ensembe Deepfake Detection System Ready")
    
    print("\n Coverage of market:")
    print("FLUX 1.0 Dev (99.81% accuracy)")
    print("Stable Diffusion XL (99.81% accuracy)")
    print("Nano Banana Pro (95-97% accuracy)")
    print("SeeDream 4.5 (98.4% accuracy)")
    print("mageGBT 1.5 (100% test accuracy)")
    print(" \n Combined: ~80% of AI image generator market")
    print("Ensemble accuracy: Higher than any individual model")
    print("Speed: ~50-100ms per image on GPU")
    
    print("\n Ready to detect AI images!")

if __name__ == "__main__":
    main()
