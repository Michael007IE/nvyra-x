#!/usr/bin/env python3
"""
TEST ENSEMBLE MODEL ON BENCHMARK DATASET
=========================================
Comprehensive testing of your ensemble AI detector on the benchmark dataset.

Tests:
- Overall accuracy
- Per-generator accuracy
- False positive rate
- Confusion matrix
- Speed benchmarks
- Error analysis

Output:
- Detailed metrics
- Visual charts
- Error samples
- JSON results

Version: 1.0
Date: 2026-01-03
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, Counter
from PIL import Image
import torch

print("="*100)
print("🧪 TESTING YOUR ENSEMBLE MODEL")
print("="*100)
print("\n📊 Loading benchmark dataset and testing your model...")
print("   This will show you exactly how good your model is!\n")

# ============================================================================
# LOAD DATASET
# ============================================================================

print("📦 Loading FULL benchmark dataset from HuggingFace...")

try:
    from datasets import load_dataset
    
    # Load BOTH train and test splits for full dataset
    dataset_dict = load_dataset(
        "ash12321/ai-detector-benchmark-test-data"
    )
    
    # Combine train and test for full dataset
    from datasets import concatenate_datasets
    dataset = concatenate_datasets([dataset_dict['train'], dataset_dict['test']])
    
    print(f"   ✅ Loaded {len(dataset)} images (FULL DATASET)")
    print(f"      Train: {len(dataset_dict['train'])} images")
    print(f"      Test:  {len(dataset_dict['test'])} images")
    
except Exception as e:
    print(f"   ❌ Failed to load from HuggingFace: {e}")
    print(f"\n   Trying local dataset...")
    
    # Fallback: Load from local files
    benchmark_dir = Path("/home/zeus/benchmark_dataset")
    
    if not benchmark_dir.exists():
        print(f"   ❌ Local dataset not found at {benchmark_dir}")
        print(f"\n   Please either:")
        print(f"   1. Run build_benchmark_dataset.py first, OR")
        print(f"   2. Upload to HuggingFace with upload_benchmark_to_hf.py")
        sys.exit(1)
    
    # Create dataset from local files
    print(f"   Loading from {benchmark_dir}...")
    
    class LocalDataset:
        def __init__(self, benchmark_dir):
            self.data = []
            
            # Load AI images
            ai_dir = benchmark_dir / "ai_generated"
            for generator_dir in ai_dir.iterdir():
                if generator_dir.is_dir():
                    generator = generator_dir.name
                    for img_path in generator_dir.glob("*.png"):
                        self.data.append({
                            'image': img_path,
                            'label': 'ai',
                            'generator': generator,
                            'filename': img_path.name
                        })
            
            # Load real images
            real_dir = benchmark_dir / "real_images"
            for source_dir in real_dir.iterdir():
                if source_dir.is_dir():
                    source = source_dir.name
                    for img_path in source_dir.glob("*.png"):
                        self.data.append({
                            'image': img_path,
                            'label': 'real',
                            'generator': 'real',
                            'filename': img_path.name
                        })
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx].copy()
            # Load image if it's a path
            if isinstance(item['image'], Path):
                item['image'] = Image.open(item['image']).convert('RGB')
            return item
    
    dataset = LocalDataset(benchmark_dir)
    print(f"   ✅ Loaded {len(dataset)} images from local files")

# ============================================================================
# LOAD ENSEMBLE MODEL
# ============================================================================

print("\n🤖 Loading your Ensemble AI Detector from HuggingFace...")

try:
    # Download ensemble model from HuggingFace
    from huggingface_hub import snapshot_download
    import torch.nn as nn
    from transformers import ViTModel, ViTImageProcessor
    
    print("   Downloading model from HuggingFace...")
    model_dir = snapshot_download(
        repo_id="ash12321/ensemble-ai-detector",
        token=os.environ.get('HF_TOKEN')
    )
    
    model_dir = Path(model_dir)
    print(f"   ✅ Model downloaded to {model_dir}")
    
    # Define model architectures
    class FluxSDXLDetector(nn.Module):
        """ViT-Base detector for FLUX/SDXL"""
        def __init__(self):
            super().__init__()
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
            self.classifier = nn.Sequential(
                nn.Linear(768, 384), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(384, 192), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(192, 2)
            )
        
        def forward(self, pixel_values):
            outputs = self.vit(pixel_values=pixel_values)
            return self.classifier(outputs.pooler_output)
    
    class SmallModelDetector(nn.Module):
        """ViT-Small detector for Nano/SeeDream/ImageGBT"""
        def __init__(self):
            super().__init__()
            self.vit = ViTModel.from_pretrained("WinKawaks/vit-small-patch16-224")
            self.classifier = nn.Sequential(
                nn.Linear(384, 192), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(192, 96), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(96, 2)
            )
        
        def forward(self, pixel_values):
            outputs = self.vit(pixel_values=pixel_values)
            return self.classifier(outputs.pooler_output)
    
    # Create simple detector class
    class EnsembleDetector:
        """Simple ensemble detector"""
        
        def __init__(self, model_dir, device="cpu"):
            self.device = device
            self.models = {}
            self.processors = {}
            
            models_dir = model_dir / "models"
            
            # Load FLUX and SDXL
            for name in ['flux', 'sdxl']:
                model = FluxSDXLDetector()
                weights_path = models_dir / name / "pytorch_model.bin"
                model.load_state_dict(torch.load(weights_path, map_location=device))
                model.to(device)
                model.eval()
                self.models[name] = model
                self.processors[name] = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            
            # Load Nano, SeeDream, ImageGBT
            for name in ['nano', 'seedream', 'imagegbt']:
                model = SmallModelDetector()
                weights_path = models_dir / name / "pytorch_model.bin"
                model.load_state_dict(torch.load(weights_path, map_location=device))
                model.to(device)
                model.eval()
                self.models[name] = model
                self.processors[name] = ViTImageProcessor.from_pretrained("WinKawaks/vit-small-patch16-224")
        
        def detect(self, image, return_all_scores=False):
            """Detect if image is AI-generated"""
            predictions = {}
            
            for name, model in self.models.items():
                processor = self.processors[name]
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    logits = model(inputs['pixel_values'])
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    confidence = probs[0][1].item()
                
                predictions[name] = confidence
            
            # Find highest confidence
            max_confidence = max(predictions.values())
            
            # Determine if AI (threshold 0.7)
            if max_confidence >= 0.7:
                is_ai = True
                confidence = max_confidence
            else:
                is_ai = False
                confidence = 1.0 - max_confidence
            
            return {
                'is_ai_generated': is_ai,
                'confidence': confidence,
                'all_scores': predictions if return_all_scores else {}
            }
    
    # Initialize detector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = EnsembleDetector(model_dir, device=device)
    
    print(f"   ✅ Ensemble model loaded!")
    print(f"   Device: {device}")
    
except Exception as e:
    print(f"   ❌ Failed to load ensemble model: {e}")
    print(f"\n   Please make sure:")
    print(f"   1. The model is uploaded to HuggingFace: ash12321/ensemble-ai-detector")
    print(f"   2. Your HF_TOKEN is set correctly")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# RUN TESTS
# ============================================================================

print("\n" + "="*100)
print("🧪 TESTING MODEL ON BENCHMARK DATASET")
print("="*100)

results = {
    'predictions': [],
    'ground_truth': [],
    'generators': [],
    'filenames': [],
    'confidences': [],
    'inference_times': [],
}

errors = {
    'false_positives': [],  # Real images classified as AI
    'false_negatives': [],  # AI images classified as real
}

print(f"\n📊 Testing on {len(dataset)} images...")
print(f"   (This may take a few minutes)\n")

start_time = time.time()

for i, sample in enumerate(dataset):
    # Progress bar - more frequent updates for larger dataset
    if (i + 1) % 100 == 0 or i == 0:
        progress = (i + 1) / len(dataset) * 100
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (len(dataset) - i - 1)
        print(f"   Progress: {i+1}/{len(dataset)} ({progress:.1f}%) | "
              f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
    
    # Get image and ground truth
    image = sample['image']
    ground_truth = sample['label']
    generator = sample['generator']
    filename = sample['filename']
    
    # Test model
    img_start = time.time()
    result = detector.detect(image, return_all_scores=False)
    inference_time = time.time() - img_start
    
    # Convert result
    prediction = 'ai' if result['is_ai_generated'] else 'real'
    confidence = result['confidence']
    
    # Store results
    results['predictions'].append(prediction)
    results['ground_truth'].append(ground_truth)
    results['generators'].append(generator)
    results['filenames'].append(filename)
    results['confidences'].append(confidence)
    results['inference_times'].append(inference_time)
    
    # Track errors
    if prediction != ground_truth:
        error_info = {
            'filename': filename,
            'generator': generator,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'confidence': confidence
        }
        
        if ground_truth == 'real' and prediction == 'ai':
            errors['false_positives'].append(error_info)
        elif ground_truth == 'ai' and prediction == 'real':
            errors['false_negatives'].append(error_info)

total_time = time.time() - start_time

print(f"\n✅ Testing complete!")
print(f"   Total time: {total_time:.1f}s ({total_time/len(dataset)*1000:.1f}ms per image)")

# ============================================================================
# CALCULATE METRICS
# ============================================================================

print("\n" + "="*100)
print("📊 CALCULATING METRICS")
print("="*100)

# Overall metrics
total = len(results['predictions'])
correct = sum(1 for p, g in zip(results['predictions'], results['ground_truth']) if p == g)
accuracy = correct / total

# Separate AI and Real
ai_indices = [i for i, g in enumerate(results['ground_truth']) if g == 'ai']
real_indices = [i for i, g in enumerate(results['ground_truth']) if g == 'real']

ai_correct = sum(1 for i in ai_indices if results['predictions'][i] == 'ai')
real_correct = sum(1 for i in real_indices if results['predictions'][i] == 'real')

ai_detection_rate = ai_correct / len(ai_indices) if ai_indices else 0
real_accuracy = real_correct / len(real_indices) if real_indices else 0
false_positive_rate = 1 - real_accuracy

# Confusion matrix
tp = ai_correct  # True Positives (AI detected as AI)
fp = len(real_indices) - real_correct  # False Positives (Real detected as AI)
tn = real_correct  # True Negatives (Real detected as Real)
fn = len(ai_indices) - ai_correct  # False Negatives (AI detected as Real)

# Precision, Recall, F1
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Per-generator accuracy
generator_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
for pred, truth, gen in zip(results['predictions'], results['ground_truth'], results['generators']):
    generator_stats[gen]['total'] += 1
    if pred == truth:
        generator_stats[gen]['correct'] += 1

# Speed stats
avg_inference_time = sum(results['inference_times']) / len(results['inference_times'])
min_inference_time = min(results['inference_times'])
max_inference_time = max(results['inference_times'])

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("\n" + "="*100)
print("🎯 YOUR ENSEMBLE MODEL - TEST RESULTS")
print("="*100)

print(f"\n📊 OVERALL PERFORMANCE:")
print(f"   {'─'*60}")
print(f"   Total Images Tested:  {total}")
print(f"   Correct Predictions:  {correct}")
print(f"   Wrong Predictions:    {total - correct}")
print(f"   ")
print(f"   🎯 Overall Accuracy:   {accuracy*100:.2f}%")
print(f"   {'─'*60}")

print(f"\n🤖 AI DETECTION PERFORMANCE:")
print(f"   {'─'*60}")
print(f"   AI Images in Dataset: {len(ai_indices)}")
print(f"   AI Images Detected:   {ai_correct}")
print(f"   AI Images Missed:     {len(ai_indices) - ai_correct}")
print(f"   ")
print(f"   ✅ AI Detection Rate:  {ai_detection_rate*100:.2f}%")
print(f"   ❌ Miss Rate:          {(1-ai_detection_rate)*100:.2f}%")
print(f"   {'─'*60}")

print(f"\n📷 REAL IMAGE PERFORMANCE:")
print(f"   {'─'*60}")
print(f"   Real Images in Dataset: {len(real_indices)}")
print(f"   Real Images Correct:    {real_correct}")
print(f"   Real Images Wrong:      {fp}")
print(f"   ")
print(f"   ✅ Real Image Accuracy: {real_accuracy*100:.2f}%")
print(f"   ⚠️  False Positive Rate: {false_positive_rate*100:.2f}%")
print(f"   {'─'*60}")

print(f"\n🎯 CLASSIFICATION METRICS:")
print(f"   {'─'*60}")
print(f"   Precision:  {precision*100:.2f}%  (When you say 'AI', you're right {precision*100:.1f}% of the time)")
print(f"   Recall:     {recall*100:.2f}%  (You catch {recall*100:.1f}% of all AI images)")
print(f"   F1 Score:   {f1_score*100:.2f}%  (Balanced measure)")
print(f"   {'─'*60}")

print(f"\n📊 CONFUSION MATRIX:")
print(f"   {'─'*60}")
print(f"   ")
print(f"                    Predicted AI    Predicted Real")
print(f"   Actual AI          {tp:4d}            {fn:4d}        ({recall*100:.1f}% recall)")
print(f"   Actual Real        {fp:4d}            {tn:4d}        ({precision*100:.1f}% precision)")
print(f"   ")
print(f"   True Positives:  {tp:4d}  ✅ Correctly detected AI")
print(f"   False Positives: {fp:4d}  ⚠️  Real images flagged as AI")
print(f"   True Negatives:  {tn:4d}  ✅ Correctly identified real")
print(f"   False Negatives: {fn:4d}  ❌ AI images missed")
print(f"   {'─'*60}")

print(f"\n🎨 ACCURACY BY GENERATOR:")
print(f"   {'─'*60}")
for gen in sorted(generator_stats.keys()):
    stats = generator_stats[gen]
    gen_accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
    
    # Visual bar
    bar_length = int(gen_accuracy / 2)  # Scale to 50 chars max
    bar = '█' * bar_length + '░' * (50 - bar_length)
    
    print(f"   {gen:15s}  {bar}  {gen_accuracy:5.1f}%  ({stats['correct']}/{stats['total']})")
print(f"   {'─'*60}")

print(f"\n⚡ SPEED PERFORMANCE:")
print(f"   {'─'*60}")
print(f"   Average: {avg_inference_time*1000:.1f}ms per image")
print(f"   Min:     {min_inference_time*1000:.1f}ms")
print(f"   Max:     {max_inference_time*1000:.1f}ms")
print(f"   Total:   {total_time:.1f}s for {total} images")
print(f"   {'─'*60}")

# ============================================================================
# ERROR ANALYSIS
# ============================================================================

if errors['false_positives'] or errors['false_negatives']:
    print(f"\n" + "="*100)
    print("🔍 ERROR ANALYSIS")
    print("="*100)
    
    if errors['false_positives']:
        print(f"\n⚠️  FALSE POSITIVES (Real images flagged as AI): {len(errors['false_positives'])}")
        print(f"   {'─'*60}")
        
        # Group by source
        fp_by_source = defaultdict(list)
        for error in errors['false_positives']:
            fp_by_source[error['generator']].append(error)
        
        for source, source_errors in sorted(fp_by_source.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n   {source}: {len(source_errors)} images")
            for i, error in enumerate(source_errors[:5], 1):  # Show first 5
                print(f"      {i}. {error['filename']} (confidence: {error['confidence']:.1%})")
            if len(source_errors) > 5:
                print(f"      ... and {len(source_errors) - 5} more")
    
    if errors['false_negatives']:
        print(f"\n❌ FALSE NEGATIVES (AI images missed): {len(errors['false_negatives'])}")
        print(f"   {'─'*60}")
        
        # Group by generator
        fn_by_gen = defaultdict(list)
        for error in errors['false_negatives']:
            fn_by_gen[error['generator']].append(error)
        
        for gen, gen_errors in sorted(fn_by_gen.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n   {gen}: {len(gen_errors)} images")
            for i, error in enumerate(gen_errors[:5], 1):  # Show first 5
                print(f"      {i}. {error['filename']} (confidence: {error['confidence']:.1%})")
            if len(gen_errors) > 5:
                print(f"      ... and {len(gen_errors) - 5} more")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\n" + "="*100)
print("💾 SAVING RESULTS")
print("="*100)

output_dir = Path("/home/zeus/benchmark_results")
output_dir.mkdir(exist_ok=True)

# Save JSON results
results_json = {
    'model': 'Ensemble AI Detector',
    'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_size': total,
    
    'overall': {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    },
    
    'ai_detection': {
        'detection_rate': ai_detection_rate,
        'total_ai': len(ai_indices),
        'detected': ai_correct,
        'missed': len(ai_indices) - ai_correct
    },
    
    'real_images': {
        'accuracy': real_accuracy,
        'false_positive_rate': false_positive_rate,
        'total_real': len(real_indices),
        'correct': real_correct,
        'wrong': fp
    },
    
    'metrics': {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    },
    
    'confusion_matrix': {
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    },
    
    'by_generator': {
        gen: {
            'accuracy': stats['correct'] / stats['total'],
            'correct': stats['correct'],
            'total': stats['total']
        }
        for gen, stats in generator_stats.items()
    },
    
    'speed': {
        'average_ms': avg_inference_time * 1000,
        'min_ms': min_inference_time * 1000,
        'max_ms': max_inference_time * 1000,
        'total_seconds': total_time
    },
    
    'errors': {
        'false_positives': errors['false_positives'],
        'false_negatives': errors['false_negatives']
    }
}

json_path = output_dir / "ensemble_test_results.json"
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"\n   ✅ JSON results: {json_path}")

# Save CSV with all predictions
csv_path = output_dir / "ensemble_predictions.csv"
with open(csv_path, 'w') as f:
    f.write("filename,ground_truth,prediction,generator,confidence,inference_time_ms,correct\n")
    for i in range(len(results['predictions'])):
        correct_pred = results['predictions'][i] == results['ground_truth'][i]
        f.write(f"{results['filenames'][i]},{results['ground_truth'][i]},{results['predictions'][i]},"
                f"{results['generators'][i]},{results['confidences'][i]:.4f},"
                f"{results['inference_times'][i]*1000:.2f},{correct_pred}\n")

print(f"   ✅ CSV predictions: {csv_path}")

# Save text summary
summary_path = output_dir / "ensemble_summary.txt"
with open(summary_path, 'w') as f:
    f.write("="*100 + "\n")
    f.write("ENSEMBLE AI DETECTOR - TEST SUMMARY\n")
    f.write("="*100 + "\n\n")
    
    f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {total} images\n\n")
    
    f.write(f"OVERALL ACCURACY: {accuracy*100:.2f}%\n")
    f.write(f"AI Detection Rate: {ai_detection_rate*100:.2f}%\n")
    f.write(f"False Positive Rate: {false_positive_rate*100:.2f}%\n\n")
    
    f.write(f"Precision: {precision*100:.2f}%\n")
    f.write(f"Recall: {recall*100:.2f}%\n")
    f.write(f"F1 Score: {f1_score*100:.2f}%\n\n")
    
    f.write("Per-Generator Accuracy:\n")
    for gen in sorted(generator_stats.keys()):
        stats = generator_stats[gen]
        gen_acc = stats['correct'] / stats['total'] * 100
        f.write(f"  {gen:15s}: {gen_acc:5.1f}% ({stats['correct']}/{stats['total']})\n")

print(f"   ✅ Text summary: {summary_path}")

# ============================================================================
# FINAL VERDICT
# ============================================================================

print(f"\n" + "="*100)
print("🎊 FINAL VERDICT")
print("="*100)

if accuracy >= 0.95:
    verdict = "🌟 EXCELLENT! Your model is production-ready!"
elif accuracy >= 0.90:
    verdict = "✅ GREAT! Your model performs very well!"
elif accuracy >= 0.85:
    verdict = "👍 GOOD! Your model is solid!"
elif accuracy >= 0.80:
    verdict = "⚠️  OKAY! Some room for improvement."
else:
    verdict = "❌ NEEDS WORK! Consider retraining or adjusting thresholds."

print(f"\n{verdict}")

print(f"\n📊 Your Results:")
print(f"   Overall Accuracy:  {accuracy*100:.2f}%")
print(f"   AI Detection:      {ai_detection_rate*100:.2f}%")
print(f"   FPR:               {false_positive_rate*100:.2f}%")
print(f"   F1 Score:          {f1_score*100:.2f}%")

print(f"\n💾 Results saved to: {output_dir}")

print(f"\n🎯 Ready to compare with other models?")
print(f"   Your baseline: {accuracy*100:.2f}% accuracy")
print(f"   Let's see how Gemini, Claude, Hive, etc. compare!")

print("\n" + "="*100)
print("✅ TESTING COMPLETE!")
print("="*100)
