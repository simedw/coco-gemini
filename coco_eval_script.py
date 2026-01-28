#!/usr/bin/env python3
"""
COCO Evaluation Script with FiftyOne and Gemini Predictions
"""

import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.types as fot
from PIL import Image
import sys
from io import StringIO

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, try to load .env manually
    if os.path.exists('.env'):
        with open('.env') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

# Import our modular components
from gemini_model import GeminiDetector

# --- CONFIGURATION ---
MAX_IMAGES = 50  # Keep it small for testing


def load_coco_class_names():
    """Load COCO class names from JSON file."""
    try:
        with open("coco_classes.json", "r") as f:
            coco_classes = json.load(f)
        # Convert string keys to integers for consistency
        return {int(k): v for k, v in coco_classes.items()}
    except FileNotFoundError:
        raise FileNotFoundError(
            "coco_classes.json not found. Please ensure it's in the current directory."
        )


def create_run_directory(model_name: str) -> Path:
    """Create a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name.replace('/', '_')}_{timestamp}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, args: argparse.Namespace, detector_info: dict):
    """Save run configuration to config.json."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model,
        "thinking_budget": args.thinking_budget,
        "max_images": args.max_images,
        "max_workers": args.max_workers,
        "preprocess_images": args.preprocess_images,
        "structured_output": args.structured_output,
        "code_execution": args.code_execution,
        "detector_info": detector_info
    }
    
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def capture_coco_evaluation(coco_gt, coco_dt):
    """Capture COCO evaluation metrics and return as structured data."""
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    
    # Capture the summarize output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    evaluator.summarize()
    sys.stdout = old_stdout
    summary_text = captured_output.getvalue()
    
    # Extract metrics from evaluator.stats
    # evaluator.stats contains [AP@0.5:0.95, AP@0.5, AP@0.75, AP@0.5:0.95 small, AP@0.5:0.95 medium, AP@0.5:0.95 large, AR@0.5:0.95 max1, AR@0.5:0.95 max10, AR@0.5:0.95 max100, AR@0.5:0.95 small, AR@0.5:0.95 medium, AR@0.5:0.95 large]
    metrics = {
        "ap_50_95": evaluator.stats[0] if len(evaluator.stats) > 0 else None,
        "ap_50": evaluator.stats[1] if len(evaluator.stats) > 1 else None,
        "ap_75": evaluator.stats[2] if len(evaluator.stats) > 2 else None,
        "ap_50_95_small": evaluator.stats[3] if len(evaluator.stats) > 3 else None,
        "ap_50_95_medium": evaluator.stats[4] if len(evaluator.stats) > 4 else None,
        "ap_50_95_large": evaluator.stats[5] if len(evaluator.stats) > 5 else None,
        "ar_50_95_max1": evaluator.stats[6] if len(evaluator.stats) > 6 else None,
        "ar_50_95_max10": evaluator.stats[7] if len(evaluator.stats) > 7 else None,
        "ar_50_95_max100": evaluator.stats[8] if len(evaluator.stats) > 8 else None,
        "ar_50_95_small": evaluator.stats[9] if len(evaluator.stats) > 9 else None,
        "ar_50_95_medium": evaluator.stats[10] if len(evaluator.stats) > 10 else None,
        "ar_50_95_large": evaluator.stats[11] if len(evaluator.stats) > 11 else None,
        "summary_text": summary_text
    }
    
    return metrics


def export_ground_truth_with_correct_ids(dataset, gt_path, coco_class_names):
    """
    Export ground truth annotations maintaining proper COCO category IDs
    instead of using FiftyOne's compressed ID mapping
    """
    # Create reverse mapping: class_name -> coco_id
    name_to_id = {name: int(coco_id) for coco_id, name in coco_class_names.items()}
    
    # COCO format structure
    coco_data = {
        "info": {
            "year": 2017,
            "version": "1.0", 
            "contributor": "COCO Consortium",
            "url": "http://cocodataset.org",
            "date_created": "2017/09/01"
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # Collect all unique categories used in the dataset
    used_categories = set()
    
    # First pass: collect all categories and create images list
    for i, sample in enumerate(dataset):
        # Add image info
        img_width = sample.metadata.width
        img_height = sample.metadata.height
        img_filename = Path(sample.filepath).name
        
        coco_data["images"].append({
            "id": i + 1,
            "file_name": img_filename,
            "height": img_height,
            "width": img_width,
            "license": None,
            "coco_url": None
        })
        
        # Collect categories from ground truth
        if hasattr(sample, 'ground_truth') and sample.ground_truth:
            for detection in sample.ground_truth.detections:
                if detection.label in name_to_id:
                    used_categories.add(detection.label)
    
    # Create categories list with proper COCO IDs
    for class_name in sorted(used_categories):
        coco_id = name_to_id[class_name]
        coco_data["categories"].append({
            "id": coco_id,
            "name": class_name,
            "supercategory": None
        })
    
    # Second pass: create annotations
    annotation_id = 1
    for i, sample in enumerate(dataset):
        image_id = i + 1
        img_width = sample.metadata.width
        img_height = sample.metadata.height
        
        if hasattr(sample, 'ground_truth') and sample.ground_truth:
            for detection in sample.ground_truth.detections:
                if detection.label in name_to_id:
                    # Convert relative coordinates to absolute COCO format
                    rel_x, rel_y, rel_w, rel_h = detection.bounding_box
                    abs_x = rel_x * img_width
                    abs_y = rel_y * img_height  
                    abs_w = rel_w * img_width
                    abs_h = rel_h * img_height
                    
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": name_to_id[detection.label],
                        "bbox": [abs_x, abs_y, abs_w, abs_h],  # COCO format: [x, y, width, height]
                        "area": abs_w * abs_h,
                        "iscrowd": 0,
                        "supercategory": detection.label
                    })
                    annotation_id += 1
    
    # Save to file
    with open(gt_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"📊 Exported {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations, {len(coco_data['categories'])} categories")


def main(args):
    print("🚀 Starting COCO Evaluation Script")
    
    # Create run directory
    run_dir = create_run_directory(args.model)
    print(f"📁 Created run directory: {run_dir}")
    
    # Set up file paths within the run directory
    gt_path = run_dir / "ground_truth.json"
    pred_path = run_dir / "predictions.json"
    
    print(f"📁 Using files: {gt_path} | {pred_path}")
    
    # Load COCO class names
    print("📋 Loading COCO class names...")
    COCO_CLASS_NAMES = load_coco_class_names()
    
    # Initialize Gemini detector
    print("🤖 Initializing Gemini detector...")
    try:
        detector = GeminiDetector(
            api_key=args.api_key,
            model_name=args.model,
            thinking_budget=args.thinking_budget,
            max_workers=args.max_workers,
            preprocess_images=args.preprocess_images,
            use_structured_output=args.structured_output,
            use_code_execution=args.code_execution
        )
        detector_info = detector.get_model_info()
        print(f"✅ Gemini detector initialized: {detector_info}")
        print(f"🔧 Parallel processing: {args.max_workers} workers")
        print(f"📐 Image preprocessing: {'enabled' if args.preprocess_images else 'disabled'}")
        print(f"🧠 Thinking budget: {args.thinking_budget}")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini detector: {e}")
        print("💡 Make sure you have set GEMINI_API_KEY environment variable or use --api-key")
        print("💡 Install Gemini dependencies with: uv sync --extra gemini")
        return
    
    # Save configuration
    config_path = save_config(run_dir, args, detector_info)
    print(f"💾 Saved configuration to {config_path}")
    
    # --- STEP 1: Load COCO Validation Set ---
    print(f"📥 Loading COCO validation dataset (max {args.max_images} images)...")
    dataset = foz.load_zoo_dataset(
        "coco-2017", split="validation", max_samples=args.max_images, shuffle=True
    )
    print(f"✅ Loaded {len(dataset)} images")

    # --- STEP 2: Export Ground Truth Annotations ---
    print("📝 Exporting ground truth annotations...")
    export_ground_truth_with_correct_ids(dataset, gt_path, COCO_CLASS_NAMES)
    print(f"✅ Ground truth exported to {gt_path}")

    # --- STEP 3: Run Gemini Object Detection ---
    print("🤖 Running Gemini object detection...")
    
    # Prepare image data for parallel processing
    image_data = [(sample.filepath, i + 1) for i, sample in enumerate(dataset)]
    
    # Track timing
    start_time = time.time()
    
    # Run parallel detection
    all_preds, stats = detector.detect_parallel(image_data)
    
    # Calculate timing
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_image = total_time / len(image_data) if len(image_data) > 0 else 0
    
    # Enhanced statistics
    enhanced_stats = {
        **stats,
        "total_time_seconds": total_time,
        "average_time_per_image": avg_time_per_image,
        "images_per_second": len(image_data) / total_time if total_time > 0 else 0
    }
    
    # Print statistics
    print(f"\n📊 Detection Statistics:")
    print(f"  ✅ Successful images: {stats['successful_images']}/{stats['total_images']}")
    print(f"  ❌ Failed images: {stats['failed_images']}")
    print(f"  🎯 Total predictions: {stats['total_predictions']}")
    print(f"  ⏱️  Total time: {total_time:.2f} seconds")
    print(f"  ⚡ Average time per image: {avg_time_per_image:.2f} seconds")
    print(f"  🔥 Images per second: {enhanced_stats['images_per_second']:.2f}")
    
    if stats['failed_images'] > 0:
        print(f"  ⚠️  Failed image IDs: {stats['failed_image_ids'][:10]}{'...' if len(stats['failed_image_ids']) > 10 else ''}")
    
    if stats['successful_images'] == 0:
        print("❌ No successful detections! Check API key and network connection.")
        return

    with open(pred_path, "w") as f:
        json.dump(all_preds, f, indent=2)
    
    print(f"✅ Generated {len(all_preds)} predictions saved to {pred_path}")

    # --- STEP 4: Evaluate using pycocotools ---
    print("📊 Evaluating predictions using pycocotools...")
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))

    # Capture evaluation metrics
    metrics = capture_coco_evaluation(coco_gt, coco_dt)
    
    print("\n" + "="*50)
    print("🏆 EVALUATION RESULTS")
    print("="*50)
    print(metrics['summary_text'])
    print("="*50)
    
    # Save results
    results = {
        "metrics": metrics,
        "statistics": enhanced_stats,
        "evaluation_completed_at": datetime.now().isoformat()
    }
    
    results_path = run_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to {results_path}")

    # --- STEP 5: Add predictions to dataset for visualization ---
    print("📋 Adding predictions to dataset for visualization...")
    
    # Group predictions by image_id for easy lookup
    predictions_by_image = {}
    for pred in all_preds:
        image_id = pred["image_id"]
        if image_id not in predictions_by_image:
            predictions_by_image[image_id] = []
        predictions_by_image[image_id].append(pred)
    
    # Add predictions to each sample in the dataset
    for i, sample in enumerate(dataset):
        image_id = i + 1  # Sequential ID matching our prediction generation
        
        detections = []
        if image_id in predictions_by_image:
            # Get image dimensions for normalization
            img_width = sample.metadata.width
            img_height = sample.metadata.height
            
            for pred in predictions_by_image[image_id]:
                # Convert absolute coordinates to relative coordinates for FiftyOne
                x, y, w, h = pred["bbox"]
                rel_x = x / img_width
                rel_y = y / img_height
                rel_w = w / img_width
                rel_h = h / img_height
                
                # Create FiftyOne Detection object
                class_name = COCO_CLASS_NAMES.get(pred['category_id'], f"category_{pred['category_id']}")
                detection = fo.Detection(
                    label=class_name,
                    bounding_box=[rel_x, rel_y, rel_w, rel_h],
                    confidence=pred["score"]
                )
                detections.append(detection)
        
        # Add predictions to sample
        sample["gemini_predictions"] = fo.Detections(detections=detections)
        sample.save()
    
    print(f"✅ Added predictions to dataset with field 'gemini_predictions'")

    # --- STEP 6: (Optional) Visualize in FiftyOne ---
    if args.ui:
        print("\n🔍 Launching FiftyOne visualization...")
        print("💡 The FiftyOne app will open in your browser")
        print("💡 You can see both 'ground_truth' and 'gemini_predictions' fields")
        print("💡 Use the sidebar to toggle between different label fields")
        print("💡 Press Ctrl+C to stop the server when done")
        
        session = fo.launch_app(dataset)
        session.wait()
    else:
        print("\n✅ Evaluation completed successfully!")
        print(f"📁 All results saved to: {run_dir}")
        print("💡 Use --ui flag to launch FiftyOne visualization")
        print(f"💡 Use: python view_run.py {run_dir} to visualize this run later")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COCO evaluation script with FiftyOne and Gemini predictions"
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        default=MAX_IMAGES,
        help=f"Maximum number of images to evaluate (default: {MAX_IMAGES})"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash). Options: gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-8b"
    )
    
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=1024,
        help="Thinking budget for Gemini models (default: 1024). Set to 0 to disable thinking."
    )
    
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch FiftyOne visualization interface"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (can also set GEMINI_API_KEY env var)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of parallel API calls (default: 10)"
    )
    
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable image preprocessing (resize/compress like HTML version)"
    )
    
    parser.add_argument(
        "--structured-output",
        action="store_true",
        help="Use Gemini's structured output with COCO class enums for better reliability"
    )

    parser.add_argument(
        "--code-execution",
        action="store_true",
        help="Enable code execution tools for iterative image analysis"
    )

    args = parser.parse_args()
    # Convert --no-preprocess to preprocess_images boolean
    args.preprocess_images = not args.no_preprocess
    main(args) 