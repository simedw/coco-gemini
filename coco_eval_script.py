#!/usr/bin/env python3
"""
COCO Evaluation Script with FiftyOne and Gemini Predictions
"""

import os
import json
import argparse
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.types as fot
from PIL import Image

# Import our modular components
from gemini_model import GeminiDetector

# --- CONFIGURATION ---
MAX_IMAGES = 50  # Keep it small for testing
GT_PATH_TEMPLATE = "coco_gt_{}.json"
PRED_PATH_TEMPLATE = "gemini_predictions_{}.json"


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
    
    # Use args for configuration
    max_images = args.max_images or MAX_IMAGES
    gt_path = args.gt_path or GT_PATH_TEMPLATE.format(max_images)
    pred_path = args.pred_path or PRED_PATH_TEMPLATE.format(max_images)
    
    print(f"📁 Using files: {gt_path} | {pred_path}")
    
    # Load COCO class names
    print("📋 Loading COCO class names...")
    COCO_CLASS_NAMES = load_coco_class_names()
    
    # Initialize Gemini detector
    print("🤖 Initializing Gemini detector...")
    try:
        detector = GeminiDetector(
            api_key=args.api_key, 
            max_workers=args.max_workers,
            preprocess_images=args.preprocess_images,
            use_structured_output=args.structured_output
        )
        print(f"✅ Gemini detector initialized: {detector.get_model_info()}")
        print(f"🔧 Parallel processing: {args.max_workers} workers")
        print(f"📐 Image preprocessing: {'enabled' if args.preprocess_images else 'disabled'}")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini detector: {e}")
        print("💡 Make sure you have set GEMINI_API_KEY environment variable or use --api-key")
        print("💡 Install Gemini dependencies with: uv sync --extra gemini")
        return
    
    # --- STEP 1: Load COCO Validation Set ---
    print(f"📥 Loading COCO validation dataset (max {max_images} images)...")
    dataset = foz.load_zoo_dataset(
        "coco-2017", split="validation", max_samples=max_images, shuffle=True
    )
    print(f"✅ Loaded {len(dataset)} images")

    # --- STEP 2: Export Ground Truth Annotations ---
    print("📝 Exporting ground truth annotations...")
    if not Path(gt_path).exists():
        # Custom export to maintain proper COCO category IDs
        export_ground_truth_with_correct_ids(dataset, gt_path, COCO_CLASS_NAMES)
        print(f"✅ Ground truth exported to {gt_path}")
    else:
        print(f"✅ Ground truth file {gt_path} already exists")

    # --- STEP 3: Run Gemini Object Detection ---
    print("🤖 Running Gemini object detection...")
    
    # Prepare image data for parallel processing
    image_data = [(sample.filepath, i + 1) for i, sample in enumerate(dataset)]
    
    # Run parallel detection
    all_preds, stats = detector.detect_parallel(image_data)
    
    # Print statistics
    print(f"\n📊 Detection Statistics:")
    print(f"  ✅ Successful images: {stats['successful_images']}/{stats['total_images']}")
    print(f"  ❌ Failed images: {stats['failed_images']}")
    print(f"  🎯 Total predictions: {stats['total_predictions']}")
    
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
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    
    print("\n" + "="*50)
    print("🏆 EVALUATION RESULTS")
    print("="*50)
    evaluator.summarize()
    print("="*50)

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
        print("💡 Use --ui flag to launch FiftyOne visualization")
        print("💡 Dataset now contains both 'ground_truth' and 'gemini_predictions' fields")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COCO evaluation script with FiftyOne and Gemini predictions"
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help=f"Maximum number of images to evaluate (default: {MAX_IMAGES})"
    )
    
    parser.add_argument(
        "--gt-path",
        type=str,
        default=None,
        help="Path to ground truth COCO JSON file (default: coco_gt_<max_images>.json)"
    )
    
    parser.add_argument(
        "--pred-path",
        type=str,
        default=None,
        help="Path to predictions JSON file (default: gemini_predictions_<max_images>.json)"
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
    
    args = parser.parse_args()
    # Convert --no-preprocess to preprocess_images boolean
    args.preprocess_images = not args.no_preprocess
    main(args) 