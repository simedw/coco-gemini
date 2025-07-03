#!/usr/bin/env python3
"""
Matrix Evaluation Script - Run evaluations across multiple models and structured/non-structured modes
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import json

# Configuration matrix
MODELS_CONFIG = {
    "gemini-2.5-flash": {
        "name": "gemini-2.5-flash",
        "thinking_budgets": [0, 1024]
    },
    "gemini-2.5-pro": {
        "name": "gemini-2.5-pro", 
        "thinking_budgets": [1024]  # 0 will cause an error for pro
    },
    "gemini-2.5-flash-lite-preview-06-17": {
        "name": "gemini-2.5-flash-lite-preview-06-17",
        "thinking_budgets": [0, 1024]
    }
}

STRUCTURED_OPTIONS = [
    {"flag": "", "name": "unstructured"},
    {"flag": "--structured-output", "name": "structured"}
]

DEFAULT_PARAMS = {
    "max_images": 5000,
    "max_workers": 30
}

def run_evaluation(model, thinking_budget, structured_config, params):
    """Run a single evaluation with the given configuration."""
    print(f"\n{'='*60}")
    print(f"🚀 Starting evaluation: {model} (thinking: {thinking_budget}, {structured_config['name']})")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        "python", "coco_eval_script.py",
        "--model", model,
        "--max-images", str(params["max_images"]),
        "--max-workers", str(params["max_workers"]),
        "--thinking-budget", str(thinking_budget)
    ]
    
    if structured_config["flag"]:
        cmd.append(structured_config["flag"])
    
    print(f"📋 Command: {' '.join(cmd)}")
    
    # Run the evaluation
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Completed in {duration:.2f} seconds")
        return {
            "status": "success",
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ Failed after {duration:.2f} seconds")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        
        return {
            "status": "failed",
            "duration": duration,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        return {
            "status": "interrupted",
            "duration": time.time() - start_time
        }

def collect_results():
    """Collect results from all runs in the runs directory."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return []
    
    results = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        config_path = run_dir / "config.json"
        results_path = run_dir / "results.json"
        
        if config_path.exists() and results_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            with open(results_path, 'r') as f:
                result = json.load(f)
            
            results.append({
                "run_name": run_dir.name,
                "model": config.get("model_name", "unknown"),
                "structured": config.get("structured_output", False),
                "max_images": config.get("max_images", 0),
                "thinking_budget": config.get("thinking_budget", 0),
                "timestamp": config.get("timestamp", ""),
                "ap_50_95": result.get("metrics", {}).get("ap_50_95", None),
                "ap_50": result.get("metrics", {}).get("ap_50", None),
                "successful_images": result.get("statistics", {}).get("successful_images", 0),
                "failed_images": result.get("statistics", {}).get("failed_images", 0),
                "total_time": result.get("statistics", {}).get("total_time_seconds", 0),
                "avg_time_per_image": result.get("statistics", {}).get("average_time_per_image", 0)
            })
    
    return sorted(results, key=lambda x: x["timestamp"], reverse=True)

def print_summary_table(results):
    """Print a summary table of all results."""
    if not results:
        print("No results found.")
        return
    
    print("\n" + "="*95)
    print("📊 MATRIX EVALUATION SUMMARY")
    print("="*95)
    
    # Group by model, thinking budget, and structured mode
    matrix_results = {}
    for result in results:
        model = result["model"]
        thinking = result["thinking_budget"]
        structured = "structured" if result["structured"] else "unstructured"
        key = f"{model}_{thinking}_{structured}"
        
        if key not in matrix_results:
            matrix_results[key] = result
        else:
            # Keep the most recent one
            if result["timestamp"] > matrix_results[key]["timestamp"]:
                matrix_results[key] = result
    
    # Print table header
    print(f"{'Model':<25} {'Think':<6} {'Mode':<12} {'mAP':<8} {'AP@0.5':<8} {'Success':<10} {'Avg Time':<10}")
    print("-" * 95)
    
    # Print results
    for model_key, model_config in MODELS_CONFIG.items():
        model = model_config["name"]
        for thinking_budget in model_config["thinking_budgets"]:
            for mode in ["structured", "unstructured"]:
                key = f"{model}_{thinking_budget}_{mode}"
                if key in matrix_results:
                    r = matrix_results[key]
                    mAP = f"{r['ap_50_95']:.3f}" if r['ap_50_95'] is not None else "N/A"
                    ap50 = f"{r['ap_50']:.3f}" if r['ap_50'] is not None else "N/A"
                    success_rate = f"{r['successful_images']}/{r['successful_images'] + r['failed_images']}"
                    avg_time = f"{r['avg_time_per_image']:.2f}s" if r['avg_time_per_image'] > 0 else "N/A"
                    
                    # Shorten model name for display
                    display_model = model.replace("gemini-2.5-", "").replace("-preview-06-17", "")
                    print(f"{display_model:<25} {thinking_budget:<6} {mode:<12} {mAP:<8} {ap50:<8} {success_rate:<10} {avg_time:<10}")
                else:
                    display_model = model.replace("gemini-2.5-", "").replace("-preview-06-17", "")
                    print(f"{display_model:<25} {thinking_budget:<6} {mode:<12} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A':<10}")
    
    print("="*95)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run matrix evaluation across multiple models and configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_matrix.py                    # Run full matrix with default settings
  python run_matrix.py --max-images 50   # Run with 50 images per evaluation
  python run_matrix.py --summary         # Show summary of previous runs
  python run_matrix.py --models flash pro # Run only specific models
        """
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_PARAMS["max_images"],
        help=f"Number of images per evaluation (default: {DEFAULT_PARAMS['max_images']})"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_PARAMS["max_workers"],
        help=f"Number of parallel workers (default: {DEFAULT_PARAMS['max_workers']})"
    )
    

    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["flash", "pro", "lite"],
        help="Specific models to run (default: all models)"
    )
    
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["structured", "unstructured"],
        help="Specific modes to run (default: both modes)"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary of previous runs and exit"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without actually running"
    )
    
    args = parser.parse_args()
    
    # Show summary and exit if requested
    if args.summary:
        results = collect_results()
        print_summary_table(results)
        return
    
    # Build model list
    model_map = {
        "flash": "gemini-2.5-flash",
        "pro": "gemini-2.5-pro",
        "lite": "gemini-2.5-flash-lite-preview-06-17"
    }
    
    if args.models:
        selected_models = {model_map[m]: MODELS_CONFIG[model_map[m]] for m in args.models}
    else:
        selected_models = MODELS_CONFIG
    
    # Build mode list
    if args.modes:
        modes = [{"flag": "--structured-output" if m == "structured" else "", "name": m} 
                for m in args.modes]
    else:
        modes = STRUCTURED_OPTIONS
    
    # Build parameters
    params = {
        "max_images": args.max_images,
        "max_workers": args.max_workers
    }
    
    # Calculate total runs
    total_runs = sum(len(config["thinking_budgets"]) for config in selected_models.values()) * len(modes)
    
    # Print configuration
    print("🚀 MATRIX EVALUATION CONFIGURATION")
    print("="*50)
    print(f"📊 Models: {', '.join(selected_models.keys())}")
    print(f"📋 Modes: {', '.join([m['name'] for m in modes])}")
    print(f"📷 Images per run: {params['max_images']}")
    print(f"🔧 Workers: {params['max_workers']}")
    print(f"🧠 Thinking budgets: per model (flash/lite: 0,1024; pro: 1024)")
    print(f"🎯 Total runs: {total_runs}")
    
    if args.dry_run:
        print("\n📋 DRY RUN - Commands that would be executed:")
        for model_name, model_config in selected_models.items():
            for thinking_budget in model_config["thinking_budgets"]:
                for mode in modes:
                    cmd = [
                        "python", "coco_eval_script.py",
                        "--model", model_name,
                        "--max-images", str(params["max_images"]),
                        "--max-workers", str(params["max_workers"]),
                        "--thinking-budget", str(thinking_budget)
                    ]
                    if mode["flag"]:
                        cmd.append(mode["flag"])
                    print(f"  {' '.join(cmd)}")
        return
    
    # Ask for confirmation
    print(f"\n⚠️  This will run {total_runs} evaluations.")
    print(f"⏱️  Estimated time: {total_runs * params['max_images'] * 0.33 / 60:.1f} minutes")
    response = input("Continue? (y/N): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("❌ Cancelled by user")
        return
    
    # Run the matrix
    print(f"\n🎯 Starting matrix evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    run_results = []
    current_run = 0
    
    for model_name, model_config in selected_models.items():
        for thinking_budget in model_config["thinking_budgets"]:
            for mode in modes:
                current_run += 1
                print(f"\n📊 Progress: {current_run}/{total_runs}")
                
                result = run_evaluation(model_name, thinking_budget, mode, params)
                run_results.append({
                    "model": model_name,
                    "thinking_budget": thinking_budget,
                    "mode": mode["name"],
                    "result": result
                })
                
                if result["status"] == "interrupted":
                    print("⚠️  Matrix evaluation interrupted by user")
                    break
            
            if run_results and run_results[-1]["result"]["status"] == "interrupted":
                break
        
        if run_results and run_results[-1]["result"]["status"] == "interrupted":
            break
    
    # Print final summary
    print(f"\n🏁 Matrix evaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    successful_runs = sum(1 for r in run_results if r["result"]["status"] == "success")
    failed_runs = sum(1 for r in run_results if r["result"]["status"] == "failed")
    interrupted_runs = sum(1 for r in run_results if r["result"]["status"] == "interrupted")
    
    print(f"✅ Successful runs: {successful_runs}")
    print(f"❌ Failed runs: {failed_runs}")
    print(f"⚠️  Interrupted runs: {interrupted_runs}")
    
    if successful_runs > 0:
        print("\n📊 Collecting and displaying results...")
        results = collect_results()
        print_summary_table(results)
        
        print(f"\n💡 View individual runs with:")
        print(f"   python view_run.py --list")
        print(f"   python view_run.py --latest")

if __name__ == "__main__":
    main() 