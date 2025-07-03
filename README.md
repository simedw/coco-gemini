# COCO Gemini Evaluation

How good is gemini actually on object detection? 
COCO 2007 Val is a classic dataset of object detection, it's a bit dated, the annotations aren't perfect, but it should give us a good base line.


## Installation
Clone the repo:
```bash
git clone https://github.com/simedw/coco-gemini && cd coco-gemini
```

Install dependencies with uv:
```bash
uv sync
```
Ensure you’re using Python ≥ 3.9


## Configuration 

1. Get a Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set your API key as an environment variable:
  ```bash
  export GEMINI_API_KEY="your-api-key-here"
  ```
  Or create a `.env` file:
  ```
  GEMINI_API_KEY=your-api-key-here
  ```

## Running 

### Basic Usage
Evaluate Gemini on 100 random images from COCO:
```bash
uv run python coco_eval_script.py --max-images 100 --max-workers 10 --model gemini-2.5-pro --ui
```

### Available Parameters

| Parameter            | Description                                          | Default              |
| -------------------- | ---------------------------------------------------- | -------------------- |
| `--max-images`       | Number of images to evaluate (max 5000 available)    | `50`                |
| `--model`            | Gemini model to use                                  | `gemini-2.5-flash`   |
| `--thinking-budget`  | Thinking budget for Gemini models (0 to disable)    | `1024`              |
| `--max-workers`      | Concurrent Gemini API requests                       | `10`                |
| `--structured-output`| Use structured output for better reliability         | `False`             |
| `--ui`               | Launch FiftyOne's web UI to visually compare results | `False`             |

### Model Options
- `gemini-2.5-pro` - Most capable, slower
- `gemini-2.5-flash` - Good balance of speed and capability  
- `gemini-2.5-flash-8b` - Fastest, lower capability

### Examples
```bash
# Evaluate with different models
uv run python coco_eval_script.py --model gemini-2.5-pro --max-images 50
uv run python coco_eval_script.py --model gemini-2.5-flash --thinking-budget 0 --max-images 100

# Use structured output for better reliability
uv run python coco_eval_script.py --structured-output --max-images 50

# Run with visualization
uv run python coco_eval_script.py --max-images 100 --ui
```

## Runs Management

Each evaluation creates a timestamped run directory under `runs/` containing:
- `config.json` - Run configuration and parameters
- `ground_truth.json` - COCO ground truth annotations
- `predictions.json` - Model predictions
- `results.json` - Evaluation metrics and performance statistics

### Viewing Past Runs

```bash
# List all available runs
python view_run.py --list

# Visualize the most recent run
python view_run.py --latest

# Visualize a specific run
python view_run.py runs/gemini-2.5-pro_20241230_143022
```

No more manual file cleanup - each run is self-contained!

## Matrix Evaluation

Run systematic comparisons across multiple models and configurations:

```bash
# Run full matrix (3 models × 2 modes = 6 evaluations)
python run_matrix.py --max-images 100

# Run specific models or modes
python run_matrix.py --models flash pro --modes structured
python run_matrix.py --models flash-8b --max-images 50

# Show what would be executed without running
python run_matrix.py --dry-run

# View summary of all matrix runs
python run_matrix.py --summary
```

The matrix evaluation runs all combinations of:
- **Models**: 
  - `gemini-2.5-flash` (thinking budgets: 0, 1024)
  - `gemini-2.5-pro` (thinking budget: 1024 only)
  - `gemini-2.5-flash-lite-preview-06-17` (thinking budgets: 0, 1024)
- **Modes**: `structured`, `unstructured`

Total combinations: **10 runs** (flash: 4, pro: 2, lite: 4)

Results are automatically collected and displayed in a comparison table with thinking budget details.

## Results 

5000 (full validation set)

## 📊 Matrix Evaluation Summary (5000 images)

| Model      | Think | Mode         | mAP   | AP@0.5 | Success    | Avg Time |
|------------|-------|--------------|-------|--------|------------|----------|
| flash      | 0     | structured   | 0.224 | 0.381  | 4953/5000  | 0.18s    |
| flash      | 0     | unstructured | 0.261 | 0.417  | 4943/5000  | 0.20s    |
| flash      | 1024  | structured   | 0.160 | 0.311  | 4977/5000  | 0.27s    |
| flash      | 1024  | unstructured | 0.161 | 0.319  | 4981/5000  | 0.28s    |
| pro        | 1024  | structured   | **0.340** | 0.517  | 4994/5000  | 0.46s    |
| pro        | 1024  | unstructured | 0.288 | 0.438  | 4975/5000  | 0.47s    |
| flash-lite | 0     | structured   | 0.156 | 0.279  | 4665/5000  | 0.37s    |
| flash-lite | 0     | unstructured | 0.211 | 0.338  | 4784/5000  | 0.23s    |
| flash-lite | 1024  | structured   | 0.140 | 0.273  | 4832/5000  | 0.27s    |
| flash-lite | 1024  | unstructured | 0.215 | 0.364  | 4886/5000  | 0.24s    |


### Gemini 2.5 Flash

### 1000 unstructured output 
````
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.300
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.466
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.303
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.248
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.485
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.280
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.393
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.400
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.112
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.352
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.586
```

### 1000 (structured output)

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.254
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.409
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.263
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.082
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.225
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.422
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.248
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.345
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.351
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.105
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.315
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.529
```

## Gemini 2.5 Flash Lite (gemini-2.5-flash-lite-preview-06-17)

### 1000 images, unstructured
It got stuck in several infinite json loops, ones it found 10000 boats 
```
  {"label": "boat", "confidence": 0.322, "box_2d": [972, 976, 998, 998]},
  {"label": "boat", "confidence": 0.322, "box_2d": [971, 979, 998, 998]},
  {"label": "boat", "confidence": 0.322, "box_2d": [972, 977, 998, 998]},
  {"label": "boat", "confidence": 0.322, "box_2d": [971, 979, 998, 998]},
  {"label": "boat", "confidence": 0.322, "box_2d": [971, 979, 998, 998]},
  {"label": "boat", "confidence": 0.322, "box_2d": [972, 977, 998, 998]},
  ...
```

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.217
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.338
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.224
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.053
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.177
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.355
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.222
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.296
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.300
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.263
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.446
```


### 1000 images, structured 
Generates a bunch of invalid outputs like `[65, 1000, 77, 998]`, essentially ymin > ymax

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.185
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.314
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.178
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.018
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.131
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.347
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.195
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.271
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.278
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.037
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.223
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.448
```

## Gemini 2.5 Pro

### 1000 unstructured
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.319
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.474
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.335
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.087
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.305
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.514
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.318
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.440
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.450
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.129
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.435
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.652
```

### 1000 structured
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.365
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.544
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.387
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.098
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.351
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.448
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.129
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.647
```


## Upcoming Tests

- Evaluation with Segmentation Masks