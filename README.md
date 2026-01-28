# COCO Gemini Evaluation

How good is Gemini at object detection? We benchmark Google's Gemini models on COCO 2017 validation set to measure their zero-shot object detection capabilities.

**Latest:** Gemini 3 Flash with Agentic Vision code execution achieves **0.451 mAP**, outperforming all previous models including the larger 3-pro-preview. Code execution provides a validated **10-16% quality improvement** over standard inference.

COCO 2017 Val is a classic dataset for object detection benchmarking. While the annotations aren't perfect, it provides a good baseline for comparing model capabilities.


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
| `--code-execution`   | Enable code execution for image analysis             | `False`             |
| `--ui`               | Launch FiftyOne's web UI to visually compare results | `False`             |

### Model Options
- `gemini-2.5-pro` - Most capable, slower
- `gemini-2.5-flash` - Good balance of speed and capability
- `gemini-2.5-flash-8b` - Fastest, lower capability
- `gemini-3-flash-preview` - Latest with Agentic Vision code execution

### Examples
```bash
# Evaluate with different models
uv run python coco_eval_script.py --model gemini-2.5-pro --max-images 50
uv run python coco_eval_script.py --model gemini-2.5-flash --thinking-budget 0 --max-images 100

# Use structured output for better reliability
uv run python coco_eval_script.py --structured-output --max-images 50

# Test with Agentic Vision code execution
uv run python coco_eval_script.py --model gemini-3-flash-preview --code-execution --max-images 100

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
  - `gemini-3-flash-preview` (thinking budgets: 0, 1024)
- **Modes**: `structured`, `unstructured`
- **Code Execution**: `enabled`, `disabled`

Total combinations: **28 runs** (flash: 8, pro: 4, lite: 8, flash-3: 8)

You can filter runs using:
```bash
# Test only with code execution enabled
python run_matrix.py --models flash-3 --code-execution-modes enabled --max-images 100

# Compare with and without code execution
python run_matrix.py --models flash-3 --code-execution-modes both --max-images 50
```

Results are automatically collected and displayed in a comparison table with thinking budget and code execution details.

## Results

## 🚀 Gemini 3 Flash with Agentic Vision (1000 images)

The new `gemini-3-flash-preview` model supports code execution tools that enable iterative image analysis through a Think-Act-Observe loop. We evaluated the impact of this feature across multiple configurations.

### Complete Results Matrix

| Think | Mode | Code Exec | mAP | AP@0.5 | Success Rate | Avg Time |
|-------|------|-----------|-----|--------|--------------|----------|
| 0 | structured | no | 0.397 | 0.562 | 1000/1000 | 0.10s |
| 0 | structured | **yes** | **0.437** | **0.605** | 997/1000 | 0.41s |
| 0 | unstructured | no | 0.393 | 0.551 | 935/1000 | 0.12s |
| 0 | unstructured | **yes** | **0.449** | **0.623** | 989/1000 | 0.39s |
| 1024 | structured | no | 0.403 | 0.570 | 999/1000 | 0.10s |
| 1024 | structured | **yes** | **0.451** | **0.636** | 999/1000 | 0.44s |
| 1024 | unstructured | no | 0.390 | 0.544 | 935/1000 | 0.11s |
| 1024 | unstructured | **yes** | **0.451** | **0.629** | 991/1000 | 0.47s |

### Key Findings

**🎯 Code Execution Impact:**
- **+10-16% mAP improvement** across all configurations
- Biggest gains with unstructured output (+14-16%)
- Consistent improvement with both thinking budgets

**📊 Detailed Improvements:**
| Configuration | mAP Gain | AP@0.5 Gain |
|--------------|----------|-------------|
| Think=0, Structured | +10.1% | +7.7% |
| Think=0, Unstructured | +14.2% | +13.1% |
| Think=1024, Structured | +11.9% | +11.6% |
| Think=1024, Unstructured | **+15.6%** | **+15.6%** |

**⚡ Performance Trade-offs:**
- Without code execution: 0.10-0.12s per image
- With code execution: 0.39-0.47s per image (**~4x slower**)
- Cost: Higher token usage due to code execution iterations

**✅ Reliability:**
- Structured output: 997-1000/1000 success rate (99.7-100%)
- Unstructured output: 935-991/1000 success rate (93.5-99.1%)
- Code execution maintains high reliability

### Recommendations

**For Maximum Quality (mAP: 0.451):**
```bash
uv run python coco_eval_script.py \
  --model gemini-3-flash-preview \
  --thinking-budget 1024 \
  --code-execution \
  --max-images 1000
```

**For Best Speed/Quality Balance (mAP: 0.403, 0.10s/img):**
```bash
uv run python coco_eval_script.py \
  --model gemini-3-flash-preview \
  --thinking-budget 1024 \
  --structured-output \
  --max-images 1000
```

**Budget-Conscious Option (mAP: 0.437, 0.41s/img):**
```bash
uv run python coco_eval_script.py \
  --model gemini-3-flash-preview \
  --thinking-budget 0 \
  --code-execution \
  --structured-output \
  --max-images 1000
```

### Comparison with Gemini 2.5/3 Models

| Model | Best mAP | AP@0.5 | Images | Config | Notes |
|-------|----------|--------|--------|--------|-------|
| **3-flash-preview (code exec)** | **0.451** | **0.636** | 1000 | think=1024, code=yes | 🏆 Best quality |
| 3-pro-preview | 0.407 | 0.582 | 5000 | think=1024, structured | Slower, more expensive |
| 2.5-pro | 0.340 | 0.517 | 5000 | think=1024, structured | Legacy model |
| 2.5-flash | 0.261 | 0.417 | 5000 | think=0, unstructured | Legacy model |

**Key Insight:** The new Gemini 3 Flash with code execution **outperforms all previous models**, including the larger 3-pro-preview, while being faster and more cost-effective.

**Claim Validation:** Google's claimed 5-10% quality improvement from Agentic Vision code execution is **validated and exceeded** — we measured **10-16% improvement** in real-world COCO object detection tasks.

---

## 📊 Matrix Evaluation Summary (5000 images - Legacy Models)

| Model      | Think | Mode         | mAP   | AP@0.5 | Success    | Avg Time |
|------------|-------|--------------|-------|--------|------------|----------|
| 2.5-flash      | 0     | structured   | 0.224 | 0.381  | 4953/5000  | 0.18s    |
| 2.5-flash      | 0     | unstructured | 0.261 | 0.417  | 4943/5000  | 0.20s    |
| 2.5-flash      | 1024  | structured   | 0.160 | 0.311  | 4977/5000  | 0.27s    |
| 2.5-flash      | 1024  | unstructured | 0.161 | 0.319  | 4981/5000  | 0.28s    |
| 2.5-pro        | 1024  | structured   | **0.340** | 0.517  | 4994/5000  | 0.46s    |
| 2.5-pro        | 1024  | unstructured | 0.288 | 0.438  | 4975/5000  | 0.47s    |
| 2.5-flash-lite | 0     | structured   | 0.156 | 0.279  | 4665/5000  | 0.37s    |
| 2.5-flash-lite | 0     | unstructured | 0.211 | 0.338  | 4784/5000  | 0.23s    |
| 2.5-flash-lite | 1024  | structured   | 0.140 | 0.273  | 4832/5000  | 0.27s    |
| 2.5-flash-lite | 1024  | unstructured | 0.215 | 0.364  | 4886/5000  | 0.24s    |
| 3-pro-preview | 1024 | structured | **0.407** | **0.582** | 4991/5000 | 3.63s | 


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