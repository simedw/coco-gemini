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
Evaluate Gemini on 100 random images from COCO:
```
uv run python coco_eval_script.py --max-images 100 --max-workers 10 --structured-output --ui
```

| Parameter       | Description                                          | Default     |
| --------------- | ---------------------------------------------------- | ----------- |
| `--max-images`  | Number of images to evaluate (max 5000 available)    | `50`       |
| `--max-workers` | Concurrent Gemini API requests                       | `10`        |
| `--ui`          | Launch FiftyOne's web UI to visually compare results | Not Enabled |


## Results 

1000 random samples

| Model                     | Configuration       | mAP (IoU=0.50:0.95) |
| ------------------------- | ------------------- | ------------------- |
| **Gemini 2.5 Pro**        | Unstructured output | 0.319               |
| **Gemini 2.5 Pro**        | Structured output   | **0.365**           |
| **Gemini 2.5 Flash**      | Unstructured output | 0.300               |
| **Gemini 2.5 Flash**      | Structured output   | 0.254               |
| **Gemini 2.5 Flash Lite** | Unstructured output | 0.217               |
| **Gemini 2.5 Flash Lite** | Structured output   | 0.185               |



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