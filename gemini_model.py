#!/usr/bin/env python3
"""
Gemini Vision API implementation for COCO object detection.

This module provides a GeminiDetector class that uses Google's Gemini Vision API
to perform object detection on images and returns results in COCO format.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import enum

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Google Gemini API not available. Install with: uv add google-genai")

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


class CocoClass(enum.Enum):
    """COCO dataset class names as enum for structured output."""
    PERSON = "person"
    BICYCLE = "bicycle"
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    AIRPLANE = "airplane"
    BUS = "bus"
    TRAIN = "train"
    TRUCK = "truck"
    BOAT = "boat"
    TRAFFIC_LIGHT = "traffic light"
    FIRE_HYDRANT = "fire hydrant"
    STOP_SIGN = "stop sign"
    PARKING_METER = "parking meter"
    BENCH = "bench"
    BIRD = "bird"
    CAT = "cat"
    DOG = "dog"
    HORSE = "horse"
    SHEEP = "sheep"
    COW = "cow"
    ELEPHANT = "elephant"
    BEAR = "bear"
    ZEBRA = "zebra"
    GIRAFFE = "giraffe"
    BACKPACK = "backpack"
    UMBRELLA = "umbrella"
    HANDBAG = "handbag"
    TIE = "tie"
    SUITCASE = "suitcase"
    FRISBEE = "frisbee"
    SKIS = "skis"
    SNOWBOARD = "snowboard"
    SPORTS_BALL = "sports ball"
    KITE = "kite"
    BASEBALL_BAT = "baseball bat"
    BASEBALL_GLOVE = "baseball glove"
    SKATEBOARD = "skateboard"
    SURFBOARD = "surfboard"
    TENNIS_RACKET = "tennis racket"
    BOTTLE = "bottle"
    WINE_GLASS = "wine glass"
    CUP = "cup"
    FORK = "fork"
    KNIFE = "knife"
    SPOON = "spoon"
    BOWL = "bowl"
    BANANA = "banana"
    APPLE = "apple"
    SANDWICH = "sandwich"
    ORANGE = "orange"
    BROCCOLI = "broccoli"
    CARROT = "carrot"
    HOT_DOG = "hot dog"
    PIZZA = "pizza"
    DONUT = "donut"
    CAKE = "cake"
    CHAIR = "chair"
    COUCH = "couch"
    POTTED_PLANT = "potted plant"
    BED = "bed"
    DINING_TABLE = "dining table"
    TOILET = "toilet"
    TV = "tv"
    LAPTOP = "laptop"
    MOUSE = "mouse"
    REMOTE = "remote"
    KEYBOARD = "keyboard"
    CELL_PHONE = "cell phone"
    MICROWAVE = "microwave"
    OVEN = "oven"
    TOASTER = "toaster"
    SINK = "sink"
    REFRIGERATOR = "refrigerator"
    BOOK = "book"
    CLOCK = "clock"
    VASE = "vase"
    SCISSORS = "scissors"
    TEDDY_BEAR = "teddy bear"
    HAIR_DRIER = "hair drier"
    TOOTHBRUSH = "toothbrush"


class Detection(BaseModel):
    """Single object detection with COCO class enum."""
    label: CocoClass = Field(description="Object class from COCO dataset")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    box_2d: List[int] = Field(
        description="Bounding box [ymin, xmin, ymax, xmax] normalized 0-1000",
        min_items=4,
        max_items=4
    )


class DetectionResponse(BaseModel):
    """Response format for object detection."""
    detections: List[Detection] = Field(description="List of detected objects")


class GeminiDetector:
    """
    Object detector using Google's Gemini Vision API with parallel processing support.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-pro", thinking_budget: int = 1024, max_workers: int = 10, preprocess_images: bool = True, use_structured_output: bool = False):
        """
        Initialize the Gemini detector.
        
        Args:
            api_key: Gemini API key. If None, will try to get from GEMINI_API_KEY env var.
            model_name: Gemini model to use for detection.
            thinking_budget: Thinking budget for Gemini models (default: 1024). Set to 0 to disable thinking.
            max_workers: Maximum number of parallel API calls.
            preprocess_images: Whether to resize/compress images like HTML version (default: True).
            use_structured_output: Whether to use Gemini's structured output with COCO class enums (default: False).
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Gemini API not available. Install with: uv add google-genai")
        
        if use_structured_output and not PYDANTIC_AVAILABLE:
            raise ImportError("Pydantic required for structured output. Install with: uv add pydantic")
        
        self.model_name = model_name
        self.thinking_budget = thinking_budget
        self.max_workers = max_workers
        self.preprocess_images = preprocess_images
        self.use_structured_output = use_structured_output
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize the client
        self.client = genai.Client(api_key=self.api_key)
        
        # Load COCO class names
        self.coco_classes = self._load_coco_classes()
        self.class_name_to_id = {name: int(id_str) for id_str, name in self.coco_classes.items()}
        
        # Create the prompt with COCO class names
        self.prompt = self._create_prompt()
        
        # Configure response format
        if self.use_structured_output:
            self.config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
                response_mime_type="application/json",
                response_schema=DetectionResponse
            )
        else:
            self.config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
                response_mime_type="application/json"
            )

        # Thread lock for thread-safe printing
        self._print_lock = threading.Lock()
    
    def _load_coco_classes(self) -> Dict[str, str]:
        """Load COCO class names from JSON file."""
        try:
            with open("coco_classes.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "coco_classes.json not found. Make sure it's in the current directory."
            )
    
    def _create_prompt(self) -> str:
        """Create the detection prompt with COCO class names."""
        class_list = ", ".join(self.coco_classes.values())
        
        prompt = f"""Look carefully at this image and detect ALL visible objects, including small ones.

IMPORTANT: Focus on finding as many objects as possible, even if they are small, distant, or partially visible.
Make sure that the bounding box is as tight as possible.
Valid object classes: {class_list}

For each detected object, provide:
- "label": exact class name from the list above
- "confidence": how certain you are (0.0 to 1.0)  
- "box_2d": bounding box [ymin, xmin, ymax, xmax] normalized 0-1000

Detect everything you can see that matches the valid classes. Don't be conservative - include objects even if you're only moderately confident.

Return as JSON array:
[
  {{
    "label": "person",
    "confidence": 0.95,
    "box_2d": [100, 200, 300, 400]
  }},
  {{
    "label": "kite", 
    "confidence": 0.80,
    "box_2d": [50, 150, 250, 350]
  }}
]"""
        
        return prompt
    
    def _preprocess_image(self, image_path: str, max_width: int = 1000, quality: int = 70) -> Image.Image:
        """
        Preprocess image like the HTML version: resize and compress.
        
        Args:
            image_path: Path to the input image
            max_width: Maximum width in pixels (default: 1000)
            quality: JPEG quality 0-100 (default: 70)
            
        Returns:
            Preprocessed PIL Image
        """
        with Image.open(image_path) as img:
            # Convert to RGB if needed (for JPEG compatibility)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if width > max_width (maintain aspect ratio)
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Compress by saving to bytes and reloading (like HTML canvas.toBlob)
            from io import BytesIO
            compressed_bytes = BytesIO()
            img.save(compressed_bytes, format='JPEG', quality=quality, optimize=True)
            compressed_bytes.seek(0)
            
            # Return the compressed image
            return Image.open(compressed_bytes)
    
    def detect(self, image_path: str, image_id: int) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Detect objects in an image using Gemini Vision API.
        
        Args:
            image_path: Path to the image file.
            image_id: Image ID for COCO format.
            
        Returns:
            Tuple of (List of detection dictionaries in COCO format, success flag).
        """
        try:
            # Load and optionally preprocess image (like HTML version)
            if self.preprocess_images:
                image = self._preprocess_image(image_path)
            else:
                image = Image.open(image_path)
            width, height = image.size
            
            # Call Gemini API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[image, self.prompt],
                config=self.config
            )
            
            # Parse response
            if self.use_structured_output:
                try:
                    # Use parsed response for structured output
                    detection_response = response.parsed
                    if detection_response is None or not hasattr(detection_response, 'detections'):
                        with self._print_lock:
                            print(f"⚠️  Image {image_id}: No structured response received")
                        return [], False
                    detections = [
                        {
                            "label": det.label.value,  # Get string value from enum
                            "confidence": det.confidence,
                            "box_2d": det.box_2d
                        }
                        for det in detection_response.detections
                    ]
                except Exception as e:
                    with self._print_lock:
                        print(f"⚠️  Image {image_id}: Failed to parse structured response: {e}")
                        print(f"Raw response: {response.text}")
                    return [], False
            else:
                try:
                    detections = json.loads(response.text)
                except json.JSONDecodeError as e:
                    with self._print_lock:
                        print(f"⚠️  Image {image_id}: Failed to parse Gemini response: {e}")
                        print(f"Raw response: {response.text}")
                    return [], False
            
            # Convert to COCO format
            coco_detections = []
            conversion_errors = 0
            for detection in detections:
                try:
                    coco_detection = self._convert_to_coco_format(
                        detection, image_id, width, height
                    )
                    if coco_detection:
                        coco_detections.append(coco_detection)
                except Exception as e:
                    conversion_errors += 1
                    with self._print_lock:
                        print(f"⚠️  Image {image_id}: Error converting detection: {e}")
                        print(f"Detection data: {detection}")
                    continue
            
            if conversion_errors > 0:
                with self._print_lock:
                    print(f"⚠️  Image {image_id}: {conversion_errors} detection(s) failed to convert")
            
            return coco_detections, True
            
        except Exception as e:
            with self._print_lock:
                print(f"⚠️  Image {image_id}: Error detecting objects in {image_path}: {e}")
            return [], False
    
    def detect_parallel(self, image_data: List[Tuple[str, int]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Detect objects in multiple images in parallel.
        
        Args:
            image_data: List of (image_path, image_id) tuples.
            
        Returns:
            Tuple of (all predictions list, statistics dict).
        """
        all_predictions = []
        failed_images = []
        successful_images = []
        
        print(f"🚀 Processing {len(image_data)} images with {self.max_workers} parallel workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(self.detect, image_path, image_id): (image_path, image_id)
                for image_path, image_id in image_data
            }
            
            # Process completed tasks
            for future in as_completed(future_to_image):
                image_path, image_id = future_to_image[future]
                try:
                    predictions, success = future.result()
                    if success:
                        all_predictions.extend(predictions)
                        successful_images.append(image_id)
                    else:
                        failed_images.append(image_id)
                except Exception as e:
                    with self._print_lock:
                        print(f"⚠️  Image {image_id}: Unexpected error: {e}")
                    failed_images.append(image_id)
        
        # Statistics
        stats = {
            "total_images": len(image_data),
            "successful_images": len(successful_images),
            "failed_images": len(failed_images),
            "total_predictions": len(all_predictions),
            "failed_image_ids": failed_images
        }
        
        return all_predictions, stats
    
    def _convert_to_coco_format(
        self, 
        detection: Dict[str, Any], 
        image_id: int, 
        img_width: int, 
        img_height: int
    ) -> Optional[Dict[str, Any]]:
        """
        Convert Gemini detection to COCO format.
        
        Args:
            detection: Gemini detection dict with label, confidence, box_2d.
            image_id: Image ID for COCO format.
            img_width: Image width in pixels.
            img_height: Image height in pixels.
            
        Returns:
            COCO format detection dict or None if invalid.
        """
        try:
            # Validate required fields
            if "label" not in detection:
                raise ValueError("Missing 'label' field")
            if "box_2d" not in detection:
                raise ValueError("Missing 'box_2d' field")
            
            label = detection["label"].lower()
            # gemini pro refuses to return confidence levels
            confidence = detection.get("confidence", 0.5)
            box_2d = detection["box_2d"]
            
            # Validate box_2d format
            if not isinstance(box_2d, list):
                raise ValueError(f"box_2d should be a list, got {type(box_2d)}")
            if len(box_2d) != 4:
                raise ValueError(f"box_2d should have 4 coordinates [ymin,xmin,ymax,xmax], got {len(box_2d)} values: {box_2d}")
            
            # Validate class name
            if label not in self.class_name_to_id:
                raise ValueError(f"Unknown class: '{label}'. Available classes: {list(self.class_name_to_id.keys())[:10]}...")
            
            category_id = self.class_name_to_id[label]
            
            # Convert coordinates from [ymin, xmin, ymax, xmax] normalized 0-1000
            # to [x, y, width, height] in pixels (COCO format)
            try:
                ymin_norm, xmin_norm, ymax_norm, xmax_norm = box_2d
            except ValueError as e:
                raise ValueError(f"Failed to unpack box_2d coordinates: {box_2d}. Error: {e}")
            
            # Validate coordinate ranges
            for i, coord in enumerate(box_2d):
                if not isinstance(coord, (int, float)):
                    raise ValueError(f"box_2d[{i}] should be a number, got {type(coord)}: {coord}")
                if coord < 0 or coord > 1000:
                    raise ValueError(f"box_2d[{i}] should be 0-1000, got {coord}")
            
            # Convert to absolute coordinates
            x1 = int(xmin_norm / 1000 * img_width)
            y1 = int(ymin_norm / 1000 * img_height)
            x2 = int(xmax_norm / 1000 * img_width)
            y2 = int(ymax_norm / 1000 * img_height)
            
            # Convert to COCO format [x, y, width, height]
            x = x1
            y = y1
            width = x2 - x1
            height = y2 - y1
            
            # Validate bounding box
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid bounding box dimensions: width={width}, height={height} (from coords: {box_2d})")
            
            # Clamp to image boundaries
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            width = min(width, img_width - x)
            height = min(height, img_height - y)
            
            return {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, width, height],
                "score": float(confidence)
            }
            
        except Exception as e:
            # Re-raise with more context - the calling function will handle printing
            raise ValueError(f"Conversion failed: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        return {
            "model_name": self.model_name,
            "thinking_budget": self.thinking_budget,
            "api_provider": "Google Gemini",
            "coco_classes_count": len(self.coco_classes),
            "max_workers": self.max_workers,
            "preprocess_images": self.preprocess_images,
            "structured_output": self.use_structured_output
        }


 