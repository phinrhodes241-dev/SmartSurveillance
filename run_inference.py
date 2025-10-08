import argparse
import cv2
import torch
import os
import sys

def detect_yolo_version():
    """Detect whether YOLOv5 or YOLOv8 is installed."""
    try:
        import yolov5
        return "v5"
    except ImportError:
        try:
            from ultralytics import YOLO
            return "v8"
        except ImportError:
            return None

def run_inference(source):
    version = detect_yolo_version()
    if version is None:
        print("❌ Neither YOLOv5 nor YOLOv8 found. Please install one.")
        sys.exit(1)

    print(f"✅ Using YOLO{version.upper()}")

    if version == "v5":
        import yolov5
        model = yolov5.load("yolov5s.pt")  # You can replace with your custom weights
        results = model(source)
        results.show()

    elif version == "v8":
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")  # Replace with your custom weights
        results = model(source, show=True)
        print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartSurveillance YOLOv5/v8 Inference")
    parser.add_argument("--source", type=str, default="0", help="Path to image/video or webcam index")
    args = parser.parse_args()

    # Convert webcam string '0' to int
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    run_inference(source)
