# Vehicle Detection System Documentation

## Overview
This project implements a real-time vehicle detection system using YOLOv3 (You Only Look Once version 3) deep learning model and OpenCV. The system can detect multiple types of vehicles including cars, trucks, buses, and motorcycles from video streams.

## Components

### 1. YOLOv3 Model
YOLOv3 is a state-of-the-art, real-time object detection system that:
- Uses a single neural network to process the entire image
- Divides the image into regions and predicts bounding boxes and probabilities
- Processes images at 30 FPS
- Has a backbone network of 53 layers (Darknet-53)

Required YOLOv3 files:
- `yolov3.weights`: Pre-trained model weights
- `yolov3.cfg`: Model configuration file
- `coco.names`: Class names from COCO dataset

### 2. OpenCV Integration
OpenCV (cv2) is used for:
- Video capture and frame processing
- Drawing detection boxes and labels
- Image preprocessing
- Neural network implementation via `cv2.dnn` module

Key OpenCV features utilized:
- `cv2.dnn.readNet()`: Loads the YOLO network
- `cv2.dnn.blobFromImage()`: Preprocesses images for the network
- `cv2.dnn.NMSBoxes()`: Performs non-maximum suppression
- `cv2.rectangle()` and `cv2.putText()`: Drawing detection visualizations

## Implementation Details

### Class Structure: VehicleDetector

1. Initialization (`__init__`):
```python
def __init__(self):
    self.net = cv2.dnn.readNet(weights_path, config_path)
    self.conf_threshold = 0.5
    self.nms_threshold = 0.4
```

2. File Verification (`check_files`):
- Verifies presence of required YOLO files
- Provides download links if files are missing

3. Vehicle Detection (`detect_vehicles`):
- Processes frames through YOLO network
- Filters detections for vehicle classes (IDs: 2, 3, 5, 7)
- Applies confidence thresholding
- Performs non-maximum suppression

4. Frame Processing (`process_frame`):
- Draws bounding boxes around detected vehicles
- Displays confidence scores and class labels
- Shows total vehicle count

### Integration Steps

1. Model Loading:
```python
self.net = cv2.dnn.readNet(weights_path, config_path)
self.classes = [line.strip() for line in f.readlines()]
```

2. Image Preprocessing:
```python
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), 
                            swapRB=True, crop=False)
```

3. Detection Pipeline:
```python
self.net.setInput(blob)
outputs = self.net.forward(self.output_layers)
```

4. Post-processing:
- Filtering detections based on confidence
- Applying non-maximum suppression
- Drawing results on frame

## Performance Considerations

1. Detection Parameters:
- Confidence threshold: 0.5
- NMS threshold: 0.4
- Input size: 416x416 pixels

2. Supported Vehicle Classes:
- Cars (ID: 2)
- Motorcycles (ID: 3)
- Buses (ID: 5)
- Trucks (ID: 7)

## References

1. YOLOv3 Paper:
- Joseph Redmon, Ali Farhadi, "YOLOv3: An Incremental Improvement", arXiv:1804.02767, 2018

2. Required Files:
- YOLOv3 Weights: https://pjreddie.com/media/files/yolov3.weights
- Configuration: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
- COCO Names: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

3. OpenCV Documentation:
- OpenCV DNN Module: https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html
- OpenCV-Python Tutorials: https://docs.opencv.org/master/d6/d00/tutorial_py_root.html

## Usage Instructions

1. Ensure all required files are in the correct directory structure:
```
project/
├── src/
│   ├── yolov3.weights
│   ├── yolov3.cfg
│   └── coco.names
└── media/
    ├── 0.mp4
    ├── 1.mp4
    └── 2.mp4
```

2. Run the script:
```python
python vehicle_detector.py
```

3. Press 'q' to exit the application

## Notes
- The system automatically resizes the display window based on input video dimensions
- Real-time detection results are displayed with bounding boxes and labels
- Total vehicle count is shown in the top-left corner