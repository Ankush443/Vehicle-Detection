import cv2
import numpy as np
import os

class ObjectDetector:
    def __init__(self):
        # Print current directory and check files
        print("Current working directory:", os.getcwd())
        print("Checking for YOLO files...")
        
        # Paths to YOLO files
        weights_path = "src/yolov3.weights"
        config_path = "src/yolov3.cfg"
        names_path = "src/coco.names"
        
        # Check if files exist
        self.check_files(weights_path, config_path, names_path)
        
        # Load YOLO
        print("Loading YOLO...")
        self.net = cv2.dnn.readNet(weights_path, config_path)
        
        # Load classes
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layer names
        self.layer_names = self.net.getLayerNames()
        try:
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Detection parameters
        self.conf_threshold = 0.5  # Confidence threshold
        self.nms_threshold = 0.4   # Non-maximum suppression threshold
        
        # Dictionary to store count of each class
        self.class_counts = {}
        
        print("Initialization complete!")

    def check_files(self, weights_path, config_path, names_path):
        missing_files = []
        if not os.path.exists(weights_path):
            missing_files.append("yolov3.weights")
        if not os.path.exists(config_path):
            missing_files.append("yolov3.cfg")
        if not os.path.exists(names_path):
            missing_files.append("coco.names")
            
        if missing_files:
            print("Missing files:", missing_files)
            print("\nPlease download the missing files:")
            print("1. yolov3.weights: https://pjreddie.com/media/files/yolov3.weights")
            print("2. yolov3.cfg: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg")
            print("3. coco.names: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")
            raise FileNotFoundError(f"Missing required files: {missing_files}")

    def detect_objects(self, frame):
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Set input to the network
        self.net.setInput(blob)
        
        # Run forward pass
        outputs = self.net.forward(self.output_layers)
        
        # Initialize lists for detected objects
        boxes = []
        confidences = []
        class_ids = []
        
        # Reset class counts
        self.class_counts = {}
        
        # Process each detection
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter based on confidence threshold
                if confidence > self.conf_threshold:
                    # Scale coordinates back to original image
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        detected = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                class_name = self.classes[class_ids[i]]
                # Update class counts
                self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
                detected.append((boxes[i], confidences[i], class_name))
        
        return detected

    def process_frame(self, frame):
        # Make a copy of the frame
        output_frame = frame.copy()
        
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Generate random colors for each class if not already created
        if not hasattr(self, 'class_colors'):
            self.class_colors = {class_name: tuple(np.random.randint(0, 255, 3).tolist())
                               for class_name in self.classes}
        
        # Draw detections
        for (box, confidence, class_name) in detections:
            x, y, w, h = box
            color = self.class_colors[class_name]
            
            # Draw rectangle
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(output_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw class counts
        y_offset = 30
        for class_name, count in self.class_counts.items():
            count_text = f"{class_name}: {count}"
            color = self.class_colors[class_name]
            cv2.putText(output_frame, count_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        return output_frame

def main():
    print("Starting object detection...")
    
    # Initialize detector
    detector = ObjectDetector()
    
    # Open video capture
    print("Opening video capture...")
    cap = cv2.VideoCapture("test1.mp4")
    
    if not cap.isOpened():
        print("Error: Could not open video capture!")
        return
    
    print("Video capture opened successfully!")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame!")
            break
        
        # Process frame
        output_frame = detector.process_frame(frame)
        
        # Resizing the window
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.resizeWindow("Object Detection", width, height)
        
        # Show frame
        cv2.imshow('Object Detection', output_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")