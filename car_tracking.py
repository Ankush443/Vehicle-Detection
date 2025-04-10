import cv2
import numpy as np
import os

class VehicleDetector:
    def __init__(self):
        # Print current directory and check files
        print("Current working directory:", os.getcwd())
        print("Checking for YOLO files...")
        
        # Paths to YOLO files
        weights_path = "src\yolov3.weights"
        config_path = "src\yolov3.cfg"
        names_path = "src\coco.names"
        
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
        
        # Counter
        self.vehicle_count = 0
        
        # Detection parameters
        self.conf_threshold = 0.5  # Confidence threshold
        self.nms_threshold = 0.4   # Non-maximum suppression threshold
        
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

    def detect_vehicles(self, frame):
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
        
        # Process each detection
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for vehicles (car, truck, bus, motorcycle)
                if confidence > self.conf_threshold and class_id in [2, 3, 5, 7]:
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
                detected.append((boxes[i], confidences[i], self.classes[class_ids[i]]))
        
        return detected

    def process_frame(self, frame):
        # Make a copy of the frame
        output_frame = frame.copy()
        
        # Detect vehicles
        detections = self.detect_vehicles(frame)
        
        # Draw detections
        for (box, confidence, class_name) in detections:
            x, y, w, h = box
            
            # Draw rectangle
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(output_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw total count
        count_text = f"Vehicles Detected: {len(detections)}"
        cv2.putText(output_frame, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return output_frame

def main():
    print("Starting vehicle detection...")
    
    # Define file paths
    file_path_0 = r"media\0.mp4"
    file_path_1 = r"media\1.mp4"
    file_path_2 = r"media\2.mp4"
    
    # Initialize detector
    detector = VehicleDetector()
    
    # Try to open video capture
    print("Opening video capture...")
    
    # Run the video
    cap = cv2.VideoCapture(file_path_2)  
    
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
        cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the frames in the video
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the frames in the video
        cv2.resizeWindow("Vehicle Detection", width, height)

        # Show frame
        cv2.imshow('Vehicle Detection', output_frame)
        
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