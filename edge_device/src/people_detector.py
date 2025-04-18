import cv2
import numpy as np
from ultralytics import YOLO
import time

class PersonDetector:
    """
    A class for detecting people in video streams using YOLOv8.
    Triggers events when 2 or more people are detected.
    Press 's' to stop the detection cleanly.
    """
    
    def __init__(self, model_path='edge_device/models/yolov8n.pt', confidence=0.5):
        """
        Initialize the person detector.
        
        Args:
            model_path: Path to YOLOv8 model file (.pt)
            confidence: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.person_class_id = 0  # In COCO dataset, person is class 0
        self.current_count = 0
        self.callbacks = []
        self.running = False
        
    def register_multi_person_callback(self, callback):
        """
        Register a callback function to be called when 2+ people are detected.
        The callback will receive the number of people detected as an argument.
        
        Args:
            callback: Function to call when multiple people are detected
        """
        self.callbacks.append(callback)
        
    def _notify_callbacks(self, count):
        """
        Notify all registered callbacks about person count.
        
        Args:
            count: Number of people detected
        """
        for callback in self.callbacks:
            callback(count)
    
    def process_frame(self, frame):
        """
        Process a single frame and detect people.
        
        Args:
            frame: The image frame to process
            
        Returns:
            Annotated frame with detection boxes and person count
        """
        # Run YOLOv8 inference on the frame
        results = self.model(frame)
        
        # Get the detections
        result = results[0]
        
        # Count people (class 0 in COCO dataset)
        person_count = 0
        
        annotated_frame = frame.copy()
        
        for box in result.boxes:
            # Check if detection is a person and confidence is above threshold
            if int(box.cls) == self.person_class_id and box.conf >= self.confidence:
                person_count += 1
                
                # Get the bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw rectangle around person
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display the count on the frame
        cv2.putText(annotated_frame, f"People count: {person_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add instructions text
        cv2.putText(annotated_frame, "Press 's' to stop", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Check if count has changed
        if person_count != self.current_count:
            self.current_count = person_count
            # Notify if 2 or more people are detected
            if person_count >= 2:
                self._notify_callbacks(person_count)
        
        return annotated_frame, person_count
    
    def start_camera(self, camera_id=0):
        """
        Start detection with a camera feed.
        
        Args:
            camera_id: Camera device ID (default: 0 for primary camera)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print("Starting person detection. Press 's' to stop cleanly.")
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Process the frame
                annotated_frame, count = self.process_frame(frame)
                
                # Display the result
                cv2.imshow("Person Detection", annotated_frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                
                # Exit on 's' key
                if key == ord('s'):
                    print("Stopping detection gracefully...")
                    self.running = False
                    break
                
        finally:
            # Release resources
            print("Cleaning up resources...")
            cap.release()
            cv2.destroyAllWindows()
            print("Detection stopped successfully.")
    
    def stop(self):
        """
        Stop the detection process externally if needed.
        """
        self.running = False
        print("Stop signal received. Detection will stop after current frame.")


# Example usage:
if __name__ == "__main__":
    def on_multiple_people(count):
        print(f"Alert: {count} people detected!")
    
    # Create detector instance
    detector = PersonDetector()
    
    # Register callback for when multiple people are detected
    detector.register_multi_person_callback(on_multiple_people)
    
    # Start webcam detection
    detector.start_camera()