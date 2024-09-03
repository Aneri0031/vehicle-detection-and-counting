import cv2
import numpy as np

# Load SSD MobileNet model and COCO class names
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

# COCO class names for SSD MobileNet
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Initialize video capture
cap = cv2.VideoCapture("video.mp4")  # Replace with your video file

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define vehicle classes and ROI line for counting
vehicle_classes = ["car", "bus", "motorbike"]  # Available classes in SSD MobileNet
line_position = 450  # Y-coordinate of the counting line
font = cv2.FONT_HERSHEY_SIMPLEX

# Vehicle count initialization
vehicle_count = {"car": 0, "bus": 0, "motorbike": 0}

# Vehicle tracking variables
tracker = {}  # Dictionary to store vehicle IDs and their positions
vehicle_id = 0  # Unique ID for each detected vehicle
vehicle_crossed = set()  # Set to keep track of counted vehicles

def draw_boxes(frame, detections, previous_centers):
    """Draw bounding boxes and count vehicles crossing the line."""
    global vehicle_id, vehicle_count
    new_centers = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        print(confidence)
        if confidence > 0.3:  # Adjusted confidence threshold
            class_id = int(detections[0, 0, i, 1])
            label = classes[class_id]

            if label in vehicle_classes:
                # Extract the bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x_max, y_max) = box.astype("int")

                # Check if bounding box is within frame limits
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                x_max = max(0, min(x_max, width - 1))
                y_max = max(0, min(y_max, height - 1))

                # Debugging: Print detection information
                print(f"Detected {label} with confidence {confidence:.2f} at ({x}, {y}), ({x_max}, {y_max})")

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), font, 0.5, (255, 255, 255), 2)

                # Calculate the center point of the bounding box
                center_y = (y + y_max) // 2
                center_x = (x + x_max) // 2
                new_centers.append((center_x, center_y))

                # Track vehicle ID
                for cid, (prev_x, prev_y) in previous_centers.items():
                    # Calculate Euclidean distance to determine if it's the same vehicle
                    if np.linalg.norm(np.array([center_x, center_y]) - np.array([prev_x, prev_y])) < 50:
                        tracker[cid] = (center_x, center_y)
                        # Check if the vehicle has crossed the line
                        if center_y > line_position and (cid, label) not in vehicle_crossed:
                            vehicle_count[label] += 1
                            vehicle_crossed.add((cid, label))
                        break
                else:
                    # Assign a new ID to new vehicles
                    tracker[vehicle_id] = (center_x, center_y)
                    vehicle_id += 1

    return {k: v for k, v in tracker.items() if k in [i for i, _ in enumerate(new_centers)]}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read the frame.")
        break

    height, width, _ = frame.shape

    # Preprocess input frame for SSD
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    
    #print("Processing frame...")  # Debugging statement
    
    try:
        detections = net.forward()  # This is where the model processes the frame
        #print("Detection complete.")  # Debugging statement
    except Exception as e:
        print(f"Error during forward pass: {e}")
        break

    # Track vehicles and count crossings
    tracker = draw_boxes(frame, detections, tracker)

    # Draw the counting line
    cv2.line(frame, (0, line_position), (width, line_position), (0, 0, 255), 2)

    # Display vehicle count on the frame
    cv2.putText(frame, f"Cars: {vehicle_count['car']}  Buses: {vehicle_count['bus']}  Motorbikes: {vehicle_count['motorbike']}",
                (10, 50), font, 1, (0, 255, 255), 2)

    # Show video frame
    cv2.imshow("Vehicle Detection and Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
