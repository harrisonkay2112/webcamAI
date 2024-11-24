import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the MobileNet SSD model from TensorFlow Hub
print("Loading model...")
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Model loaded!")

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the webcam.")
        break

    # Convert frame to RGB and resize for the model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = tf.image.resize(rgb_frame, (640, 640))  # Model expects 640x640
    input_tensor = tf.convert_to_tensor([resized_frame], dtype=tf.uint8)  # Ensure dtype is uint8

    # Perform object detection
    detections = detector(input_tensor)
    detection_boxes = detections["detection_boxes"].numpy()[0]
    detection_classes = detections["detection_classes"].numpy()[0].astype(int)
    detection_scores = detections["detection_scores"].numpy()[0]

    # Draw detection results on the frame
    height, width, _ = frame.shape
    for i in range(len(detection_boxes)):
        if detection_scores[i] > 0.5:  # Confidence threshold
            ymin, xmin, ymax, xmax = detection_boxes[i]
            (left, top, right, bottom) = (xmin * width, ymin * height, xmax * width, ymax * height)

            # Draw bounding box
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            # Label the detection
            label = f"Object {detection_classes[i]}: {detection_scores[i]:.2f}"
            cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the video feed with detections
    cv2.imshow("Object Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()