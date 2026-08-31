"""Real-time webcam object detection with SSD MobileNet V2.

Captures frames from the default webcam, runs each through an SSD MobileNet V2
detector pretrained on COCO, draws labelled bounding boxes, announces newly
seen objects out loud, and writes a CSV log of every detection on exit.

Press 'q' in the video window to quit.
"""

import datetime

import cv2
import numpy as np
import pandas as pd
import pyttsx3
import tensorflow as tf
import tensorflow_hub as hub

# --- Configuration ---------------------------------------------------------

MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
MODEL_INPUT_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.5
LOG_PATH = "detections.csv"

# COCO label map, keyed by the class ID the model actually emits.
#
# This model returns IDs from the 90-entry COCO map, which has gaps -- IDs 12,
# 26, 29, 30, 45, 66, 68, 69, 71 and 83 are unused. Indexing a dense 80-element
# list with `id - 1` therefore drifts out of alignment at the first gap and
# mislabels everything after it (a stop sign, ID 13, comes back as "parking
# meter"). Keying by ID avoids the problem entirely.
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard",
    42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass",
    47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
    53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
    58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier",
    90: "toothbrush",
}


def main():
    print("Loading model...")
    detector = hub.load(MODEL_URL)
    print("Model loaded!")

    engine = pyttsx3.init()
    detections_log = []

    # Objects announced on the previous frame. Speech is synchronous, so
    # announcing every detection every frame would stall the capture loop
    # several times a second. Only speak when something new appears.
    previously_seen = set()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from the webcam.")
                break

            # The model wants a uint8 RGB tensor at a fixed size.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = tf.image.resize(rgb_frame, MODEL_INPUT_SIZE)
            input_tensor = tf.convert_to_tensor(
                [resized_frame], dtype=tf.uint8
            )

            detections = detector(input_tensor)
            boxes = detections["detection_boxes"].numpy()[0]
            classes = detections["detection_classes"].numpy()[0].astype(int)
            scores = detections["detection_scores"].numpy()[0]

            height, width, _ = frame.shape
            currently_seen = set()

            for box, class_id, score in zip(boxes, classes, scores):
                if score < CONFIDENCE_THRESHOLD:
                    continue

                # Unknown IDs are possible; don't crash on them.
                label = COCO_LABELS.get(class_id, f"unknown ({class_id})")
                currently_seen.add(label)

                ymin, xmin, ymax, xmax = box
                left, top = int(xmin * width), int(ymin * height)
                right, bottom = int(xmax * width), int(ymax * height)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{label}: {score:.2f}", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                )

                detections_log.append({
                    "timestamp": datetime.datetime.now(),
                    "object": label,
                    "score": float(score),
                })

            for label in currently_seen - previously_seen:
                engine.say(f"I see a {label}")
            if currently_seen != previously_seen:
                engine.runAndWait()
            previously_seen = currently_seen

            cv2.imshow("Object Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Runs on 'q', on a read failure, and on Ctrl-C, so the log is never
        # lost just because the loop ended unexpectedly.
        if detections_log:
            pd.DataFrame(detections_log).to_csv(LOG_PATH, index=False)
            print(f"Wrote {len(detections_log)} detections to {LOG_PATH}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
