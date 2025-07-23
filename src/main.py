import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pytesseract
import subprocess
from tensorflow.keras.models import load_model

# ==============================
# Text-to-Speech Function
# ==============================
def text_to_speech(text):
    if text.strip():  # Ensure there's text to speak
        print(f"Speaking: {text}")
        subprocess.run(["espeak", text], check=True)
    else:
        print("No text detected for speech conversion.")

# ==============================
# Load the Trained Model
# ==============================
def custom_loss(alpha=0.5):
    def loss(y_true, y_pred):
        y_true_resized = tf.image.resize(y_true, tf.shape(y_pred)[1:3])
        L_c_pos = -y_true_resized * tf.keras.backend.log(y_pred + tf.keras.backend.epsilon())
        L_c_neg = -(1 - y_true_resized) * tf.keras.backend.log(1 - y_pred + tf.keras.backend.epsilon())
        return tf.keras.backend.mean(L_c_pos + alpha * L_c_neg)
    return loss

def load_fpn_fcn_model(model_path):
    return load_model(model_path, custom_objects={'custom_loss': custom_loss()})

# ==============================
# Preprocess the Frame (Instead of Image File)
# ==============================
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orig_h, orig_w = gray_frame.shape
    new_h = ((orig_h // 8) + 1) * 8 if orig_h % 8 != 0 else orig_h
    new_w = ((orig_w // 8) + 1) * 8 if orig_w % 8 != 0 else orig_w
    resized_frame = cv2.resize(gray_frame, (new_w, new_h)) / 255.0
    resized_frame = np.expand_dims(resized_frame, axis=[0, -1])
    return resized_frame, (orig_h, orig_w), gray_frame

# ==============================
# Predict Text & Height Maps
# ==============================
def predict_maps(model, frame):
    image, (orig_h, orig_w), original_frame = preprocess_frame(frame)
    pred_text_map, pred_height_map = model.predict(image)
    pred_text_map = cv2.resize(np.squeeze(pred_text_map), (orig_w, orig_h))
    pred_height_map = cv2.resize(np.squeeze(pred_height_map), (orig_w, orig_h))
    return original_frame, pred_text_map, pred_height_map

# ==============================
# Generate Bounding Boxes from Text Map
# ==============================
def remove_overlapping_boxes(bounding_boxes, iou_threshold=0.5):
    def iou(box1, box2):
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2

        inter_x1 = max(x1, x1_)
        inter_y1 = max(y1, y1_)
        inter_x2 = min(x2, x2_)
        inter_y2 = min(y2, y2_)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    filtered_boxes = []
    bounding_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)  # Sort by area (largest first)

    while bounding_boxes:
        current_box = bounding_boxes.pop(0)
        filtered_boxes.append(current_box)

        bounding_boxes = [
            box for box in bounding_boxes if iou(current_box, box) < iou_threshold
        ]

    return filtered_boxes

# ==============================
# Generate Bounding Boxes
# ==============================
def generate_bounding_boxes(text_map, original_image, threshold=0.4, area_ratio_threshold=0.01):
    h, w = original_image.shape[:2]
    text_map_resized = cv2.resize(text_map, (w, h))
    text_map_resized = (text_map_resized * 255).astype(np.uint8)

    _, text_mask = cv2.threshold(text_map_resized, int(threshold * 255), 255, cv2.THRESH_BINARY)
    kernel = np.ones((7,7), np.uint8)
    text_mask = cv2.dilate(text_mask, kernel, iterations=3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(text_mask)
    boxed_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    bounding_boxes = []
    image_area = w * h

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > area_ratio_threshold * image_area:  # Filter based on relative area
            bounding_boxes.append((x, y, x + w, y + h))

    # Step 1: Remove completely nested boxes
    filtered_boxes = []
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        inside = False
        for other in bounding_boxes:
            if other == box:
                continue
            ox1, oy1, ox2, oy2 = other
            if ox1 <= x1 and oy1 <= y1 and ox2 >= x2 and oy2 >= y2:  # If inside another box
                inside = True
                break
        if not inside:
            filtered_boxes.append(box)

    # Step 2: Remove smaller overlapping boxes
    final_boxes = remove_overlapping_boxes(filtered_boxes, iou_threshold=0.5)

    # Draw final bounding boxes
    for x1, y1, x2, y2 in final_boxes:
        cv2.rectangle(boxed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return boxed_image, final_boxes

# ==============================
# Recognize Text Using Tesseract OCR
# ==============================
def recognize_text(original_frame, bounding_boxes, lang="eng"):
    recognized_text = []
    custom_config = f"--psm 6 -l {lang}"

    for i, (x1, y1, x2, y2) in enumerate(bounding_boxes):
        roi = original_frame[y1:y2, x1:x2]
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB) if len(roi.shape) == 2 else roi
        text = pytesseract.image_to_string(roi, config=custom_config).strip() or "[No Text Detected]"
        recognized_text.append({"box": (x1, y1, x2, y2), "text": text})
        print(f"Box {i+1} [{x1}, {y1}, {x2}, {y2}]: {text}")

    return recognized_text

# ==============================
# Real-Time Webcam Processing
# ==============================
def run_real_time_ocr(model_path, lang="eng"):
    print("Loading Model...")
    model = load_fpn_fcn_model(model_path)

    print("Starting Webcam...")
    cap = cv2.VideoCapture(0)  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        cv2.imshow("Live Feed - Press 'S' to Scan | 'Q' to Quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Press 's' to scan
            print("Processing Frame...")
            original_frame, text_map, height_map = predict_maps(model, frame)

            print("Generating Bounding Boxes...")
            boxed_frame, bounding_boxes = generate_bounding_boxes(text_map, original_frame)

            print("Performing OCR...")
            recognized_texts = recognize_text(original_frame, bounding_boxes, lang)

            # Remove empty text and placeholders
            valid_texts = [item["text"] for item in recognized_texts if item["text"].strip() and item["text"] != "[No Text Detected]"]

            if valid_texts:  # Only speak if valid text exists
                full_text = " ".join(valid_texts)
                print("Converting Text to Speech...")
                text_to_speech(full_text)
            else:
                print("No text detected, skipping speech.")

            print("Displaying Processed Frame...")
            cv2.imshow("Processed Frame", boxed_frame)

        elif key == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam Closed.")


# ==============================
# Run the Real-Time Pipeline
# ==============================
if __name__ == "__main__":
    model_path = r"D:\PTU FILES\final project\fpn_fcn_checkpoint_epoch_50.h5"
    run_real_time_ocr(model_path, lang="eng")
