import cv2
import numpy as np
import os
import time

# Constants
DATASET_PATH = "dataset/"  # Folder where images are stored
IMAGE_SIZE = (100, 100)    # Resize all images to 100x100
MODEL_PATH = "face_model.yml"  # Trained model file
WINDOW_SIZE = (800, 600)  # Window size for displaying the image
CONFIDENCE_THRESHOLD = 0.61  # Minimum confidence to recognize a face

# Load dataset and preprocess
def load_images(path):
    images = []
    labels = []
    label_dict = {}  # Map names to numeric labels
    label_count = 0

    for person in os.listdir(path):
        person_path = os.path.join(path, person)
        if not os.path.isdir(person_path):
            continue

        if person not in label_dict:
            label_dict[person] = label_count
            label_count += 1

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            img = cv2.resize(img, IMAGE_SIZE)

            images.append(img)
            labels.append(label_dict[person])

    return images, np.array(labels), label_dict

# Train EigenFace Recognizer
def train_recognizer(images, labels):
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(images, labels)
    recognizer.save(MODEL_PATH)  # Save trained model
    return recognizer

# Recognize face from an image
def recognize_face_from_image(image_path, recognizer, label_dict):
    start_time = time.time()  # Start timing

    reverse_label_dict = {v: k for k, v in label_dict.items()}  # Reverse mapping
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load the image.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, IMAGE_SIZE)
        
        label, confidence = recognizer.predict(face)
        
        # Normalize confidence score
        max_confidence = 10000  # Assumed max confidence, change if needed
        normalized_confidence = 1 - (confidence / max_confidence)
        normalized_confidence = max(0.0, min(1.0, normalized_confidence))  # Clamp between 0 and 1
        
        if normalized_confidence < CONFIDENCE_THRESHOLD:
            name = "Unknown"
        else:
            name = reverse_label_dict.get(label, "Unknown")
        
        # Draw rectangle and label
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{name} ({normalized_confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time  # Compute execution time
    print(f"Recognition completed in {elapsed_time:.4f} seconds.")

    resized_img = cv2.resize(img, WINDOW_SIZE)
    cv2.imshow("Recognized Face", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main Execution
if __name__ == "__main__":
    total_start_time = time.time()  # Start total execution time

    print("Loading dataset...")
    dataset_start_time = time.time()
    images, labels, label_dict = load_images(DATASET_PATH)
    dataset_end_time = time.time()
    print(f"Dataset loaded in {dataset_end_time - dataset_start_time:.4f} seconds.")

    print("Training model...")
    training_start_time = time.time()
    recognizer = train_recognizer(images, labels)
    training_end_time = time.time()
    print(f"Model trained in {training_end_time - training_start_time:.4f} seconds.")

    test_image_path = input("input the path of the img: ")
    recognize_face_from_image(test_image_path, recognizer, label_dict)

    total_end_time = time.time()
    print(f"Total execution time: {total_end_time - total_start_time:.4f} seconds.")
