import cv2
import numpy as np
import os

# Constants
DATASET_PATH = "dataset/"  # Folder where images are stored
IMAGE_SIZE = (100, 100)    # Resize all images to 100x100

# Step 1: Load dataset and preprocess
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

# Step 2: Train EigenFace Recognizer
def train_recognizer(images, labels):
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(images, labels)
    recognizer.save("face_model.yml")  # Save trained model
    return recognizer

# Step 3: Recognize face from webcam
def recognize_faces(recognizer, label_dict):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    reverse_label_dict = {v: k for k, v in label_dict.items()}  # Reverse mapping

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, IMAGE_SIZE)

            label, confidence = recognizer.predict(face)
            name = reverse_label_dict.get(label, "Unknown")

            # Draw rectangle and label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Live Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Execution

print("Loading dataset...")
images, labels, label_dict = load_images(DATASET_PATH)

print("Training model...")
recognizer = train_recognizer(images, labels)

print("Starting real-time face recognition...")
recognize_faces(recognizer, label_dict)
