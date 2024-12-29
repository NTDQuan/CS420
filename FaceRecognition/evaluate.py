import os

import cv2
import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


from data.config import cfg_mnet
from data.faces_database import FacesDatabase

from models.face_detection.detector import RetinaFaceClient
from models.face_recognition.identifier import Identifier
from utils.image_util import norm_crop

def evaluate_recognition(face_identifier, images_names, images_embs, test_dataset_path):
    """
    Evaluate the face recognition model.
    :param face_identifier: Face recognition model instance.
    :param face_database: Loaded face embeddings and names from the database.
    :param test_dataset: Test dataset with images and ground-truth labels.
    :return: Evaluation metrics.
    """
    true_labels = []
    predicted_labels = []

    # Iterate over the test dataset
    for filename in os.listdir(test_dataset_path):
        image_path = os.path.join(test_dataset_path, filename)
        image = cv2.imread(image_path)
        print(f"Processing image: {filename}")
        true_labels.append(filename.split("_")[0])

        raw_image, dets, landms, _, _ = detector.detect_faces(image)

        aligned_face = None

        for i in range(len(dets)):
            # Reshape landmark to (5, 2)
            try:
                landmark_reshaped = landms[i].reshape(5, 2)
            except ValueError as e:
                print(f"Error reshaping landmark: {landms[i]}. Skipping this face.")
                continue

            aligned_face = norm_crop(img=raw_image, landmark=landmark_reshaped)

            # Identify the person
            score, name = face_identifier.identify(aligned_face, images_embs, images_names)
            predicted_labels.append(name)

    # Calculate evaluation metrics
    print("True label ", true_labels)
    print("Predicted label", predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average="weighted")
    recall = recall_score(true_labels, predicted_labels, average="weighted")
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    cm = confusion_matrix(true_labels, predicted_labels)

    return accuracy, precision, recall, f1, cm

if __name__ == "__main__":
    from data.faces_database import FacesDatabase
    from models.face_recognition.identifier import Identifier
    from utils.image_util import norm_crop

    database_path = "data/face_bank"

    # Initialize face detector and face identifier
    model_path = "faceDetector.onnx"
    detector = RetinaFaceClient(model_file=model_path, cfg=cfg_mnet)
    face_identifier = Identifier(model_file="iresnet100.onnx")

    # Initialize the FacesDatabase
    face_database = FacesDatabase(path=database_path, face_identifier=face_identifier, face_detector=detector)
    face_database.add_face(raw_face_path="data/face_bank/new_face", face_save_dir="data/face_bank/faces", features_path="data/face_bank/features")
    images_names, images_embs = face_database.read_feature(feature_path="data/face_bank/features")

    # Evaluate the model
    accuracy, precision, recall, f1, cm = evaluate_recognition(face_identifier, images_names, images_embs, "Face_test")

    # Print evaluation results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(cm)