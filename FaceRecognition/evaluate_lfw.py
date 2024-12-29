import cv2
import numpy as np
import os
import logging
import csv

from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data.config import cfg_mnet
from models.face_detection.detector import RetinaFaceClient
from models.face_recognition.identifier import Identifier
from utils.image_util import preprocess_image, norm_crop

LFW_PATH = 'lfw/lfw-deepfunneled/lfw-deepfunneled'
PAIRS_FILE = 'lfw/pairs.csv'
DETECTOR_MODEL = 'faceDetector.onnx'
IDENTIFIER_MODEL = 'iresnet100.onnx'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(path):
    logging.info(f"Loading image from {path}")
    img = cv2.imread(path)
    return img

def evaluate_lfw(detector, identifier, lfw_path, pairs_file):
    true_labels = []
    pred_labels = []

    with open(pairs_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header

        for row in tqdm(reader, desc="Processing pairs"):
            row = [item for item in row if item]  # Remove empty strings
            if len(row) == 3:
                # Positive pair
                name, img1, img2 = row
                if img1.isdigit() and img2.isdigit():
                    img1_path = os.path.join(lfw_path, name, f"{name}_{int(img1):04d}.jpg")
                    img2_path = os.path.join(lfw_path, name, f"{name}_{int(img2):04d}.jpg")
                    true_labels.append(1)
                else:
                    logging.warning(f"Invalid image numbers: {row}")
                    continue
            elif len(row) == 4:
                # Negative pair
                name1, img1, name2, img2 = row
                if img1.isdigit() and img2.isdigit():
                    img1_path = os.path.join(lfw_path, name1, f"{name1}_{int(img1):04d}.jpg")
                    img2_path = os.path.join(lfw_path, name2, f"{name2}_{int(img2):04d}.jpg")
                    true_labels.append(0)
                else:
                    logging.warning(f"Invalid image numbers: {row}")
                    continue
            else:
                logging.warning(f"Invalid pair format: {row}")
                continue

            img1 = load_image(img1_path)
            img2 = load_image(img2_path)

            # Detect faces
            logging.info("Detecting faces in the first image")
            img1_raw, dets1, landms1, _, _ = detector.detect_faces(img1)
            logging.info("Detecting faces in the second image")
            img2_raw, dets2, landms2, _, _ = detector.detect_faces(img2)

            if dets1.shape[0] == 0 or dets2.shape[0] == 0:
                logging.warning("No faces detected in one of the images.")
                pred_labels.append(0)
                continue

            # Align and crop faces using norm_crop
            if landms1 is not None and len(landms1[0]) == 10:
                landmark_reshaped1 = landms1[0].reshape(5, 2)
                face1 = norm_crop(img=img1_raw, landmark=landmark_reshaped1)
            else:
                logging.warning("Landmarks for the first image are not available or incorrect.")
                pred_labels.append(0)
                continue

            if landms2 is not None and len(landms2[0]) == 10:
                landmark_reshaped2 = landms2[0].reshape(5, 2)
                face2 = norm_crop(img=img2_raw, landmark=landmark_reshaped2)
            else:
                logging.warning("Landmarks for the second image are not available or incorrect.")
                pred_labels.append(0)
                continue

            # Get embeddings
            logging.info("Extracting embeddings for the first face")
            emb1 = identifier.represent(face1)
            logging.info("Extracting embeddings for the second face")
            emb2 = identifier.represent(face2)

            # Compare embeddings
            logging.info("Comparing embeddings")
            score, _ = identifier.compare_encodings(emb1, np.array([emb2]))
            pred_labels.append(1 if score >= 0.5 else 0)

    accuracy = accuracy_score(true_labels, pred_labels)
    logging.info(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    # Initialize the models
    logging.info("Initializing the face detector")
    detector = RetinaFaceClient(model_file=DETECTOR_MODEL, cfg=cfg_mnet)
    logging.info("Initializing the identifier")
    identifier = Identifier(model_file=IDENTIFIER_MODEL)

    # Evaluate the models on LFW dataset
    logging.info("Evaluating the models on LFW dataset")
    evaluate_lfw(detector, identifier, LFW_PATH, PAIRS_FILE)