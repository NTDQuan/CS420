from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort
import torch
import requests
import csv
import os.path as osp
import os

from data.config import cfg_mnet
from data.faces_database import FacesDatabase
from models.face_detection.detector import RetinaFaceClient
from models.face_recognition.identifier import Identifier
from utils.image_util import norm_crop

def visualize_results(frame, dets, landms=None, vis_thres=0.5, names=None, scores=None):
    for i, b in enumerate(dets):
        if b[4] < vis_thres:
            continue

        text = "{:.4f}".format(b[4])
        b = list(map(int, b[:4]))  # Ensure bounding box is in integer format

        # Draw the bounding box
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cx = b[0]
        cy = b[1] - 5
        cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw the name and score
        if names is not None and scores is not None:
            name = names[i] if i < len(names) else "Unknown"
            score = scores[i] if i < len(scores) else "N/A"
            if isinstance(score, np.ndarray):
                score = np.mean(score)
            cv2.putText(frame, f"Name: {name}", (b[0], b[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"Score: {score:.2f}", (b[0], b[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # If landmarks are provided, draw them
        if landms is not None and len(landms) > 0 and i < len(landms):
            l = list(map(int, landms[i]))
            cv2.circle(frame, (l[0], l[1]), 2, (0, 0, 255), -1)  # Right eye
            cv2.circle(frame, (l[2], l[3]), 2, (0, 255, 255), -1)  # Left eye
            cv2.circle(frame, (l[4], l[5]), 2, (255, 0, 255), -1)  # Nose tip
            cv2.circle(frame, (l[6], l[7]), 2, (0, 255, 0), -1)  # Right mouth corner
            cv2.circle(frame, (l[8], l[9]), 2, (255, 0, 0), -1)  # Left mouth corner

    return frame

def process_frame(detector, face_identifier, face_database, images_names, images_embs, frame):
    # Detect faces
    img_raw, dets, landms, orig_width, orig_height = detector.detect_faces(frame)

    names = []
    scores = []

    # Process detections (e.g., recognize or add faces to the database)
    for i in range(len(dets)):
        if landms is not None and len(landms[i]) == 10:
            landmark_reshaped = landms[i].reshape(5, 2)
            aligned_face = norm_crop(img=img_raw, landmark=landmark_reshaped)

            # Identify the person
            score, name = face_identifier.identify(aligned_face, images_embs, images_names)
            names.append(name)
            scores.append(score)

    # Visualize results
    frame = visualize_results(img_raw, dets, landms, vis_thres=0.7, names=names, scores=scores)

    # Resize frame to original size
    frame_resized = cv2.resize(frame, (orig_width, orig_height))
    return names, frame_resized


def send_access_log(personnel_id, frame):
    try:
        # Determine access status
        status = 'Accept' if personnel_id != 'Unknown' else 'Reject'

        # Prepare the timestamp and image filename
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        image_filename = f"{personnel_id}_{timestamp}.jpg"
        image_save_dir = "static/access_images"
        os.makedirs(image_save_dir, exist_ok=True)  # Create directory if it doesn't exist
        image_path = osp.join(image_save_dir, image_filename)

        # Save the image
        cv2.imwrite(image_path, frame)

        # Send the log data to the server
        with open(image_path, 'rb') as image_file:
            files = {'frame': image_file}
            data = {'personnel_id': personnel_id}
            response = requests.post('http://localhost:8000/write_access_log', data=data, files=files)

        if response.status_code == 200:
            print(f"Access log recorded successfully for personnel_id: {personnel_id}")
        else:
            print(f"Failed to record access log: {response.text}")

    except Exception as e:
        print(f"An error occurred while logging access: {e}")

def main():
    # Paths
    database_path = "data/face_bank"

    # Initialize face detector and face identifier
    model_path = "faceDetector.onnx"
    detector = RetinaFaceClient(model_file=model_path, cfg=cfg_mnet)
    face_identifier = Identifier(model_file="iresnet100.onnx")

    # Initialize the FacesDatabase
    face_database = FacesDatabase(path=database_path, face_identifier=face_identifier, face_detector=detector)
    face_database.add_face(raw_face_path="data/face_bank/new_face", face_save_dir="data/face_bank/faces", features_path="data/face_bank/features", profile_dir='static/face')
    images_names, images_embs = face_database.read_feature(feature_path="data/face_bank/features")

    # Start video capture
    source = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = source.read()
            if not ret:
                break

            names, frame_resized = process_frame(detector, face_identifier, face_database, images_names, images_embs, frame)

            # Show the frame
            cv2.imshow("frame", frame_resized)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                for name in names:
                    send_access_log(name, frame)

    finally:
        # Cleanup
        source.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()