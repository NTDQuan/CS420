import os
import os.path as osp
import shutil

import cv2
import numpy as np

from utils.image_util import norm_crop


class FacesDatabase:
    def __init__(self, path, face_identifier, face_detector):
        self.path = path
        self.face_identifier = face_identifier
        self.face_detector = face_detector

    @staticmethod
    def _is_image_file(filename):
        return filename.lower().endswith(("jpg", "png", "jpeg"))

    def add_face(self, raw_face_path, face_save_dir, features_path, profile_dir):
        images_name = []
        images_emb = []

        # Create backup directories
        backup_raw_face_dir = raw_face_path + "_backup"

        try:
            # Backup the current face_save_dir and features_path
            if os.path.exists(face_save_dir):
                shutil.copytree(raw_face_path, backup_raw_face_dir, dirs_exist_ok=True)
                shutil.copytree(raw_face_path, profile_dir, dirs_exist_ok=True)
        except PermissionError as e:
            print(f"Permission error: {e}. Please check file permissions.")
        except Exception as e:
            print(f"An error occurred during backup: {e}")

        # Batch processing for images
        for name_person in os.listdir(raw_face_path):
            person_image_path = osp.join(raw_face_path, name_person)
            person_face_path = osp.join(face_save_dir, name_person)
            os.makedirs(person_face_path, exist_ok=True)

            for image_name in filter(self._is_image_file, os.listdir(person_image_path)):
                input_image = cv2.imread(osp.join(person_image_path, image_name))
                print(f"Processing image: {image_name} for person: {name_person}")

                # Detect faces
                raw_image, bboxes, landmarks, _, _ = self.face_detector.detect_faces(input_image)

                if len(bboxes) == 0 or len(landmarks) == 0:
                    print(f"No faces detected in image: {image_name}")
                    continue  # Skip to next image

                for i in range(len(bboxes)):
                    if landmarks[i].shape != (10,):
                        print(f"Invalid landmark shape: {landmarks[i]}. Skipping this face.")
                        continue

                    landmark_reshaped = landmarks[i].reshape(5, 2)

                    # Align face and compute embeddings
                    aligned_face = norm_crop(raw_image, landmark_reshaped)
                    cv2.imwrite(osp.join(person_face_path, f"{i}.jpg"), aligned_face)
                    face_embedding = self.face_identifier.represent(aligned_face)

                    # Append face embedding and name
                    if face_embedding is not None:
                        images_emb.append(face_embedding)
                        images_name.append(name_person)

        # After adding, delete every file in the new_face folder
        for name_person in os.listdir(raw_face_path):
            person_image_path = osp.join(raw_face_path, name_person)
            shutil.rmtree(person_image_path)

        if images_emb == [] and images_name == []:
            print("No new person found!")
            return None

        images_emb = np.array(images_emb)
        images_name = np.array(images_name)

        features = self.read_feature(features_path)

        print(features)

        if features is not None:
            old_images_name, old_images_emb = features

            images_name = np.hstack((old_images_name, images_name))
            images_emb = np.vstack((old_images_emb, images_emb))

            print("Update features")

        np.savez_compressed(features_path, images_name=images_name, images_emb=images_emb)

    def read_feature(self, feature_path):
        try:
            data = np.load(feature_path + ".npz", allow_pickle=True)
            return data["images_name"], data["images_emb"]
        except FileNotFoundError:
            return None


    def delete_person(self, feature_path, delete_id):
        try:
            # Load existing features
            data = np.load(feature_path + ".npz", allow_pickle=True)
            images_name = data["images_name"]
            images_emb = data["images_emb"]

            # Check if the ID exists
            if delete_id not in images_name:
                print(f"ID '{delete_id}' not found in the database.")
                return

            # Filter out the person to be deleted
            mask = images_name != delete_id
            updated_images_name = images_name[mask]
            updated_images_emb = images_emb[mask]

            # Save the updated features back to the file
            np.savez_compressed(feature_path, images_name=updated_images_name, images_emb=updated_images_emb)
            print(f"Successfully deleted person with ID '{delete_id}' from the database.")
        except FileNotFoundError:
            print(f"Feature file '{feature_path}' not found. Cannot delete person.")
        except Exception as e:
            print(f"An error occurred while deleting person with ID '{delete_id}': {e}")

    def edit_person(self, feature_path, old_name, new_name):
        try:
            # Load existing features
            data = np.load(feature_path + ".npz", allow_pickle=True)
            images_name = data["images_name"]
            images_emb = data["images_emb"]

            # Check if the old name exists
            if old_name not in images_name:
                print(f"Person '{old_name}' not found in the database.")
                return

            # Update the name
            images_name[images_name == old_name] = new_name

            # Save the updated features back to the file
            np.savez_compressed(feature_path, images_name=images_name, images_emb=images_emb)
            print(f"Successfully edited person name from '{old_name}' to '{new_name}'.")
        except FileNotFoundError:
            print(f"Feature file '{feature_path}' not found. Cannot edit person.")
        except Exception as e:
            print(f"An error occurred while editing person name from '{old_name}' to '{new_name}': {e}")

