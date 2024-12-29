import csv
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
import os
import os.path as osp
from io import BytesIO
import sqlite3

from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from data.config import cfg_mnet
from data.faces_database import FacesDatabase
from models.face_detection.detector import RetinaFaceClient
from models.face_recognition.identifier import Identifier
from utils.image_util import norm_crop

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Outputs to the console
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Paths
database_path = "data/face_bank"
model_path = "faceDetector.onnx"
face_save_dir = "data/face_bank/faces"
features_path = "data/face_bank/features"

# Initialize face detector and face identifier
detector = RetinaFaceClient(model_file=model_path, cfg=cfg_mnet)
face_identifier = Identifier(model_file="iresnet100.onnx")

# Initialize the FacesDatabase
face_database = FacesDatabase(path=database_path, face_identifier=face_identifier, face_detector=detector)

class FrameRequest(BaseModel):
    frame: str

class Person(BaseModel):
    name: str
    image: Optional[str] = None  # Base64 encoded image

class DeletePersonRequest(BaseModel):
    id: str

def init_db():
    conn = sqlite3.connect("data/server_face_database.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS person (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        image TEXT,
                        create_time TEXT,
                        modified_time TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS access (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER,
                        person_name TEXT,
                        image TEXT,
                        access_date TEXT,
                        status TEXT,
                        FOREIGN KEY(person_id) REFERENCES person(id))''')
    conn.commit()
    conn.close()

init_db()

@app.post("/add_person")
async def add_person(id: str = Form(...), name: str = Form(...), file: UploadFile = File(...)):
    logger.info(f"Received request to add person with id: {id}")
    logger.info(f"Received request to add person with name: {name}")
    logger.debug(f"Received file: {file.filename}")

    # Validate inputs
    if not id:
        logger.error("Id is required but not provided.")
        return JSONResponse(status_code=400, content={"error": "Id is required"})
    if not name:
        logger.error("Name is required but not provided.")
        return JSONResponse(status_code=400, content={"error": "Name is required"})
    if not file.filename.endswith(("jpg", "jpeg", "png")):
        logger.error("Invalid file format uploaded.")
        return JSONResponse(status_code=400,
                            content={"error": "Invalid file format. Only jpg, jpeg, and png are supported."})

    # Paths
    raw_face_path = osp.join("data/face_bank", "new_face")
    os.makedirs(raw_face_path, exist_ok=True)
    logger.debug(f"Raw face path created or exists: {raw_face_path}")

    new_person_path = osp.join(raw_face_path, id)
    os.makedirs(new_person_path, exist_ok=True)
    file_path = osp.join(new_person_path, file.filename)
    logger.debug(f"File path for uploaded image: {file_path}")

    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(f"File saved successfully: {file_path}")

        # Load the image
        image = cv2.imread(file_path)
        if image is None:
            logger.error("Uploaded file could not be read as an image.")
            return JSONResponse(status_code=400, content={"error": "Uploaded file could not be read as an image."})

        # Detect faces
        faces = detector.detect_faces(image)
        logger.debug(f"Faces detected: {faces}")
        if len(faces[1]) == 0:
            logger.warning("No face detected in the uploaded image.")
            return JSONResponse(status_code=400, content={"error": "No face detected in the uploaded image."})
        if len(faces[1]) > 1:
            logger.warning("Multiple faces detected in the uploaded image.")
            return JSONResponse(status_code=400, content={
                "error": "Multiple faces detected. Please upload an image with a single face."})

        # Crop and normalize the detected face
        landmark_reshaped = faces[2][0].reshape(5, 2)
        cropped_face = norm_crop(image, landmark_reshaped)
        cropped_face_path = osp.join(face_save_dir, f"{id}.jpg")
        cv2.imwrite(cropped_face_path, cropped_face)
        logger.info(f"Cropped face saved successfully: {cropped_face_path}")

        # Encode the face
        face_database.add_face(raw_face_path="data/face_bank/new_face", face_save_dir="data/face_bank/faces", features_path="data/face_bank/features", profile_dir='static/face')
        logger.info(f"Person added to database: {id}")

        profile_folder_path = osp.join('static/face', id)
        profile_image_path = osp.join(profile_folder_path, file.filename)

        # Insert into SQLite database
        conn = sqlite3.connect("data/server_face_database.db")
        cursor = conn.cursor()
        create_time = modified_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO person (id, name, image, create_time, modified_time) VALUES (?, ?, ?, ?, ?)",
                       (id, name, profile_image_path, create_time, modified_time))
        conn.commit()
        conn.close()

        return {"status": "Person added successfully"}
    except Exception as e:
        logger.exception("An error occurred while processing the face.")
        return JSONResponse(status_code=500,
                            content={"error": f"An error occurred while processing the face: {str(e)}"})

@app.post("/delete_person")
async def delete_person(request: DeletePersonRequest):
    logger.info(f"Received request to delete person with id: {request.id}")

    # Check the received request payload
    logger.debug(f"Request payload: {request.dict()}")

    # Delete from FacesDatabase
    conn = sqlite3.connect("data/server_face_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, image FROM person WHERE id = ?", (request.id,))
    person_data = cursor.fetchone()

    if person_data:
        person_name, image_path = person_data
        face_database.delete_person(feature_path="data/face_bank/features", delete_id=person_name)

        cursor.execute("DELETE FROM person WHERE id = ?", (request.id,))
        conn.commit()

        # Remove the person's images from the file system
        if image_path:
            try:
                os.remove(image_path)
                logger.info(f"Deleted image: {image_path}")
            except FileNotFoundError:
                logger.warning(f"Image file not found: {image_path}")
            except Exception as e:
                logger.error(f"An error occurred while deleting image file {image_path}: {e}")

    conn.close()
    return {"status": "Person deleted"}

@app.post("/edit_person")
async def edit_person(old_name: str = Form(...), new_name: str = Form(...)):
    logger.info(f"Received request to edit person from name: {old_name} to new name: {new_name}")

    # Edit in FacesDatabase
    face_database.edit_person(feature_path="data/face_bank/features", old_name=old_name, new_name=new_name)

    # Edit in SQLite database and rename images
    conn = sqlite3.connect("data/server_face_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, image FROM person WHERE name = ?", (old_name,))
    person_data = cursor.fetchone()

    if person_data:
        person_id, old_image_path = person_data
        new_image_path = old_image_path.replace(old_name, new_name)
        cursor.execute("UPDATE person SET name = ?, image = ? WHERE id = ?", (new_name, new_image_path, person_id))
        conn.commit()

        # Rename the person's images in the file system
        if old_image_path:
            try:
                os.rename(old_image_path, new_image_path)
                logger.info(f"Renamed image from {old_image_path} to {new_image_path}")
            except FileNotFoundError:
                logger.warning(f"Image file not found: {old_image_path}")
            except Exception as e:
                logger.error(f"An error occurred while renaming image file {old_image_path}: {e}")

    conn.close()
    return {"status": "Person edited"}


@app.post("/write_access_log")
async def write_access_log(personnel_id: str = Form(...), frame: UploadFile = File(...)):
    logger.info(f"Received request to log access for id: {personnel_id}")

    try:
        status = 'Accept' if personnel_id != 'Unknown' else 'Reject'
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        image_filename = f"{personnel_id}_{timestamp}.jpg"
        image_save_dir = "static/access_images"
        os.makedirs(image_save_dir, exist_ok=True)
        image_path = osp.join(image_save_dir, image_filename)

        with open(image_path, "wb") as buffer:
            buffer.write(await frame.read())
        logger.info(f"Image saved successfully: {image_path}")

        conn = sqlite3.connect("data/server_face_database.db")
        cursor = conn.cursor()

        # Fetch the person's name using the personnel_id
        cursor.execute("SELECT name FROM person WHERE id = ?", (personnel_id,))
        person_data = cursor.fetchone()

        if person_data:
            person_name = person_data[0]
        else:
            person_name = "Unknown"

        # Insert the access log into the database
        cursor.execute('''INSERT INTO access (person_id, person_name, image, access_date, status)
                          VALUES (?, ?, ?, ?, ?)''',
                       (personnel_id, person_name, image_path,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), status))

        conn.commit()
        conn.close()

        logger.info(f"Access log recorded: {personnel_id}, {person_name}, {status}, {image_path}")
        return {"status": "Access log recorded successfully"}

    except Exception as e:
        logger.error(f"An error occurred while logging access: {e}")
        return JSONResponse(status_code=500, content={"error": f"An error occurred while logging access: {str(e)}"})

@app.get("/access_logs")
async def get_all_access_logs():
    logger.info("Fetching all access logs")
    try:
        conn = sqlite3.connect("data/server_face_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM access")
        rows = cursor.fetchall()
        conn.close()

        access_logs = []
        for row in rows:
            image_path = row[3].replace("\\", "/")
            access_logs.append({
                "id": row[0],
                "person_id": row[1],
                "person_name": row[2],
                "image": image_path,
                "access_date": row[4],
                "status": row[5]
            })

        return {"access_logs": access_logs}

    except Exception as e:
        logger.error(f"An error occurred while fetching access logs: {e}")
        return JSONResponse(status_code=500, content={"error": f"An error occurred while fetching access logs: {str(e)}"})

@app.get("/persons")
async def get_all_persons():
    logger.info("Fetching all persons")
    try:
        conn = sqlite3.connect("data/server_face_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM person")
        rows = cursor.fetchall()
        conn.close()

        persons = []
        for row in rows:
            image_path = row[2].replace("\\", "/")
            persons.append({
                "id": row[0],
                "name": row[1],
                "image": image_path,
                "modified_time": row[4]
            })

        return {"persons": persons}

    except Exception as e:
        logger.error(f"An error occurred while fetching persons: {e}")
        return JSONResponse(status_code=500, content={"error": f"An error occurred while fetching persons: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)