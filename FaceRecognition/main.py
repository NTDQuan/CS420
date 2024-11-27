import torch
import os
import cv2
import time
import argparse
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np

from data import cfg_mnet, cfg_re50
from module.FaceDetection import FaceDetection
from module.FaceRecognition import FaceRecognition
from module import postprocess
from config import get_config
from utils.utils import prepare_facebank, load_facebank

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--dataset', default='FDDB', type=str, choices=['FDDB'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.5, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
args = parser.parse_args()

conf = get_config()
print(conf)

face_recognition = FaceRecognition(conf, True)
face_recognition.load_state(conf, 'model_mobilefacenet.pth', True)
print('Face Recognition loaded')

if args.update:
    targets, names = prepare_facebank(conf, face_recognition.model, tta = args.tta)
    print('facebank updated')
else:
    targets, names = load_facebank(conf)
    print('facebank loaded')


class App:
    def __init__(self, face_detector, face_recognition, targets, names):
        self.face_detector = face_detector
        self.face_recognition = face_recognition
        self.targets = targets
        self.names = names
        self.cap = cv2.VideoCapture(0)

    def run(self):
        if not self.cap.isOpened():
            print("Cannot open camera")
            return
        print("Starting camera")
        while True:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            img_raw, detections, landmarks = self.face_detector.detect_faces(frame)
            self._draw_detections(img_raw, detections, landmarks)

            fps = 1 / (time.time() - start_time)
            cv2.putText(img_raw, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Detection', img_raw)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                self._perform_face_recognition(frame, detections, landmarks)

        self.cap.release()
        cv2.destroyAllWindows()

    def _perform_face_recognition(self, frame, detections, landmarks):
        """Perform face recognition."""
        faces = self.face_detector.extract_face(frame, detections, landmarks, align_faces=True)
        print(faces)
        recognized_faces = []

        for face in faces:
            if isinstance(face, torch.Tensor):
                face = face.cpu().numpy()  # Convert to NumPy array if it's a tensor
            if isinstance(face, np.ndarray):
                face = Image.fromarray(np.uint8(face))  # Convert NumPy array to PIL Image
            # Preprocess and infer
            idx, dist = self.face_recognition.infer(conf, [face], self.targets, tta=False)
            if idx[0] != -1:
                name = self.names[idx[0] + 1]  # Adjust indexing to match facebank
                recognized_faces.append(name)
            else:
                recognized_faces.append("Unknown")

        print(f"Recognized Faces: {recognized_faces}")
        print(dist)

    @staticmethod
    def _draw_detections(img, detections, landmarks):
        for i, det in enumerate(detections):
            confidence = det[4]
            cv2.rectangle(img, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 0, 255), 2)
            if landmarks is not None and i < len(landmarks):
                for j in range(5):
                    cv2.circle(img, (int(landmarks[i][j * 2]), int(landmarks[i][j * 2 + 1])), 2, (0, 255, 0), 2)

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == 'mobile0.25':
        cfg = cfg_mnet
    elif args.network == 'resnet50':
        cfg = cfg_re50

    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else 'cuda')

    face_detector = FaceDetection(device, args.confidence_threshold, True, cfg, args.nms_threshold, args.trained_model)

    # Initialize face recognition and load facebank
    face_recognition = FaceRecognition(conf, True)
    face_recognition.load_state(conf, 'model_mobilefacenet.pth', True)
    print('Face Recognition loaded')

    targets, names = load_facebank(conf)
    print('Facebank loaded')

    app = App(face_detector, face_recognition, targets, names)
    app.run()


