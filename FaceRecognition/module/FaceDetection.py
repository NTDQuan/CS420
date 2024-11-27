import os
import torch
import numpy as np
import cv2
from typing import Union, Any, Optional, Dict, Tuple, List
from layers.functions.prior_box import PriorBox
from data import cfg_mnet, cfg_re50
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from module import postprocess

class FaceDetection:
    def __init__(self, device, confidence_threshold, load_to_cpu, cfg, nms_threshold, pretrained_path):
        self.cfg = cfg
        self.nms_threshold = nms_threshold
        self.pretrained_path = pretrained_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.load_to_cpu = load_to_cpu
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        model = RetinaFace(cfg=self.cfg, phase='test')
        print('Loading pretrained model from {}'.format(self.pretrained_path))
        if self.load_to_cpu:
            pretrained_dict = torch.load(self.pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(self.pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        model.load_state_dict(pretrained_dict, strict=False)
        print('Finished loading model!')
        return model.to(self.device)
    
    def detect_faces(self, img, save_dir="captured_faces", align_faces=False):
        img_raw = img.copy()
        img = np.float32(img)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        scale_landms = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0], 
                                    img.shape[1], img.shape[0], img.shape[1], img.shape[0], 
                                    img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)
        scale_landms = scale_landms.to(self.device)

        loc, conf, landms = self.model(img)  # forward pass
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])

        # Check shapes of landms and boxes
        print(f'Boxes shape: {boxes.shape}, Scores shape: {scores.shape}, Landmarks shape: {landms.shape}')

        landms = landms * scale_landms
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        dets = dets[keep, :]
        landms = landms[keep]

        return img_raw, dets, landms

    @staticmethod
    def extract_face(img_raw, dets, landms, save_dir="captured_faces", align_faces=False,
                     target_size: Optional[Tuple[int, int]] = None, min_max_norm: bool = True):
        """Extract and save detected faces."""
        os.makedirs(save_dir, exist_ok=True)
        extracted_faces = []

        for i, (box, landmarks) in enumerate(zip(dets, landms)):
            x1, y1, x2, y2 = box[:4].astype(int)
            facial_img = img_raw[y1:y2, x1:x2]  # Extract facial image

            if align_faces and len(landmarks) >= 6:  # Ensure landmarks are valid
                left_eye = (landmarks[0], landmarks[1])
                right_eye = (landmarks[2], landmarks[3])
                nose = (landmarks[4], landmarks[5])

                # Align the face using the landmarks
                facial_img, rotate_angle, rotate_direction = postprocess.alignment_procedure(facial_img, left_eye,
                                                                                             right_eye, nose)

                # Rotate the facial area using the alignment info
                facial_area = postprocess.rotate_facial_area((x1, y1, x2, y2), rotate_angle, rotate_direction,
                                                             img_raw.shape[:2])
                x1, y1, x2, y2 = facial_area  # Update facial area coordinates after rotation

            # If a target size is provided, resize the image
            if target_size is not None:
                facial_img = postprocess.resize_image(facial_img, target_size, min_max_norm=min_max_norm)

            # Save the extracted and aligned face image
            save_path = os.path.join(save_dir, f"face_{i}.jpg")
            try:
                extracted_faces.append(facial_img)
                cv2.imwrite(save_path, facial_img)  # Save the image
            except Exception as e:
                print(f"Failed to save face image: {e}")

        print("Faces saved successfully.")

        return extracted_faces








    
