from PIL.ImageOps import mirror
from models.FaceRecognition import Backbone, Arcface, MobileFaceNet, l2_norm
from torchvision import transforms
import torch
import numpy as np
import cv2

class FaceRecognition(object):
    def __init__(self, conf, inference=False):
        self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
        print('MobileFaceNet loaded')

        self.threshold = conf.threshold

    def load_state(self, conf, fixed_str, from_save_folder=False):
        # Determine the save path based on the flag
        save_path = conf.save_path if from_save_folder else conf.model_path

        # Construct the full file path
        full_path = save_path / fixed_str  # Use fixed_str directly without "model_" prefix

        # Debugging step: print the resolved path
        print(f"Loading model from: {full_path}")

        # Check if the file exists
        if not full_path.exists():
            raise FileNotFoundError(f"Model file not found at {full_path}")

        # Load the model's state_dict onto the CPU if CUDA is unavailable
        self.model.load_state_dict(torch.load(full_path, map_location=torch.device('cpu')))


    def infer(self, conf, faces, targets_embs, tta= False):
        self.model.eval()
        embs = []
        for img in faces:
            img = img.resize((112, 112))

            if tta:
                mirror = transforms.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))

        source_embs = torch.cat(embs)

        diff = source_embs.unsqueeze(-1) - targets_embs.transpose(1, 0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1
        return min_idx, minimum