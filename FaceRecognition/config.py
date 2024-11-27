from easydict import EasyDict as edict
from pathlib import Path
import torch
from torchvision import transforms

def get_config():
    conf = edict()
    conf.data_path = Path('data')
    conf.model_path = Path('weights')
    conf.input_size = [112, 112]
    conf.save_path = Path('weights')
    conf.embedding_size = 512
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.threshold = 0.7
    conf.facebank_path = conf.data_path/'face_bank'
    conf.test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return conf