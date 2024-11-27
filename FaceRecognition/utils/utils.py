from PIL import Image
import numpy as np
import torch
from torchvision import transforms as trans
from models.FaceRecognition import Backbone, Arcface, MobileFaceNet, l2_norm

def prepare_facebank(conf, model, tta=True):
    model.eval()
    embeddings = []
    names = ['Unknown']

    print("Starting the face bank preparation...")
    for path in conf.facebank_path.iterdir():
        if path.is_file():
            continue
        else:
            embs = []
            print(f"Processing folder: {path.name}")

            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    try:
                        img = Image.open(file)
                    except Exception as e:
                        print(f"Error opening image {file}: {e}")
                        continue

                    print(f"Processing image: {file.name}")
                    img = img.resize((112, 112))
                    with torch.no_grad():
                        if tta:
                            print(f"Original image size: {img.size}")
                            mirror = trans.functional.hflip(img)
                            emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:
                            print(f"Original image size: {img.size}")
                            print(f"New image size: {conf.test_transform(img).to(conf.device).unsqueeze(0).size()}")
                            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        if len(embs) == 0:
            print(f"No embeddings found for folder: {path.name}")
            continue
        embedding = torch.cat(embs).mean(0, keepdim=True)
        print(f"Folder embedding shape: {embedding.shape}")

        embeddings.append(embedding)
        names.append(path.name)

    print(f"Total embeddings shape: {torch.cat(embeddings).shape}")
    print(f"Names: {names}")

    embeddings = torch.cat(embeddings)
    names = np.array(names)

    print(f"Saving embeddings to {conf.facebank_path / 'facebank.pth'}")
    torch.save(embeddings, conf.facebank_path/'facebank.pth')

    print(f"Saving names to {conf.facebank_path / 'names.npy'}")
    np.save(conf.facebank_path/'names', names)

    return embeddings, names

def load_facebank(conf):
    print(f"Loading face bank from {conf.facebank_path / 'facebank.pth'}")
    embeddings = torch.load(conf.facebank_path/'facebank.pth')
    names = np.load(conf.facebank_path/'names.npy')

    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Loaded names: {names}")
    return embeddings, names


