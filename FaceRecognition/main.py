import torch
import numpy as np
import cv2
import time
import argparse
import torch.backends.cudnn as cudnn
from models.retinaface import RetinaFace
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset', default='FDDB', type=str, choices=['FDDB'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.5, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = model_keys - ckpt_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing key:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used key:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def detect_faces(net, img, device, confidence_threshold):
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
    img = img.to(device)
    scale = scale.to(device)
    scale_landms = scale_landms.to(device)

    loc, conf, landms = net(img)  # forward pass
    priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])

    # Check shapes of landms and boxes
    print(f'Boxes shape: {boxes.shape}, Scores shape: {scores.shape}, Landmarks shape: {landms.shape}')

    landms = landms * scale_landms
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4)
    dets = dets[keep, :]
    landms = landms[keep]

    return img_raw, dets, landms

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == 'mobile0.25':
        cfg = cfg_mnet
    elif args.network == 'resnet50':
        cfg = cfg_re50
    
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else 'cuda')
    net = net.to(device)

    cap = cv2.VideoCapture(0)

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        img_raw, dets, landms = detect_faces(net, frame, device, args.confidence_threshold)

        for i, b in enumerate(dets):
            confidence_score = b[4]
            if confidence_score < args.confidence_threshold:
                continue
            print(f'Face {i+1} confidence score: {confidence_score:.2f}')
            cv2.rectangle(img_raw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
            
            if landms is not None and i < len(landms):
                lm = landms[i]
                for j in range(5):
                    cv2.circle(img_raw, (int(lm[j * 2]), int(lm[j * 2 + 1])), 2, (0, 255, 0), 2)

        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)  # Calculate FPS
        print(f"FPS: {fps:.2f}")

        # Display FPS on the frame
        cv2.putText(img_raw, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Real-time Face Detection', img_raw)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


