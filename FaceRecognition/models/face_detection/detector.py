import time

import cv2
import numpy as np
import torch
import os.path as osp
import onnxruntime

from layers.functions.prior_box import PriorBox
from models.face_detection.retinaface.retinaface import RetinaFace
from utils.retianaface_utils.box_utils import decode_landm, decode
from utils.retianaface_utils.nms.py_cpu_nms import py_cpu_nms


class RetinaFaceClient:
    def __init__(self, model_file = None, session=None, cfg=None):
        self.model_file = model_file
        self.session = session
        self.taskname = "detection"
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            self.session = self._load_model(self.model_file, 'retinface_mobilenet.trt')
        self.nms_threshold = 0.4
        self.device = 'cpu'
        self.cfg = cfg
        self.confidence_threshold = 0.5
        self.top_k = 1
        self.keep_top_k = 750

    def _load_model(self, model_path, engine_path):
        session =onnxruntime.InferenceSession(
            model_path,
            providers=[
                (
                    'TensorrtExecutionProvider',
                    {
                        'device_id': 0,
                        'trt_max_workspace_size': 2147483648,
                        'trt_fp16_enable': True,
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': '{}'.format(engine_path),
                    }
                ),
                (
                    'CUDAExecutionProvider',
                    {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }
                )
            ]
        )
        return session

    def detect_faces(self, img_raw):
        frame_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img = np.float32(frame_rgb)
        orig_height, orig_width, _ = img.shape

        img_raw = cv2.resize(img_raw, (640, 640))
        img = cv2.resize(img, (640, 640))
        im_height, im_width, _ = img.shape
        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        tic = time.time()
        inputs = {"input": img}
        loc, conf, landms = self.session.run(None, inputs)

        tic = time.time()
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width), format="numpy")
        priors = priorbox.forward()

        prior_data = priors

        boxes = decode(np.squeeze(loc, axis=0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / 1
        scores = np.squeeze(conf, axis=0)[:, 1]

        landms = decode_landm(np.squeeze(landms.data, axis=0), prior_data, self.cfg['variance'])

        scale1 = np.array([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        landms = landms * scale1 / 1

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        return img_raw, dets, landms, orig_width, orig_height






