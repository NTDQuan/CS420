import time

import cv2
import numpy as np
import torch
import os.path as osp
import onnxruntime

from utils.image_util import preprocess_image


class Identifier:
    def __init__(self, model_file = None, session=None):
        self.model_file = model_file
        self.session = session
        self.taskname = "identification"
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            self.session = self._load_model(self.model_file, 'arcface_resnet.trt')

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

    def represent(self, img_raw):
        img = preprocess_image(img_raw)
        input_name = self.session.get_inputs()[0].name

        inputs = {input_name: img.numpy()}

        embeddings = self.session.run(None, inputs)
        return embeddings[0] if embeddings else None


    def identify(self, img_raw, emb_data, name_data):
        query_emb = self.represent(img_raw)
        if query_emb is None:
            raise ValueError("Failed to extract features from query image.")

        if emb_data.size == 0:
            return -1, "Unknown"

        score, best_match = self.compare_encodings(query_emb, emb_data)

        if score < 0.5:
            return -1, "Unknown"

        name = name_data[best_match]

        return score, name

    @staticmethod
    def compare_encodings(encoding, encodings):
        sims = np.dot(encodings, encoding.T)
        pare_index = np.argmax(sims)
        score = sims[pare_index]
        return score, pare_index




