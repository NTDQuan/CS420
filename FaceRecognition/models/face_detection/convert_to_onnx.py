import torch

from data.config import cfg_mnet
from models.face_detection.retinaface.retinaface import RetinaFace


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage
        )
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
        )
    model.load_state_dict(pretrained_dict, strict=False)
    return model

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    cfg = cfg_mnet

    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, "weight/mobilenet0.25_Final.pth", True)
    net.eval()
    print("Finished loading model!")
    print(net)
    device = torch.device("cpu")
    net = net.to(device)

    output_onnx = 'faceDetector.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))

    input_names = ["input"]
    output_names = ["bbox", "confidence", "landmark"]

    dynamic_axes = {
        "input": {0: "batch"},
        "bbox": {0: "batch"},
        "confidence": {0: "batch"},
        "landmark": {0: "batch"}}


    inputs = torch.randn(1, 3, 640, 640).to(device)
    torch_out = torch.onnx.export(net, inputs, output_onnx, export_params=True, verbose=False,
                                   input_names=input_names, output_names=output_names, opset_version=11,
                                   dynamic_axes=dynamic_axes)