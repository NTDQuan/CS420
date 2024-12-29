import torch

from models.face_recognition.model import iresnet100


def convert_to_onnx(model_name, weight_path, onnx_path, device="cpu"):
    # Load the PyTorch model
    model = iresnet100()
    weight = torch.load(weight_path, map_location=device)
    model.load_state_dict(weight)
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 112, 112, device=device)

    torch.onnx.export(
        model,                          # PyTorch model
        dummy_input,                    # Dummy input for the model
        onnx_path,                      # Path to save ONNX file
        export_params=True,             # Store trained parameters
        opset_version=11,               # ONNX opset version
        do_constant_folding=True,       # Optimize constant folding
        input_names=['input'],          # Input tensor names
        output_names=['output'],        # Output tensor names
        dynamic_axes={                  # Support dynamic batch size
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model has been converted to ONNX and saved to {onnx_path}")

if __name__ == "__main__":
    model_name = "iresnet100"
    weight_path = "./weight/arcface_r100.pth"
    onnx_path = "../../iresnet100.onnx"
    convert_to_onnx(model_name, weight_path, onnx_path)