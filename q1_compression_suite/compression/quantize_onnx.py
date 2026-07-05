import os
import argparse
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model(input_model_path: str, output_model_path: str):
    """
    Quantizes an ONNX model from Float32 to Int8 precision locally on CPU
    using ONNX Runtime Quantization Tools. This represents the local on-device
    quantization pipeline for host-side validation.
    """
    if not os.path.exists(input_model_path):
        raise FileNotFoundError(f"Input model not found at {input_model_path}")
        
    print(f"=== Quantizing ONNX model: {input_model_path} ===")
    print(f"Original size: {os.path.getsize(input_model_path) / 1024**2:.2f} MB")
    
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=QuantType.QUInt8
    )
    
    print("Quantization complete!")
    print(f"Quantized size: {os.path.getsize(output_model_path) / 1024**2:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8 dynamic weights.")
    parser.add_argument("--input", type=str, required=True, help="Path to input FP32 ONNX model")
    parser.add_argument("--output", type=str, required=True, help="Path to output INT8 ONNX model")
    args = parser.parse_args()
    
    quantize_onnx_model(args.input, args.output)
