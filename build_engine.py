"""
ONNX export + TensorRT engine builder for rice panicle detection model.

YOLOv10's default end2end export produces (batch, 300, 6) output which is
incompatible with the standard YOLO postprocessor. This script disables
end2end and uses torch.onnx.export directly to get the raw (batch, 5, 8400)
output format that matches YOLOv8.

Usage:
    # Step 1: Export ONNX from PyTorch
    python build_engine.py --export --weights daosui_v1.pt

    # Step 2: Convert ONNX to TensorRT engine
    python build_engine.py --build --onnx best.onnx --engine daosui_v1.engine
"""

import argparse
import os
import subprocess
import sys


def export_onnx(weights: str, imgsz: int) -> str:
    """Export PyTorch model to ONNX (raw format, no end2end NMS)."""
    import torch
    from ultralytics import YOLO

    model = YOLO(weights)

    # Disable end2end to get raw output (batch, 4+nc, 8400) instead of (batch, 300, 6)
    det = model.model.model[-1]
    det.end2end = False
    m = model.model.eval()

    # Wrap to return only the first (main) output tensor
    class _Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            out = self.model(x)
            return out[0] if isinstance(out, (list, tuple)) else out

    wrapped = _Wrapper(m).eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)

    onnx_path = "daosui_v1.onnx"
    torch.onnx.export(
        wrapped, dummy,
        onnx_path,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={
            "images": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify
    import onnx
    omodel = onnx.load(onnx_path)
    for inp in omodel.graph.input:
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name} {dims}")
    for out in omodel.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} {dims}")

    print(f"ONNX exported: {onnx_path} ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")
    return onnx_path


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    min_batch: int = 1,
    opt_batch: int = 32,
    max_batch: int = 32,
    imgsz: int = 640,
) -> None:
    """Convert ONNX to TensorRT engine using trtexec."""
    trtexec = os.path.join(
        os.path.expanduser("~/Downloads/TensorRT-10.10.0.31/bin/trtexec")
    )

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes=images:{min_batch}x3x{imgsz}x{imgsz}",
        f"--optShapes=images:{opt_batch}x3x{imgsz}x{imgsz}",
        f"--maxShapes=images:{max_batch}x3x{imgsz}x{imgsz}",
    ]

    if fp16:
        cmd.append("--fp16")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Engine saved: {engine_path} ({os.path.getsize(engine_path) / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine for rice panicle detection")
    parser.add_argument("--export", action="store_true", help="Export ONNX from PyTorch weights")
    parser.add_argument("--build", action="store_true", help="Build TensorRT engine from ONNX")
    parser.add_argument("--weights", default="daosui_v1.pt", help="PyTorch weights path")
    parser.add_argument("--onnx", default="daosui_v1.onnx", help="ONNX model path")
    parser.add_argument("--engine", default="daosui_v1.engine", help="Output engine path")
    parser.add_argument("--imgsz", type=int, default=640, help="Model input size")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=32)
    parser.add_argument("--max-batch", type=int, default=32)

    args = parser.parse_args()

    if args.export:
        onnx_path = export_onnx(args.weights, args.imgsz)
        print(f"\nNext step: python build_engine.py --build --onnx {onnx_path} --engine {args.engine}")

    if args.build:
        if not os.path.exists(args.onnx):
            print(f"ONNX file not found: {args.onnx}")
            print("Run with --export first, or provide --onnx path.")
            sys.exit(1)
        build_engine(
            args.onnx, args.engine,
            fp16=args.fp16,
            min_batch=args.min_batch,
            opt_batch=args.opt_batch,
            max_batch=args.max_batch,
            imgsz=args.imgsz,
        )

    if not args.export and not args.build:
        parser.print_help()


if __name__ == "__main__":
    main()
