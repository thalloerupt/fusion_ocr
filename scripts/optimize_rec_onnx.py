#!/usr/bin/env python3
"""为 PP-OCRv6 rec 模型追加图内 ArgMax/ReduceMax，避免回传完整 6906 类概率张量。"""
from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "models/PP-OCRv6_tiny_rec.onnx"
DST = ROOT / "models/PP-OCRv6_tiny_rec_compact.onnx"


def main():
    model = onnx.load(SRC)
    logits = model.graph.output[0].name
    model.graph.node.extend(
        [
            helper.make_node(
                "ArgMax",
                [logits],
                ["token_ids"],
                axis=2,
                keepdims=0,
                name="CTCArgMax",
            ),
            helper.make_node(
                "ReduceMax",
                [logits],
                ["token_probs"],
                axes=[2],
                keepdims=0,
                name="CTCMaxProb",
            ),
        ]
    )
    del model.graph.output[:]
    model.graph.output.extend(
        [
            helper.make_tensor_value_info("token_ids", TensorProto.INT64, ["batch", "steps"]),
            helper.make_tensor_value_info("token_probs", TensorProto.FLOAT, ["batch", "steps"]),
        ]
    )
    onnx.checker.check_model(model)
    onnx.save(model, DST)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
