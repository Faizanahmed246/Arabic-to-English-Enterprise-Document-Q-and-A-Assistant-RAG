import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
WEIGHTS = "onnx/model_qint8_avx512_vnni.onnx"
TOKENIZER = "onnx/tokenizer.json"
INDEX_VERSION = "e5-small-int8-614241f-v1"
MAX_TOKENS = 512


def download(cache: Path) -> tuple[str, str]:
    return (
        hf_hub_download(MODEL, TOKENIZER, revision=REVISION, cache_dir=cache),
        hf_hub_download(MODEL, WEIGHTS, revision=REVISION, cache_dir=cache),
    )


class E5:
    def __init__(self, cache: Path):
        tokenizer_path, weights_path = download(cache)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        if self.pad_id is None:
            raise RuntimeError("The embedding tokenizer has no padding token.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.add_session_config_entry("session.disable_prepacking", "1")
        self.session = ort.InferenceSession(
            weights_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_names = [item.name for item in self.session.get_inputs()]

    def offsets(self, text: str) -> list[tuple[int, int]]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return [(start, end) for start, end in encoded.offsets if end > start]

    def encode(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        role = "query" if query else "passage"
        embeddings = []
        for start in range(0, len(texts), 8):
            batch = []
            for text in texts[start : start + 8]:
                ids = self.tokenizer.encode(f"{role}: {text}").ids
                if len(ids) > MAX_TOKENS:
                    ids = ids[: MAX_TOKENS - 1] + ids[-1:]
                batch.append(ids)
            shape = (len(batch), max(map(len, batch)))
            input_ids = np.full(shape, self.pad_id, dtype=np.int64)
            attention_mask = np.zeros(shape, dtype=np.int64)
            for i, ids in enumerate(batch):
                input_ids[i, : len(ids)] = ids
                attention_mask[i, : len(ids)] = 1
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": np.zeros(shape, dtype=np.int64),
            }
            hidden = self.session.run(None, {name: inputs[name] for name in self.input_names})[0]
            mask = inputs["attention_mask"][..., None]
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
            normalized = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
            embeddings.extend(normalized.astype(np.float32).tolist())
        return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the pinned embedding assets.")
    parser.add_argument("cache", type=Path)
    download(parser.parse_args().cache)
