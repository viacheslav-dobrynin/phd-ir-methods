import os
import json
from typing import Dict, Mapping, Optional
import numpy as np
from dataclasses import dataclass

META_NAME = "emb_store.meta.json"
DATA_NAME = "emb_store.bin"
IDX_NAME = "emb_store.index.jsonl"


@dataclass(frozen=True)
class IndexEntry:
    start: int
    length: int


class EmbsStore(Mapping):
    def __init__(self, base_path: str):
        meta_path = os.path.join(base_path, META_NAME)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.dim: int = int(meta["dim"])
        self.dtype: np.dtype = np.dtype(meta["dtype"])
        self.data_path = os.path.join(base_path, meta["data_path"])
        self.index_path = os.path.join(base_path, meta["index_path"])

        self._mm = np.memmap(self.data_path, dtype=self.dtype, mode="r")
        self._idx: Dict[str, IndexEntry] = {}
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                assert line.strip()
                rec = json.loads(line)
                self._idx[rec["doc_id"]] = IndexEntry(
                    int(rec["start"]), int(rec["length"])
                )

    def get_embs(self, doc_id: str) -> np.ndarray:
        """Returs embs [length, dim] (store's dtype)"""
        entry = self._idx.get(doc_id)
        if not entry:
            raise KeyError(f"Document ID '{doc_id}' not found in the index.")
        return self.__get_embs_by_entry(entry).astype(np.float32)

    def __iter__(self):
        return iter(self._idx)

    def __getitem__(self, doc_id: str):
        return self.get_embs(doc_id)

    def __len__(self) -> int:
        return len(self._idx)

    def __get_embs_by_entry(self, entry: IndexEntry) -> np.ndarray:
        assert entry.length > 0
        left = entry.start * self.dim
        right = (entry.start + entry.length) * self.dim
        return self._mm[left:right].reshape(entry.length, self.dim)


class EmbsStoreBuilder:
    """
    Write-only, streaming store builder.
    Usage:
        b = EmbsStoreBuilder(base_path, dtype=np.float16, overwrite=True)
        b.add(doc_id, embs_np)  # embs: [n, dim] float32/16
        ...
        b.finish()
    After finish, use EmbsStore(base_path) and read.
    """

    def __init__(
        self,
        base_path: str,
        model_id: str,
        dtype: np.dtype = np.float16,
        overwrite: bool = False,
    ):
        os.makedirs(base_path, exist_ok=True)
        self.model_id = model_id
        self.dtype = np.dtype(dtype)
        self.data_path = os.path.join(base_path, DATA_NAME)
        self.meta_path = os.path.join(base_path, META_NAME)
        self.index_path = os.path.join(base_path, IDX_NAME)

        if not overwrite and any(
            os.path.exists(p) for p in (self.data_path, self.meta_path, self.index_path)
        ):
            raise FileExistsError(f"{base_path} already contains an EmbsStore")

        for p in (self.data_path, self.meta_path, self.index_path):
            if os.path.exists(p):
                os.remove(p)

        self._f_data = open(self.data_path, "wb")
        self._f_index = open(self.index_path, "w", encoding="utf-8")

        self.dim: Optional[int] = None
        self.total = 0
        self._cursor = 0
        self._closed = False

    def add(self, doc_id: str, embs) -> None:
        if self._closed:
            raise RuntimeError("Builder already closed")
        if hasattr(embs, "detach"):  # torch.Tensor
            embs = embs.detach().cpu().numpy()
        arr = np.asarray(embs)
        arr = np.squeeze(arr)
        arr = np.atleast_2d(arr)
        assert arr.ndim == 2
        if self.dim is None:
            self.dim = int(arr.shape[1])
        elif int(arr.shape[1]) != self.dim:
            raise ValueError(
                f"dim mismatch for {doc_id}: got {arr.shape[1]}, expected {self.dim}"
            )
        arr = np.ascontiguousarray(arr, dtype=self.dtype)
        start = self._cursor
        length = int(arr.shape[0])
        self._f_data.write(arr.tobytes())
        self._f_index.write(
            json.dumps({"doc_id": str(doc_id), "start": start, "length": length}) + "\n"
        )
        self._cursor += length
        self.total += length

    def close(self) -> None:
        if self._closed:
            return
        self._f_data.flush()
        self._f_data.close()
        self._f_index.flush()
        self._f_index.close()
        meta = {
            "model_id": self.model_id,
            "dtype": self.dtype.name,
            "dim": int(self.dim or 0),
            "total": int(self.total),
            "data_path": DATA_NAME,
            "index_path": IDX_NAME,
            "version": 1,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        self._closed = True
