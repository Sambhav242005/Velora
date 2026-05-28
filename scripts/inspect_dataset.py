import argparse
import json
from pathlib import Path

import numpy as np

DTYPE = {"uint16": np.uint16, "uint32": np.uint32, "int32": np.int32, "int64": np.int64}

parser = argparse.ArgumentParser()
parser.add_argument("--meta", default="data/processed/meta.json")
args = parser.parse_args()
meta = json.loads(Path(args.meta).read_text())
print(json.dumps(meta, indent=2))
for split in ["train", "val"]:
    path = Path(args.meta).parent / f"{split}.bin"
    arr = np.memmap(path, dtype=DTYPE[meta["dtype"]], mode="r")
    print(split, "tokens:", len(arr), "first 20 ids:", arr[:20].tolist())
