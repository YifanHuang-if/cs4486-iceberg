"""Prepare caches with source and cache fingerprint checks; raw data stays unchanged."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from src.v2.cache import scan
ROOT=Path(__file__).resolve().parents[2]
def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""): h.update(block)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw",type=Path,default=ROOT/"data/raw")
    args=parser.parse_args()
    out=ROOT/"work/cache";out.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/"configs/validation.json").read_text())
    source_hashes={s:sha256(args.raw/f"{s}.json") for s in ["train","test"]}
    if source_hashes["train"]!=config["train_file_sha256"]:
        raise ValueError("Training JSON differs from the frozen validation source")
    manifest=out/"v3_cache_manifest.json"
    previous=json.loads(manifest.read_text()) if manifest.exists() else {}
    report={"source_sha256":source_hashes,"splits":{}}
    rebuilt=False
    for split,labelled,expected in [("train",True,1604),("test",False,8424)]:
        target=out/f"{split}.npz"
        expected_hash=previous.get("splits",{}).get(split,{}).get("cache_sha256")
        reusable=(target.exists() and previous.get("source_sha256")==source_hashes
                  and expected_hash and sha256(target)==expected_hash)
        if not reusable:
            values=scan(args.raw/f"{split}.json",labelled)
            temp=out/f"{split}.v3tmp.npz"
            np.savez(temp,**{k:v for k,v in values.items() if v is not None})
            temp.replace(target);rebuilt=True
        with np.load(target,allow_pickle=False) as cache:
            if cache["images"].shape!=(expected,2,75,75): raise ValueError("Invalid image cache")
            if len(set(cache["ids"].tolist()))!=expected: raise ValueError("Invalid cached IDs")
            report["splits"][split]={"rows":expected,"cache_sha256":sha256(target)}
    if rebuilt:
        for name in ["v3_image_stats.npz","v3_shape_features.npz"]:
            (out/name).unlink(missing_ok=True)
    manifest.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
