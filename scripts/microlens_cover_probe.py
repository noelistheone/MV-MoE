"""Probe MicroLens-100k raw covers WITHOUT downloading the whole archive.

Reads the remote zip's central directory with HTTP Range requests, lists every member, maps our
MMRec item set (Recsys/data/microlens/i_id_mapping.csv, column 'asin' = raw video id) to cover
files, then range-fetches a handful of covers and checks they decode. Writes
results/phase_shortvideo2/microlens_cover_probe.json and saves the sampled covers under
data/microlens_raw/cover_samples/.
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/MicroLens-100k_covers.zip"
ROOT = Path("/workspace/MechInterp")
OUT = ROOT / "results" / "phase_shortvideo2" / "microlens_cover_probe.json"
UA = {"User-Agent": "curl/8.5.0"}  # server returns 403 to the default Python-urllib agent
SAMPLES = ROOT / "data" / "microlens_raw" / "cover_samples"


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url, self.pos, self.bytes_fetched, self.requests = url, 0, 0, 0
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}", **UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.requests += 1
        self.bytes_fetched += len(data)
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def main():
    t0 = time.time()
    f = HTTPRangeFile(URL)
    zf = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16))
    infos = [i for i in zf.infolist() if not i.is_dir()]
    cd_bytes = f.bytes_fetched
    names = {Path(i.filename).stem: i for i in infos}
    im = pd.read_csv("/workspace/Recsys/data/microlens/i_id_mapping.csv", sep="\t")
    ours = [str(a) for a in im.asin.tolist()]
    have = [a for a in ours if a in names]
    our_bytes = sum(names[a].compress_size for a in have)
    # sample: first 3 item ids, last 2, and 3 spread in the middle (deterministic)
    idx = [0, 1, 2, len(ours) // 3, len(ours) // 2, 2 * len(ours) // 3, len(ours) - 2, len(ours) - 1]
    from PIL import Image
    SAMPLES.mkdir(parents=True, exist_ok=True)
    samples = []
    for k in idx:
        a = ours[k]
        rec = {"itemID": int(k), "raw_video_id": a, "in_zip": a in names}
        if a in names:
            data = zf.read(names[a])
            (SAMPLES / Path(names[a].filename).name).write_bytes(data)
            img = Image.open(io.BytesIO(data)); img.load()
            rec.update(member=names[a].filename, bytes=len(data), size_wh=list(img.size), mode=img.mode)
        samples.append(rec)
    exts = pd.Series([Path(i.filename).suffix.lower() for i in infos]).value_counts().to_dict()
    res = {"url": URL, "archive_bytes": f.size, "n_members": len(infos), "member_ext_counts": exts,
           "example_members": [i.filename for i in infos[:5]],
           "our_items": len(ours), "our_items_with_cover": len(have),
           "our_items_missing_cover": [a for a in ours if a not in names][:50],
           "our_covers_compressed_bytes": our_bytes,
           "our_covers_uncompressed_bytes": sum(names[a].file_size for a in have),
           "central_directory_bytes_fetched": cd_bytes, "total_bytes_fetched": f.bytes_fetched,
           "http_range_requests": f.requests, "samples": samples, "wall_s": time.time() - t0,
           "accessed": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "our_items_missing_cover"}, indent=1))


if __name__ == "__main__":
    main()
