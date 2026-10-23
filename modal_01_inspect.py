"""Step 1: download IndicTTS Hindi (IIT Madras recordings, HF copy by SPRINGLab, IIT Madras)
into a Modal volume and report what is actually in it. Nothing is trained here.

run (from your laptop):  modal run modal_01_inspect.py
prints: license text from the dataset card, utterances and hours per gender, sample rates,
        duration spread, how many transcripts contain Latin-script (English) words, 5 examples.
also saves 2 sample wavs per gender to the volume so you can listen:
        modal volume get hinglish-tts samples ./samples
"""
import modal

app = modal.App("hinglish-tts-inspect")
vol = modal.Volume.from_name("hinglish-tts", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub>=0.34,<1.0", "pyarrow", "soundfile", "numpy")

REPO = "SPRINGLab/IndicTTS-Hindi"
RAW = "/vol/raw/indictts_hindi"


@app.function(image=image, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("huggingface")],
              cpu=4, memory=8192, timeout=3600)
def inspect():
    import collections, glob, io, json, os, re
    import numpy as np, pyarrow.parquet as pq, soundfile as sf
    from huggingface_hub import snapshot_download

    snapshot_download(REPO, repo_type="dataset", local_dir=RAW,
                      allow_patterns=["*.parquet", "README.md", "*.txt", "LICENSE*"])
    vol.commit()
    card = open(f"{RAW}/README.md", encoding="utf-8").read() if os.path.exists(f"{RAW}/README.md") else ""
    files = sorted(glob.glob(f"{RAW}/**/*.parquet", recursive=True))

    gender_names = None
    dur = collections.defaultdict(list)
    srs, chans = collections.Counter(), collections.Counter()
    latin_utts, n, examples, saved = 0, 0, [], collections.Counter()
    os.makedirs("/vol/samples", exist_ok=True)

    for f in files:
        t = pq.read_table(f)
        if gender_names is None:
            try:
                meta = json.loads(t.schema.metadata[b"huggingface"])
                gender_names = meta["info"]["features"]["gender"]["names"]
            except Exception:
                gender_names = []
        for row in t.to_pylist():
            a = row["audio"]
            info = sf.info(io.BytesIO(a["bytes"]))
            g = row.get("gender")
            g = gender_names[g] if isinstance(g, int) and g < len(gender_names) else str(g)
            dur[g].append(info.frames / info.samplerate)
            srs[info.samplerate] += 1
            chans[info.channels] += 1
            text = row["text"]
            if re.search(r"[A-Za-z]{2,}", text):
                latin_utts += 1
            if len(examples) < 5:
                examples.append({"gender": g, "sec": round(info.frames / info.samplerate, 2), "text": text})
            if saved[g] < 2:
                with open(f"/vol/samples/hindi_{g}_{saved[g]}.wav", "wb") as w:
                    w.write(a["bytes"])
                saved[g] += 1
            n += 1

    stats = {
        "repo": REPO, "parquet_files": len(files), "utterances": n,
        "hours_total": round(sum(map(sum, dur.values())) / 3600, 2),
        "per_gender": {g: {"utts": len(d), "hours": round(sum(d) / 3600, 2),
                           "sec_p5_p50_p95": [round(float(x), 1) for x in np.percentile(d, [5, 50, 95])],
                           "over_30s": int(sum(x > 30 for x in d))} for g, d in dur.items()},
        "sample_rates": dict(srs), "channels": dict(chans),
        "utts_with_latin_words": latin_utts, "examples": examples,
    }
    os.makedirs("/vol/stats", exist_ok=True)
    json.dump(stats, open("/vol/stats/indictts_hindi.json", "w"), indent=2, ensure_ascii=False)
    vol.commit()
    return card[:3000], stats


@app.local_entrypoint()
def main():
    import json
    card, stats = inspect.remote()
    print("=" * 30, "DATASET CARD (first 3000 chars)", "=" * 30)
    print(card or "(no README.md in the repo)")
    print("=" * 30, "STATS", "=" * 30)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
