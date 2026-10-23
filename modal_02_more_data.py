"""Step 2: download and inspect the other two human-transcribed datasets into the Modal volume.

  IndicVoices-R, Hindi (AI4Bharat, CC BY 4.0): ~75 h, hundreds of speakers, mostly unscripted
      (extempore) Hindi with English words spoken naturally; manually transcribed with QA.
      Original gated repo first; if that fails, the open SPRINGLab mirror of the same data.
  Svarah (AI4Bharat, CC BY 4.0): 9.6 h of English spoken by 117 Indians from 19 states.

Before running, click "Agree and access" on both pages while logged in to Hugging Face:
  https://huggingface.co/datasets/ai4bharat/indicvoices_r
  https://huggingface.co/datasets/ai4bharat/Svarah

run:     modal run modal_02_more_data.py
listen:  modal volume get hinglish-tts samples ./samples --force
"""
import modal

app = modal.App("hinglish-tts-more-data")
vol = modal.Volume.from_name("hinglish-tts", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub>=0.34,<1.0", "pyarrow", "soundfile", "numpy")

SOURCES = {
    "indicvoices_r_hindi": [("ai4bharat/indicvoices_r", ["Hindi/*", "README.md"]),
                            ("SPRINGLab/IndicVoices-R_Hindi", ["*.parquet", "README.md"])],
    "svarah": [("ai4bharat/Svarah", ["*.parquet", "README.md", "*.csv", "*.json"])],
}


@app.function(image=image, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("huggingface")],
              cpu=4, memory=16384, timeout=4 * 3600)
def fetch_and_inspect(name: str):
    import collections, glob, io, json, os, re
    import numpy as np, pyarrow.parquet as pq, soundfile as sf
    from huggingface_hub import snapshot_download

    raw, used, errors = f"/vol/raw/{name}", None, []
    for repo, patterns in SOURCES[name]:
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=raw, allow_patterns=patterns, max_workers=8)
            if glob.glob(f"{raw}/**/*.parquet", recursive=True):
                used = repo
                break
            errors.append(f"{repo}: downloaded but no parquet files")
        except Exception as e:  # gated and not accepted, or not found
            errors.append(f"{repo}: {type(e).__name__}: {str(e)[:300]}")
    vol.commit()
    if used is None:
        return {"name": name, "errors": errors}

    files = sorted(glob.glob(f"{raw}/**/*.parquet", recursive=True))
    schema = pq.read_schema(files[0])
    cols = schema.names
    audio_col = next(c for c in cols if c.lower() in ("audio", "audio_filepath", "wav"))
    text_cols = [c for c in cols if c.lower() in ("text", "verbatim", "normalized", "transcript", "sentence")]
    spk_col = next((c for c in cols if "speaker" in c.lower()), None)
    gen_col = next((c for c in cols if c.lower() == "gender"), None)

    dur, srs, spk, gen = [], collections.Counter(), collections.Counter(), collections.Counter()
    latin = collections.Counter()
    first_row, examples, saved, snr = None, [], 0, []
    os.makedirs("/vol/samples", exist_ok=True)
    for f in files:
        t = pq.read_table(f)
        for row in t.to_pylist():
            a = row[audio_col]
            b = a["bytes"] if isinstance(a, dict) else None
            if b is None:
                continue
            info = sf.info(io.BytesIO(b))
            dur.append(info.frames / info.samplerate)
            srs[info.samplerate] += 1
            if spk_col: spk[str(row[spk_col])] += 1
            if gen_col: gen[str(row[gen_col])] += 1
            if row.get("snr") is not None: snr.append(float(row["snr"]))
            for c in text_cols:
                if re.search(r"[A-Za-z]{2,}", str(row[c] or "")):
                    latin[c] += 1
            if first_row is None:
                first_row = {k: (str(v)[:200] if k != audio_col else "<audio>") for k, v in row.items()}
            if len(examples) < 8 and len(dur) % 500 == 1:
                examples.append({c: row[c] for c in text_cols} | {"sec": round(dur[-1], 2)})
            if saved < 3 and len(dur) % 3000 == 1:
                open(f"/vol/samples/{name}_{saved}.wav", "wb").write(b)
                saved += 1

    stats = {
        "name": name, "repo_used": used, "errors_before": errors, "parquet_files": len(files),
        "columns": cols, "first_row": first_row,
        "utterances": len(dur), "hours": round(sum(dur) / 3600, 2),
        "sec_p5_p50_p95": [round(float(x), 1) for x in np.percentile(dur, [5, 50, 95])],
        "over_30s": int(sum(d > 30 for d in dur)),
        "sample_rates": dict(srs), "speakers": len(spk),
        "utts_per_speaker_p50": int(np.median(list(spk.values()))) if spk else None,
        "gender": dict(gen),
        "snr_p5_p50_p95": [round(float(x), 1) for x in np.percentile(snr, [5, 50, 95])] if snr else None,
        "utts_with_latin_words": dict(latin), "examples": examples,
    }
    os.makedirs("/vol/stats", exist_ok=True)
    json.dump(stats, open(f"/vol/stats/{name}.json", "w"), indent=2, ensure_ascii=False)
    vol.commit()
    return stats


@app.local_entrypoint()
def main():
    import json
    for stats in fetch_and_inspect.map(list(SOURCES)):  # both datasets in parallel
        print("=" * 30, stats["name"], "=" * 30)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
