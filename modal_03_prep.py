"""Step 3: turn the three raw datasets into one clean, audited training corpus.

For every utterance:
  audio  48 kHz (IndicTTS, IndicVoices-R) or 16 kHz (Svarah)  ->  24 kHz mono FLAC (Mimi's rate),
         soxr VHQ resampling, RMS normalized to -20 dBFS with a -1 dBFS peak cap
  text   NFC, danda -> ".", whitespace collapsed, terminal punctuation ensured (Pocket TTS
         appends "." to every prompt at inference, so training text should end the same way)
  drop   duration outside [1, 30] s, digits (the model cannot verbalize them reliably and
         we will not guess number readings), characters outside Devanagari/Latin/basic
         punctuation, Svarah fragments under 3 words. Every drop is counted, nothing silent.
  split  validation is speaker-disjoint for IndicVoices-R and Svarah; IndicTTS has only 2
         speakers, so 1.5% of each speaker's utterances are held out instead.

IndicVoices-R ships two human transcripts: "verbatim" (as pronounced, e.g. बौहौत) and
"normalized" (standard spelling, बहुत). We train on normalized text, because that is what a
user will type, and force-align on verbatim when both have the same word count, because
verbatim matches the audio more closely. Word timings transfer one-to-one.

Writes to the volume:
  /vol/audio/<dataset>/<id>.flac
  /vol/manifests/all.jsonl                      every kept utterance + split + metadata
  /vol/stats/prep.json                          kept/dropped counts and hours per dataset/split
  /vol/stats/deva_words.tsv                     Devanagari word frequencies (text only), used to
                                                find English loanwords written in Devanagari

run:   modal run modal_03_prep.py
then:  modal volume get hinglish-tts stats/deva_words.tsv . --force      (send me this file)
"""
import modal

app = modal.App("hinglish-tts-prep")
vol = modal.Volume.from_name("hinglish-tts", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pyarrow", "soundfile", "numpy", "soxr")

TARGET_SR = 24000
DATASETS = {
    "indictts_hi": "/vol/raw/indictts_hindi",
    "ivr_hi": "/vol/raw/indicvoices_r_hindi",
    "svarah_en": "/vol/raw/svarah",
}


def clean_text(t: str) -> str:
    import re, unicodedata
    t = unicodedata.normalize("NFC", t or "")
    t = t.replace("।", ".").replace("॥", ".").replace("‌", "").replace("‍", "")
    t = re.sub(r"\s+([,.?!;:])", r"\1", t)          # "Fruits ?" -> "Fruits?"
    t = " ".join(t.split()).strip()
    if t and t[-1] not in ".?!":
        t = t.rstrip(",;:- ") + "."
    return t


def bad_chars(t: str) -> set:
    ok = lambda c: ("ऀ" <= c <= "ॿ") or c.isascii() and (c.isalpha() or c in " ,.?!'-")
    return {c for c in t if not ok(c)}


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=8192, timeout=3600)
def process_file(ds: str, parquet: str):
    import hashlib, io, os, re
    import numpy as np, pyarrow.parquet as pq, soundfile as sf, soxr

    out_dir = f"/vol/audio/{ds}"
    os.makedirs(out_dir, exist_ok=True)
    rows, drops = [], {}

    def drop(reason):
        drops[reason] = drops.get(reason, 0) + 1

    t = pq.read_table(parquet)
    gender_names = ["female", "male"]  # IndicTTS class labels, from its dataset card
    for i, r in enumerate(t.to_pylist()):
        a = r.get("audio") or r.get("audio_filepath")
        if not isinstance(a, dict) or not a.get("bytes"):
            drop("no_audio_bytes"); continue

        if ds == "indictts_hi":
            text = clean_text(r["text"]); align = text
            g = r["gender"]; gender = gender_names[g] if isinstance(g, int) else str(g).lower()
            speaker = f"indictts_{gender}"
            meta = {}
        elif ds == "ivr_hi":
            text = clean_text(r.get("normalized") or r["text"])
            verb = clean_text(r.get("verbatim") or "")
            same = verb and len(verb.split()) == len(text.split())
            align = verb if same else text
            gender, speaker = str(r["gender"]).lower(), f"ivr_{r['speaker_id']}"
            meta = {"align_src": "verbatim" if same else "normalized", "scenario": r.get("scenario"),
                    "state": r.get("state"), "snr": r.get("snr")}
        else:  # svarah_en
            text = clean_text(r["text"]); align = text
            gender = str(r["gender"]).lower()
            key = "|".join(str(r.get(k)) for k in ["primary_language", "native_place_state",
                  "native_place_district", "gender", "age-group", "highest_qualification",
                  "job_category", "occupation_domain"])
            speaker = "svarah_" + hashlib.md5(key.encode()).hexdigest()[:10]
            meta = {"l1": r.get("primary_language"), "state": r.get("native_place_state"),
                    "src_path": a.get("path")}

        if not text:
            drop("empty_text"); continue
        if re.search(r"\d", text):
            drop("digits"); continue
        if bad_chars(text):
            drop("other_chars"); continue
        if ds == "svarah_en" and len(text.split()) < 3:
            drop("under_3_words"); continue

        wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        dur = len(wav) / sr
        if dur < 1.0:
            drop("under_1s"); continue
        if dur > 30.0:
            drop("over_30s"); continue
        if sr != TARGET_SR:
            wav = soxr.resample(wav, sr, TARGET_SR, quality="VHQ")
        rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-9)
        if rms < 1e-4:
            drop("silent"); continue
        wav = wav * (10 ** (-20 / 20) / rms)
        peak = float(np.abs(wav).max())
        if peak > 0.89:                       # -1 dBFS
            wav = wav * (0.89 / peak)

        uid = f"{ds}_{os.path.basename(parquet).split('.')[0]}_{i:05d}"
        path = f"{out_dir}/{uid}.flac"
        sf.write(path, wav, TARGET_SR, format="FLAC", subtype="PCM_16")
        rows.append({"id": uid, "dataset": ds, "path": path, "duration": round(len(wav) / TARGET_SR, 3),
                     "transcript": text, "align_text": align, "speaker": speaker, "gender": gender,
                     "orig_sr": sr, **meta})
    vol.commit()
    return ds, rows, drops


@app.local_entrypoint()
def main():
    import json
    files = []
    # list parquet files through a tiny remote call (the volume is not mounted locally)
    for ds, raw in DATASETS.items():
        for p in list_parquet.remote(raw):
            files.append((ds, p))
    print(f"{len(files)} parquet files")
    results = list(process_file.starmap(files, order_outputs=False))
    summary = finalize.remote(results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


@app.function(image=image, volumes={"/vol": vol})
def list_parquet(raw: str):
    import glob
    return sorted(glob.glob(f"{raw}/**/*.parquet", recursive=True))


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=8192, timeout=1800)
def finalize(results):
    import collections, json, os, random, re

    rows = [r for _, rs, _ in results for r in rs]
    drops = collections.defaultdict(collections.Counter)
    for ds, _, d in results:
        drops[ds].update(d)

    # ---- speaker-disjoint / held-out validation split
    rng = random.Random(0)
    by_spk = collections.defaultdict(list)
    for r in rows:
        by_spk[(r["dataset"], r["speaker"])].append(r)
    val_spk = set()
    for ds, n_per_gender in [("ivr_hi", 4), ("svarah_en", 4)]:
        for g in ["female", "male"]:
            cands = sorted(s for (d, s), rs in by_spk.items()
                           if d == ds and rs[0]["gender"] == g and 20 <= len(rs) <= 200)
            rng.shuffle(cands)
            val_spk.update(cands[:n_per_gender])
    for r in rows:
        if r["dataset"] == "indictts_hi":
            r["split"] = "val" if rng.random() < 0.015 else "train"
        else:
            r["split"] = "val" if r["speaker"] in val_spk else "train"

    os.makedirs("/vol/manifests", exist_ok=True)
    with open("/vol/manifests/all.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Devanagari word frequencies (text only) for loanword detection
    freq = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["dataset"] == "svarah_en":
            continue
        for w in re.findall(r"[ऀ-ॿ]+", r["transcript"]):
            freq[w][r["dataset"]] += 1
    os.makedirs("/vol/stats", exist_ok=True)
    with open("/vol/stats/deva_words.tsv", "w", encoding="utf-8") as f:
        f.write("word\ttotal\tivr_hi\tindictts_hi\n")
        for w, c in sorted(freq.items(), key=lambda kv: -sum(kv[1].values())):
            f.write(f"{w}\t{sum(c.values())}\t{c['ivr_hi']}\t{c['indictts_hi']}\n")

    summ = {}
    for ds in DATASETS:
        for split in ["train", "val"]:
            rs = [r for r in rows if r["dataset"] == ds and r["split"] == split]
            summ[f"{ds}/{split}"] = {"utts": len(rs), "hours": round(sum(r["duration"] for r in rs) / 3600, 2),
                                     "speakers": len({r["speaker"] for r in rs})}
    ivr = [r for r in rows if r["dataset"] == "ivr_hi"]
    out = {"kept": summ, "dropped": {k: dict(v) for k, v in drops.items()},
           "total_hours": round(sum(r["duration"] for r in rows) / 3600, 2),
           "ivr_align_on_verbatim": round(sum(r.get("align_src") == "verbatim" for r in ivr) / max(1, len(ivr)), 3),
           "deva_word_types": len(freq)}
    json.dump(out, open("/vol/stats/prep.json", "w"), indent=2, ensure_ascii=False)
    vol.commit()
    return out
