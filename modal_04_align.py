"""Step 4: word-level forced alignment (Pocket TTS refuses to train without word timings).

Uses Kyutai's own aligner (training/scripts/align_data.py, CTC Viterbi over a wav2vec2 model),
run unmodified from the pinned pocket-tts commit:
  Hindi (IndicTTS + IndicVoices-R):  Harveenchadha/vakyansh-wav2vec2-hindi-him-4200
                                      (MIT, trained on 4200 h of Hindi, Devanagari characters)
  English (Svarah):                   facebook/wav2vec2-large-960h-lv60-self
The text fed to the aligner is "align_text" (verbatim spelling where available); afterwards the
timings are copied one-to-one onto the training transcript's words (same word count, checked).

Then a sanity audit, and utterances that look misaligned are flagged, not silently kept:
  - aligner failed, or more than 20% of the words got no timing
  - a word longer than 3 s or a word shorter than 15 ms (below one CTC frame)
  - first word starts after 3 s, or last word ends more than 3 s before the end of the audio
Hindi is split into 6 shards on 6 GPUs in parallel (L4, or A10G/L40S/T4 if L4s are busy; ~20-40 min, ~$2).

run:  modal run modal_04_align.py
"""
import modal

app = modal.App("hinglish-tts-align")
vol = modal.Volume.from_name("hinglish-tts", create_if_missing=True)
POCKET_COMMIT = "a922b912dfa366975294afb568d815b305acde5d"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("torch==2.6.0", "torchaudio==2.6.0", "transformers>=4.40,<5", "sphn>=0.1.4",
                 "tqdm", "soundfile", "numpy")
    .run_commands(f"git clone https://github.com/kyutai-labs/pocket-tts /opt/pocket-tts && "
                  f"cd /opt/pocket-tts && git checkout {POCKET_COMMIT} && pip install -e .")
)
MODELS = {"hi": "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200",
          "en": "facebook/wav2vec2-large-960h-lv60-self"}
N_HI_SHARDS = 6


@app.function(image=image, volumes={"/vol": vol}, gpu=["L4", "A10G", "L40S", "T4"], timeout=4 * 3600,
              cpu=4.0, memory=16384, secrets=[modal.Secret.from_name("huggingface")])
def align_shard(lang: str, idx: int, n: int):
    """Output goes straight to the volume with --resume and is committed every minute, so an
    interrupted shard continues where it stopped instead of starting over."""
    import json, os, subprocess, threading, time
    rows = [json.loads(l) for l in open("/vol/manifests/all.jsonl", encoding="utf-8")]
    rows = [r for r in rows if (r["dataset"] == "svarah_en") == (lang == "en")]
    rows = rows[idx::n]
    inp = f"/tmp/{lang}_{idx}_in.jsonl"
    out = f"/vol/align/{lang}_{idx}.jsonl"
    os.makedirs("/vol/align", exist_ok=True)
    with open(inp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r | {"transcript": r["align_text"], "final_text": r["transcript"]},
                               ensure_ascii=False) + "\n")
    stop = threading.Event()
    def committer():
        while not stop.wait(60):
            vol.commit()
    threading.Thread(target=committer, daemon=True).start()
    t0 = time.time()
    env = os.environ | {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    p = subprocess.run(["python", "-m", "training.scripts.align_data", inp, out, "--resume",
                        "--model", MODELS[lang], "--device", "cuda", "--batch-size", "32",
                        "--shard", str(idx)], cwd="/opt/pocket-tts", env=env)
    stop.set()
    vol.commit()
    n_out = sum(1 for _ in open(out, encoding="utf-8")) if os.path.exists(out) else 0
    return {"lang": lang, "shard": idx, "in": len(rows), "aligned": n_out, "returncode": p.returncode,
            "minutes": round((time.time() - t0) / 60, 1), "log_tail": "see the Modal app logs"}


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=8192, timeout=1800)
def merge_and_audit():
    import collections, glob, json
    out, stats = [], collections.defaultdict(collections.Counter)
    word_durs = collections.defaultdict(list)
    for f in sorted(glob.glob("/vol/align/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            ds = r["dataset"]
            final_words = r["final_text"].split()
            words = r["words"]
            st = stats[ds]
            st["aligned"] += 1
            if len(final_words) != len(words):
                st["drop_wordcount_mismatch"] += 1; continue
            timed = [w for w in words if w["start"] is not None]
            reasons = []
            if len(timed) < 0.8 * len(words):
                reasons.append("over_20pct_untimed")
            if any(w["end"] - w["start"] > 3.0 for w in timed):
                reasons.append("word_over_3s")
            if any(w["end"] - w["start"] < 0.015 for w in timed):
                reasons.append("word_under_20ms")
            if timed and timed[0]["start"] > 3.0:
                reasons.append("late_first_word")
            if timed and r["duration"] - timed[-1]["end"] > 3.0:
                reasons.append("early_last_word")
            for x in reasons:
                st["flag_" + x] += 1
            if reasons:
                st["dropped"] += 1; continue
            for w, fw in zip(words, final_words):
                w["align_word"] = w["word"]; w["word"] = fw
                if w["start"] is not None:
                    word_durs[ds].append(w["end"] - w["start"])
            st["untimed_words"] += len(words) - len(timed)
            st["words"] += len(words)
            r["transcript"] = r.pop("final_text")
            out.append(r)
            st["kept"] += 1
            st["kept_sec"] += r["duration"]
    by_split = collections.Counter()
    for r in out:
        by_split[f'{r["dataset"]}/{r["split"]}'] += r["duration"] / 3600
    with open("/vol/manifests/aligned.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import statistics
    res = {"per_dataset": {ds: dict(c) | {"kept_hours": round(c["kept_sec"] / 3600, 2),
                                         "median_word_sec": round(statistics.median(word_durs[ds]), 3) if word_durs[ds] else None}
                           for ds, c in stats.items()},
           "hours_by_split": {k: round(v, 2) for k, v in sorted(by_split.items())},
           "example": {k: out[0][k] for k in ["id", "transcript", "words"]} if out else None}
    for d in res["per_dataset"].values():
        d.pop("kept_sec", None)
    json.dump(res, open("/vol/stats/align.json", "w"), indent=2, ensure_ascii=False)
    vol.commit()
    return res


@app.local_entrypoint()
def main():
    import json
    jobs = [("hi", i, N_HI_SHARDS) for i in range(N_HI_SHARDS)] + [("en", 0, 1)]
    for r in align_shard.starmap(jobs, order_outputs=False):
        print(f"[{r['lang']} shard {r['shard']}] in={r['in']} aligned={r['aligned']} rc={r['returncode']}")
        if r["returncode"] != 0 or r["aligned"] < 0.5 * r["in"]:
            print(r["log_tail"])
    print(json.dumps(merge_and_audit.remote(), indent=2, ensure_ascii=False))
