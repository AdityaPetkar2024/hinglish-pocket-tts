"""Export, smoke-test and publish the Hinglish Pocket TTS model and its training data.
Runs inside the pinned pocket-tts environment on Modal (see modal_tts.py ::release / ::dataset).

  export    EMA weights of the chosen checkpoint -> model.safetensors (flow_lm + Mimi), tokenizer.json,
            pocket-tts inference configs (one for the local folder, one for hf://)
  smoke     generate test sentences with both inference paths, from local files or from the hub
  upload    model repo (private) on the Hugging Face Hub
  dataset   final training data as parquet shards (audio + transcript + word timings) -> private
            dataset repo, with the preprocessing artefacts (loanword map, stats)
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, "/opt/pocket-tts")
BASE_CFG = "/opt/pocket-tts/pocket_tts/config/english_2026-09_24l.yaml"
SMOKE = [
    "मुझे कल office जाना है, लेकिन मेरी car खराब हो गई है.",
    "आज की meeting में हम new project का budget discuss करेंगे.",
    "I think यह plan काफ़ी अच्छा है, let's start करते हैं.",
]


def write_configs(out: Path, prefix: str, tag: str):
    """pocket-tts inference config pointing at `prefix` (a local folder or hf://repo/)."""
    import yaml
    c = yaml.safe_load(open(BASE_CFG))
    n_bins = len(json.load(open(out / "tokenizer.json", encoding="utf-8"))["model"]["vocab"])
    c["weights_path"] = f"{prefix}model.safetensors"
    c.pop("weights_path_without_voice_cloning", None)
    c["flow_lm"]["lookup_table"].update({"tokenizer": "tokenizers", "tokenizer_path": f"{prefix}tokenizer.json",
                                         "n_bins": n_bins})
    rc = dict(c.get("replace_characters") or {})
    # the training transcripts had dandas turned into periods and zero-width joiners removed
    rc.update({"\u0964": ".", "\u0965": ".", "\u200c": "", "\u200d": ""})
    c["replace_characters"] = rc
    c["default_temperature"] = 0.3
    c["default_eos_threshold"] = -1.0          # the value used in the evaluation
    yaml.safe_dump(c, open(out / f"pocket_tts_hinglish{tag}.yaml", "w"), allow_unicode=True, sort_keys=False)


def cmd_export(a):
    import torch
    from training.eval.librispeech import load_run
    from training.checkpointing import export_pocket_safetensors
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, mimi, step = load_run(a.run_dir, torch.device("cpu"), use_ema=True, checkpoint=a.checkpoint)
    export_pocket_safetensors(out / "model.safetensors", model.flow_lm, mimi)
    shutil.copy(a.tokenizer, out / "tokenizer.json")
    write_configs(out, f"{out}/", "_local")
    write_configs(out, f"hf://{a.repo}/", "")
    json.dump({"source_checkpoint": a.checkpoint, "step": step, "weights": "EMA (decay 0.999)",
               "params_flow_lm": sum(p.numel() for p in model.flow_lm.parameters())},
              open(out / "export_info.json", "w"), indent=2)
    print(json.dumps(json.load(open(out / "export_info.json")), indent=2))


def cmd_smoke(a):
    import soundfile as sf
    sys.path.insert(0, "/root")
    from infer import synthesize
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open("/vol/data/valid.jsonl", encoding="utf-8")]
    voice = next(r["path"] for r in rows if r["dataset"] == "indictts_hi" and 4 <= r["duration"] <= 8)
    report = []
    for cfg in (2.0, 1.0):
        for i, text in enumerate(SMOKE):
            t0 = time.time()
            wav, sr = synthesize(text, voice, cfg=cfg, source=a.source)
            p = out / f"cfg{int(cfg)}_{i}.wav"
            sf.write(p, wav, sr)
            report.append({"cfg": cfg, "text": text, "seconds_audio": round(len(wav) / sr, 2),
                           "seconds_compute": round(time.time() - t0, 2), "file": str(p)})
    shutil.copy(voice, out / ("voice_prompt" + Path(voice).suffix))
    json.dump(report, open(out / "smoke.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_upload(a):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, private=True, exist_ok=True)
    out = Path(a.out)
    for f in ["model.safetensors", "tokenizer.json", "pocket_tts_hinglish.yaml", "export_info.json"]:
        api.upload_file(path_or_fileobj=str(out / f), path_in_repo=f, repo_id=a.repo)
    for f in a.extra:
        api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=a.repo)
    info = api.model_info(a.repo)
    print(json.dumps({"repo": a.repo, "private": info.private, "files": [s.rfilename for s in info.siblings]},
                     indent=2))


def cmd_dataset(a):
    import pyarrow as pa, pyarrow.parquet as pq
    from huggingface_hub import HfApi
    t0 = time.time()
    if not Path("/tmp/audio").exists():
        subprocess.run(["tar", "-xf", "/vol/pack/train_pack.tar", "-C", "/tmp", "audio"], check=True)
    print(f"audio unpacked in {time.time() - t0:.0f}s", flush=True)
    ds = Path("/tmp/ds")
    (ds / "data").mkdir(parents=True, exist_ok=True)
    features = {"id": {"dtype": "string", "_type": "Value"}, "split": {"dtype": "string", "_type": "Value"},
                "source_dataset": {"dtype": "string", "_type": "Value"},
                "speaker": {"dtype": "string", "_type": "Value"},
                "duration": {"dtype": "float32", "_type": "Value"},
                "transcript": {"dtype": "string", "_type": "Value"},
                "words": {"dtype": "string", "_type": "Value"},
                "audio": {"_type": "Audio", "sampling_rate": 24000}}
    meta = {b"huggingface": json.dumps({"info": {"features": features}}).encode()}
    counts = {}
    for split in ["train", "valid"]:
        rows = [json.loads(l) for l in open(f"/vol/data/{split}.jsonl", encoding="utf-8")]
        per = 2500
        for s in range(0, len(rows), per):
            chunk = rows[s:s + per]
            cols = {k: [] for k in ["id", "split", "source_dataset", "speaker", "duration", "transcript", "words", "audio"]}
            for r in chunk:
                local = r["path"].replace("/vol/audio/", "/tmp/audio/")
                cols["id"].append(r["id"]); cols["split"].append(split)
                cols["source_dataset"].append(r["dataset"]); cols["speaker"].append(r["speaker"])
                cols["duration"].append(float(r["duration"])); cols["transcript"].append(r["transcript"])
                cols["words"].append(json.dumps([{"word": w["word"], "start": w["start"], "end": w["end"]}
                                                 for w in r["words"]], ensure_ascii=False))
                cols["audio"].append({"bytes": open(local, "rb").read(), "path": os.path.basename(local)})
            table = pa.table({**{k: v for k, v in cols.items() if k not in ("duration", "audio")},
                              "duration": pa.array(cols["duration"], pa.float32()),
                              "audio": pa.array(cols["audio"], pa.struct([("bytes", pa.binary()), ("path", pa.string())]))})
            table = table.replace_schema_metadata(meta)
            n = len(range(0, len(rows), per))
            pq.write_table(table, ds / "data" / f"{split}-{s // per:05d}-of-{n:05d}.parquet")
            counts[split] = counts.get(split, 0) + len(chunk)
        print(f"{split}: {counts[split]} rows", flush=True)
    for f in a.extra:
        shutil.copy(f, ds / os.path.basename(f))
    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(repo_id=a.repo, folder_path=str(ds), repo_type="dataset")
    info = api.dataset_info(a.repo)
    print(json.dumps({"repo": a.repo, "private": info.private, "rows": counts,
                      "files": len(info.siblings), "minutes": round((time.time() - t0) / 60, 1)}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export"); p.add_argument("--run_dir"); p.add_argument("--checkpoint")
    p.add_argument("--tokenizer"); p.add_argument("--repo"); p.add_argument("--out")
    p = sub.add_parser("smoke"); p.add_argument("--source"); p.add_argument("--out")
    p = sub.add_parser("upload"); p.add_argument("--repo"); p.add_argument("--out"); p.add_argument("--extra", nargs="*", default=[])
    p = sub.add_parser("dataset"); p.add_argument("--repo"); p.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    {"export": cmd_export, "smoke": cmd_smoke, "upload": cmd_upload, "dataset": cmd_dataset}[a.cmd](a)


if __name__ == "__main__":
    main()
