"""Hinglish Pocket TTS on Modal: build data/tokenizers, train, and check progress.

Stages (run from ~/hinglish-tts on your laptop):
  modal run modal_tts.py::smoke                 1 min, L4: GPU + pinned pocket-tts install work
  modal run modal_tts.py::build                 ~10 min, CPU: manifests, tokenizers, embeddings, configs
  modal run modal_tts.py::pack                  ~15 min, CPU: bundle latents + audio into one tar (after
                                                the first training run has precomputed the latents)
  modal run --detach modal_tts.py::train --run extended   ~1-1.5 h, H100 (laptop can be closed)
  modal run modal_tts.py::status --run extended           training progress so far
  modal run modal_tts.py::evaluate                         ~20 min, GPU: extended vs baseline vs released English
  modal run modal_tts.py::release                          ~15 min, GPU: export the 5k EMA model, test it,
                                                           upload to the PRIVATE HF model repo, test from the hub
  modal run --detach modal_tts.py::dataset                 ~1 h, CPU: final training data -> PRIVATE HF dataset repo
  modal run modal_tts.py::speak --file demo.txt --voice me.wav   demo clips into ./demo (model from the hub)

All code runs in Kyutai's pinned environment (uv.lock at commit a922b91). Everything is stored on
the private Modal volume "hinglish-tts"; nothing is published.
"""
import modal

POCKET_COMMIT = "a922b912dfa366975294afb568d815b305acde5d"
PY = "/opt/pocket-tts/.venv/bin/python"
REPO = "/opt/pocket-tts"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "build-essential", "curl")
    .pip_install("uv")
    .run_commands(
        f"git clone https://github.com/kyutai-labs/pocket-tts {REPO} && "
        f"cd {REPO} && git checkout {POCKET_COMMIT} && uv sync --frozen --group dev",
    )
    .run_commands(f"uv pip install --python {REPO}/.venv/bin/python indic-transliteration cmudict pyarrow soundfile")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_file("tts_tools.py", "/root/tts_tools.py")
    .add_local_file("tts_eval.py", "/root/tts_eval.py")
    .add_local_file("eval_sentences.tsv", "/root/eval_sentences.tsv")
    .add_local_file("metrics.py", "/root/metrics.py")
    .add_local_file("analyze.py", "/root/analyze.py")
    .add_local_file("xscript_phon.py", "/root/xscript_phon.py")
    .add_local_file("loan/loanword_map.tsv", "/root/loanword_map.tsv")
    .add_local_file("infer.py", "/root/infer.py")
    .add_local_file("tts_publish.py", "/root/tts_publish.py")
    .add_local_file("hf_model/README.md", "/root/hf_model/README.md")
    .add_local_file("hf_dataset/README.md", "/root/hf_dataset/README.md")
)
app = modal.App("hinglish-tts")
vol = modal.Volume.from_name("hinglish-tts", create_if_missing=True)
hf = modal.Secret.from_name("huggingface")
MODEL_REPO = "AdityaPetkar2006/pocket-tts-hinglish"
DATA_REPO = "AdityaPetkar2006/hinglish-tts-data"
FINAL_RUN, FINAL_STEP = "extended", 5000          # chosen on the evaluation (see /vol/eval/main)
RELEASE_DIR = "/vol/release/hinglish"


def sh(cmd, cwd=REPO):
    import subprocess
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


@app.function(image=image, gpu=["L4", "A10G", "L40S"], timeout=900, secrets=[hf])
def smoke_fn():
    sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
    sh(f"{PY} -c \"import torch, pocket_tts, transformers; print('torch', torch.__version__, 'cuda', "
       f"torch.version.cuda, 'available', torch.cuda.is_available()); "
       f"x = torch.randn(1024, 1024, device='cuda'); print('matmul ok', float((x @ x).sum()) != 0)\"")
    # the base model's weights are gated on HF; fail here, early, if access is missing
    sh(f"{PY} -c \"import yaml; from pocket_tts.utils.utils import download_if_necessary as d; "
       f"c = yaml.safe_load(open('pocket_tts/config/english_2026-09_24l.yaml')); "
       f"print(d(c['weights_path'])); print(d(c['flow_lm']['lookup_table']['tokenizer_path']))\"")


@app.function(image=image, volumes={"/vol": vol}, secrets=[hf], cpu=8, memory=32768, timeout=3 * 3600)
def build_fn():
    T = f"{PY} /root/tts_tools.py"
    sh(f"{T} manifests --aligned /vol/manifests/aligned.jsonl --map /root/loanword_map.tsv --out /vol/data")
    sh("cp /root/loanword_map.tsv /vol/data/loanword_map.tsv")
    sh(f"{T} tokenizers --data /vol/data --out /vol/tok")
    sh(f"{T} transplant --tokenizer /vol/tok/extended.json --out /vol/weights/extended_init.safetensors")
    sh(f"{T} configs --data /vol/data --tok /vol/tok --weights /vol/weights/extended_init.safetensors "
       f"--runs /vol/runs --out /vol/configs --steps 20000")
    vol.commit()


@app.function(image=image, volumes={"/vol": vol}, secrets=[hf], gpu=["H100", "A100-80GB"], cpu=8, memory=65536,
              timeout=8 * 3600)
def train_fn(run: str, max_steps: int = 0):
    """Trains from a LOCAL copy of the latents and audio: ~86k small files read off the network
    volume by a fresh container starve the GPU (0.2-0.3 it/s instead of 6). ::pack bundles them
    into one tar first, on a cheap CPU container."""
    import os, subprocess, time
    vol.reload()
    train_jsonl = None
    if os.path.exists("/vol/pack/train_pack.tar"):
        # one big sequential read instead of ~86k small ones (made by ::pack, on a cheap CPU box)
        t0 = time.time()
        sh("tar -xf /vol/pack/train_pack.tar -C /tmp && "
           "sed -i 's#\"/vol/audio/#\"/tmp/audio/#g' /tmp/data/train_latents.jsonl && "
           "du -sh /tmp/data /tmp/audio && head -c 200 /tmp/data/train_latents.jsonl", cwd="/")
        print(f"unpacked training data locally in {time.time() - t0:.0f}s", flush=True)
        train_jsonl = "/tmp/data/train_latents.jsonl"
    elif os.path.exists("/vol/data/train_latents.meta.json"):
        # latents exist but are not packed: training straight off the volume crawls at ~0.3 it/s
        raise SystemExit("run `modal run modal_tts.py::pack` first (cheap CPU job), then train again")
    # rewrite the config with the venv's yaml (the modal-side python has no pyyaml)
    sh(f"{PY} -c \"import yaml; c = yaml.safe_load(open('/vol/configs/{run}.yaml')); "
       f"c['data']['train_jsonl'] = {train_jsonl!r} or c['data']['train_jsonl']; "
       f"c['max_steps'] = {max_steps} or c['max_steps']; "
       f"yaml.safe_dump(c, open('/tmp/run.yaml', 'w'), allow_unicode=True, sort_keys=False)\"")
    p = subprocess.Popen([PY, "-m", "training.train", "/tmp/run.yaml"], cwd=REPO)
    last = time.time()
    while p.poll() is None:
        time.sleep(20)
        if time.time() - last > 300:       # persist checkpoints/logs every 5 min
            vol.commit(); last = time.time()
    vol.commit()
    if p.returncode != 0:
        raise SystemExit(f"training exited with {p.returncode}")


@app.function(image=image, volumes={"/vol": vol}, secrets=[hf], gpu=["H100", "A100-80GB", "L40S"],
              cpu=8, memory=65536, timeout=3 * 3600)
def evaluate_fn(systems_json: str, out: str):
    import json, threading
    vol.reload()
    stop = threading.Event()
    threading.Thread(target=lambda: [vol.commit() for _ in iter(lambda: stop.wait(120), True)],
                     daemon=True).start()       # keep generated audio even if a later step fails
    try:
        sh(f"{PY} /root/tts_eval.py --systems '{systems_json}' --out {out}", cwd=REPO)
    finally:
        stop.set(); vol.commit()
    return json.load(open(f"{out}/summary.json"))


@app.function(image=image, volumes={"/vol": vol}, cpu=4, memory=8192, timeout=3 * 3600)
def pack_fn():
    import time
    vol.reload()
    t0 = time.time()
    sh("mkdir -p /vol/pack && tar -cf /vol/pack/train_pack.tar.tmp -C /vol "
       "data/train_latents.jsonl data/train_latents.meta.json data/latents audio && "
       "mv /vol/pack/train_pack.tar.tmp /vol/pack/train_pack.tar && ls -la /vol/pack", cwd="/")
    vol.commit()
    print(f"packed in {(time.time() - t0) / 60:.1f} min", flush=True)


@app.function(image=image, volumes={"/vol": vol}, timeout=300)
def status_fn(run: str):
    import glob, json, os
    vol.reload()
    d = f"/vol/runs/{run}"
    out = {"checkpoints": sorted(os.path.basename(x) for x in glob.glob(f"{d}/checkpoint_*.pt")),
           "samples": sorted(os.path.relpath(x, d) for x in glob.glob(f"{d}/samples/**/*.wav", recursive=True))[-8:]}
    try:
        lines = open(f"{d}/progress.jsonl").read().splitlines()
        out["progress_last"] = [json.loads(l) for l in lines[-6:]]
        out["valid"] = [json.loads(l) for l in lines if "valid" in l][-6:]
    except FileNotFoundError:
        out["progress_last"] = "no progress.jsonl yet"
    return out


@app.function(image=image, volumes={"/vol": vol}, secrets=[hf], gpu=["L40S", "A100-80GB", "H100", "A10G"],
              cpu=8, memory=65536, timeout=2 * 3600)
def release_fn():
    import glob, json, os
    vol.reload()
    P = f"{PY} /root/tts_publish.py"
    sh(f"{P} export --run_dir /vol/runs/{FINAL_RUN} --checkpoint /vol/runs/{FINAL_RUN}/checkpoint_{FINAL_STEP:08d}.pt "
       f"--tokenizer /vol/tok/extended.json --repo {MODEL_REPO} --out {RELEASE_DIR}")
    vol.commit()
    sh(f"{P} smoke --source {RELEASE_DIR} --out /vol/release/smoke_local", cwd="/root")
    extra = [f for f in ["/root/hf_model/README.md", "/root/infer.py", "/vol/eval/main/summary.json"] if os.path.exists(f)]
    sh(f"{P} upload --repo {MODEL_REPO} --out {RELEASE_DIR} --extra {' '.join(extra)}")
    sh(f"{P} smoke --source {MODEL_REPO} --out /vol/release/smoke_hub", cwd="/root")   # fresh download
    vol.commit()
    stats = {}
    for f in ["/vol/stats/prep.json", "/vol/stats/align.json", "/vol/data/manifest_stats.json",
              "/vol/tok/tokenizer_stats.json", f"{RELEASE_DIR}/export_info.json"]:
        if os.path.exists(f):
            stats[f] = json.load(open(f))
    wavs = {f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}": open(f, "rb").read()
            for f in sorted(glob.glob("/vol/release/smoke_*/*"))}
    return stats, wavs


@app.function(image=image, volumes={"/vol": vol}, secrets=[hf], cpu=8, memory=32768, timeout=5 * 3600)
def dataset_fn():
    import os
    vol.reload()
    # refresh the model card (dataset numbers were added after ::release)
    sh(f"{PY} -c \"from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='/root/hf_model/README.md', "
       f"path_in_repo='README.md', repo_id='{MODEL_REPO}'); print('model card updated')\"")
    extra = [f for f in ["/root/hf_dataset/README.md", "/vol/data/loanword_map.tsv", "/vol/stats/prep.json",
                         "/vol/stats/align.json", "/vol/data/manifest_stats.json", "/vol/tok/tokenizer_stats.json"]
             if os.path.exists(f)]
    sh(f"{PY} /root/tts_publish.py dataset --repo {DATA_REPO} --extra {' '.join(extra)}")


@app.function(image=image, volumes={"/vol": vol}, secrets=[hf], gpu=["L40S", "A100-80GB", "H100", "A10G"],
              cpu=4, memory=32768, timeout=3600)
def speak_fn(lines: list, voice: bytes, voice_name: str, cfg: float):
    """Runs infer.py inside the pocket-tts venv (the container's own python has no torch)."""
    import glob, json, os
    vol.reload()
    if voice:
        vpath = f"/tmp/{voice_name}"
        open(vpath, "wb").write(voice)
    else:     # default: an IndicTTS speaker, prompt recording from the held-out utterances
        rows = [json.loads(l) for l in open("/vol/data/valid.jsonl", encoding="utf-8")]
        vpath = next(r["path"] for r in rows if r["dataset"] == "indictts_hi" and 4 <= r["duration"] <= 8)
    open("/tmp/lines.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    os.makedirs("/tmp/demo", exist_ok=True)
    sh(f"PYTHONPATH={REPO} {PY} /root/infer.py --file /tmp/lines.txt --voice {vpath} --out /tmp/demo/clip.wav "
       f"--cfg {cfg} --repo {MODEL_REPO}")
    files = sorted(glob.glob("/tmp/demo/clip*.wav"))
    return [open(f, "rb").read() for f in files]


@app.local_entrypoint()
def release():
    import json, os
    stats, wavs = release_fn.remote()
    os.makedirs("release_check", exist_ok=True)
    for name, data in wavs.items():
        path = os.path.join("release_check", name.replace("/", "__"))
        open(path, "wb").write(data)
    json.dump(stats, open("release_check/stats.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nsaved {len(wavs)} files + stats.json in ./release_check  (smoke_hub__* = generated from the HF repo)")


@app.function(image=image, secrets=[hf], timeout=600)
def cards_fn():
    sh(f"{PY} -c \"from huggingface_hub import HfApi; a = HfApi(); "
       f"a.upload_file(path_or_fileobj='/root/hf_model/README.md', path_in_repo='README.md', repo_id='{MODEL_REPO}'); "
       f"a.upload_file(path_or_fileobj='/root/hf_dataset/README.md', path_in_repo='README.md', repo_id='{DATA_REPO}', "
       f"repo_type='dataset'); print('both cards updated')\"")


@app.local_entrypoint()
def cards():
    """Re-upload the two README cards only (1 min, CPU)."""
    cards_fn.remote()


@app.local_entrypoint()
def dataset():
    dataset_fn.remote()


@app.local_entrypoint()
def speak(file: str = "demo.txt", voice: str = "", cfg: float = 2.0):
    """One wav per non-empty line of `file` (UTF-8), in the voice of `voice` (a local 4-10 s recording;
    empty = a held-out IndicTTS speaker). Saved to ./demo/."""
    import os
    lines = [l.strip() for l in open(file, encoding="utf-8").read().splitlines() if l.strip()]
    vb = open(voice, "rb").read() if voice else b""
    wavs = speak_fn.remote(lines, vb, os.path.basename(voice) or "voice.wav", cfg)
    os.makedirs("demo", exist_ok=True)
    for i, (line, data) in enumerate(zip(lines, wavs), 1):
        open(f"demo/{i:02d}.wav", "wb").write(data)
        print(f"demo/{i:02d}.wav  {line}")


@app.local_entrypoint()
def smoke():
    smoke_fn.remote()


@app.local_entrypoint()
def build():
    build_fn.remote()


@app.local_entrypoint()
def pack():
    pack_fn.remote()


@app.local_entrypoint()
def train(run: str = "extended", max_steps: int = 0):
    train_fn.remote(run, max_steps)


@app.local_entrypoint()
def evaluate(systems: str = "extended@5000,baseline@5000,extended@10000,extended@15000,released", tag: str = "main"):
    """Score systems on the same sentences and voices. Each entry is run@step (EMA weights of
    /vol/runs/<run>/checkpoint_<step>.pt) or `released` (Kyutai's English 24-layer model, fed
    romanized text). The first two are compared with a paired bootstrap."""
    import json
    specs = []
    for entry in systems.split(","):
        if entry == "released":
            specs.append({"name": "released_english_roman", "released": True,
                          "args_yaml": "/vol/configs/released.yaml", "roman_input": True})
        else:
            entry, _, opt = entry.partition(":")          # e.g. extended@5000:cfg1
            run, step = entry.split("@")
            spec = {"name": f"{run}_{int(step) // 1000}k", "run_dir": f"/vol/runs/{run}",
                    "checkpoint": f"/vol/runs/{run}/checkpoint_{int(step):08d}.pt", "ema": True}
            if opt.startswith("cfg"):
                spec["cfg"] = float(opt[3:]); spec["name"] += f"_{opt}"
            specs.append(spec)
    res = evaluate_fn.remote(json.dumps(specs), f"/vol/eval/{tag}")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"audio + per-clip results: modal volume get hinglish-tts eval/{tag} ./eval_{tag} --force")


@app.local_entrypoint()
def status(run: str = "extended"):
    import json
    print(json.dumps(status_fn.remote(run), indent=2, ensure_ascii=False))
