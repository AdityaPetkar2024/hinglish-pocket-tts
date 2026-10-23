"""Data and tokenizer tools for the Hinglish Pocket TTS fine-tune. Runs inside the pinned
pocket-tts environment (uv.lock) on Modal; see modal_05_build.py.

  manifests    apply the reviewed loanword map (Devanagari spelling -> English spelling for
               English words spoken inside Hindi speech), write train/valid manifests
  tokenizers   (a) EXTENDED: the released English tokenizer + Devanagari pieces appended, so
                   every English string tokenizes exactly as before (checked)
               (b) BASELINE: Kyutai's recipe, a fresh 4000-piece SentencePiece on our text
  transplant   new text-embedding table for (a): rows 0..3999 are the released English
               embeddings, Devanagari rows are drawn from N(mean, std) of the English rows
  configs      training yamls for both runs (identical except tokenizer + embedding init)
"""
import argparse, collections, json, os, random, re, subprocess, sys, unicodedata
from pathlib import Path

POCKET = Path("/opt/pocket-tts")
BASE_CFG = "pocket_tts/config/english_2026-09_24l.yaml"
DEVA = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
TRAIL = ".,?!;:"

SAMPLE_SENTENCES = [
    "मुझे कल office जाना है, लेकिन मेरी car खराब हो गई है.",
    "आज की meeting में हम new project के बारे में discuss करेंगे.",
    "Please अपना password किसी के साथ share मत करना.",
    "I think यह plan काफ़ी अच्छा है, let's start करते हैं.",
]


def apply_map(text_words, mapping):
    out, n = [], 0
    for w in text_words:
        core = w.rstrip(TRAIL)
        tail = w[len(core):]
        key = unicodedata.normalize("NFC", core)
        if key in mapping:
            out.append(mapping[key] + tail); n += 1
        else:
            out.append(w)
    if out and LATIN.match(out[0][:1] or ""):
        out[0] = out[0][0].upper() + out[0][1:]
    return out, n


def cmd_manifests(a):
    mapping = {}
    for line in open(a.map, encoding="utf-8"):
        if not line.strip() or line.startswith("devanagari"):
            continue
        d, e = line.rstrip("\n").split("\t")[:2]
        mapping[unicodedata.normalize("NFC", d)] = e
    os.makedirs(a.out, exist_ok=True)
    fo = {s: open(f"{a.out}/{s}.jsonl", "w", encoding="utf-8") for s in ["train", "valid"]}
    st = collections.defaultdict(collections.Counter)
    used = collections.Counter()
    examples = []
    for line in open(a.aligned, encoding="utf-8"):
        r = json.loads(line)
        words = [w["word"] for w in r["words"]]
        new, n = apply_map(words, mapping)
        for w, nw in zip(r["words"], new):
            if nw != w["word"]:
                used[f'{w["word"].rstrip(TRAIL)} -> {nw.rstrip(TRAIL)}'] += 1
            w["word"] = nw
        r["transcript"] = " ".join(new)
        ds, split = r["dataset"], "valid" if r["split"] == "val" else "train"
        k = f"{ds}/{split}"
        st[k]["utts"] += 1
        st[k]["sec"] += r["duration"]
        has_d, has_l = bool(DEVA.search(r["transcript"])), bool(LATIN.search(r["transcript"]))
        st[k]["code_mixed_utts"] += has_d and has_l
        st[k]["code_mixed_sec"] += r["duration"] if has_d and has_l else 0
        st[k]["rescripted_words"] += n
        if n and len(examples) < 12 and random.random() < 0.01:
            examples.append(r["transcript"])
        keep = {k2: r[k2] for k2 in ["path", "duration", "transcript", "words", "id", "dataset", "speaker"]}
        fo[split].write(json.dumps(keep, ensure_ascii=False) + "\n")
    for f in fo.values():
        f.close()
    res = {k: {"utts": v["utts"], "hours": round(v["sec"] / 3600, 2),
               "code_mixed_utts": v["code_mixed_utts"], "code_mixed_hours": round(v["code_mixed_sec"] / 3600, 2),
               "rescripted_words": v["rescripted_words"]} for k, v in sorted(st.items())}
    out = {"splits": res, "map_entries": len(mapping), "map_entries_used": len(used),
           "top_rewrites": used.most_common(25), "examples": examples}
    json.dump(out, open(f"{a.out}/manifest_stats.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=POCKET, check=True)


def english_tokenizer_path():
    import yaml
    cfg = yaml.safe_load((POCKET / BASE_CFG).read_text())
    from pocket_tts.utils.utils import download_if_necessary
    return str(download_if_necessary(cfg["flow_lm"]["lookup_table"]["tokenizer_path"])), cfg


def cmd_tokenizers(a):
    import sentencepiece as spm, tokenizers
    os.makedirs(a.out, exist_ok=True)
    train = [json.loads(l)["transcript"] for l in open(f"{a.data}/train.jsonl", encoding="utf-8")]
    valid = [json.loads(l)["transcript"] for l in open(f"{a.data}/valid.jsonl", encoding="utf-8")]

    # ---- (b) baseline: Kyutai's finetune_language recipe, fresh 4000-piece tokenizer
    run([sys.executable, "-m", "training.scripts.train_tokenizer", f"{a.out}/mixed4000",
         f"{a.data}/train.jsonl", "--vocab-size", "4000", "--character-coverage", "1.0"])
    run([sys.executable, "-m", "training.scripts.convert_tokenizer", f"{a.out}/mixed4000.model",
         "--out", f"{a.out}/mixed4000.json", "--texts", f"{a.data}/valid.jsonl"])

    # ---- (a) extended: Devanagari-only SentencePiece, appended to the English vocab
    deva_txt = f"{a.out}/deva_only.txt"
    with open(deva_txt, "w", encoding="utf-8") as f:
        for t in train:
            ws = [w.rstrip(TRAIL) for w in t.split() if DEVA.search(w)]
            if ws:
                f.write(" ".join(ws) + "\n")
    run([sys.executable, "-m", "training.scripts.train_tokenizer", f"{a.out}/deva", deva_txt,
         "--vocab-size", str(a.deva_vocab), "--character-coverage", "1.0"])
    sp = spm.SentencePieceProcessor(model_file=f"{a.out}/deva.model")

    en_path, _ = english_tokenizer_path()
    en = json.loads(Path(en_path).read_text(encoding="utf-8"))
    assert en["model"]["type"] == "Unigram", en["model"]["type"]
    en_vocab = en["model"]["vocab"]
    n_en = len(en_vocab)
    have = {p for p, _ in en_vocab}
    new_pieces = []
    for i in range(sp.get_piece_size()):
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_byte(i):
            continue
        p = sp.id_to_piece(i)
        if p in have or not DEVA.search(p):
            continue
        new_pieces.append((p, sp.get_score(i)))

    def build(delta):
        j = json.loads(json.dumps(en))
        j["model"]["vocab"] = en_vocab + [[p, s + delta] for p, s in new_pieces]
        return j, tokenizers.Tokenizer.from_str(json.dumps(j))

    # choose the score offset that best reproduces the standalone Devanagari segmentation
    probe = [w.rstrip(TRAIL) for t in valid for w in t.split() if DEVA.search(w)][:5000]
    want = [sp.encode(w, out_type=str) for w in probe]
    best = None
    for delta in [x / 2 for x in range(-16, 17)]:
        _, tok = build(delta)
        got = [tok.encode(w).tokens for w in probe]
        agree = sum(g == w for g, w in zip(got, want)) / len(probe)
        if best is None or (agree, -abs(delta)) > (best[0], -abs(best[1])):
            best = (agree, delta)
    j, tok = build(best[1])

    # English must be untouched: Latin-only text gets identical ids from old and new tokenizer
    en_tok = tokenizers.Tokenizer.from_file(en_path)
    latin = [t for t in train + valid if not DEVA.search(t)][:3000]
    latin += [w for t in train if DEVA.search(t) for w in t.split() if LATIN.search(w)][:3000]
    diff = sum(en_tok.encode(t).ids != tok.encode(t).ids for t in latin)
    assert diff == 0, f"{diff} English strings tokenize differently"
    unk = sum(i == j["model"].get("unk_id", 0) for t in valid for i in tok.encode(t).ids)
    byte_ids = {i for i, (p, _) in enumerate(j["model"]["vocab"]) if re.fullmatch(r"<0x[0-9A-F]{2}>", p)}
    n_tok = sum(len(tok.encode(t).ids) for t in valid)
    n_byte = sum(i in byte_ids for t in valid for i in tok.encode(t).ids)
    Path(f"{a.out}/extended.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")

    base = tokenizers.Tokenizer.from_file(f"{a.out}/mixed4000.json")
    def tpw(tk, texts):
        return round(sum(len(tk.encode(t).ids) for t in texts) / sum(len(t.split()) for t in texts), 3)
    deva_texts = [t for t in valid if DEVA.search(t)]
    res = {"english_vocab": n_en, "devanagari_pieces_added": len(new_pieces),
           "extended_vocab": len(j["model"]["vocab"]), "score_offset": best[1],
           "agreement_with_standalone_deva_sp": round(best[0], 4),
           "english_strings_checked_identical": len(latin), "valid_unk_tokens": unk,
           "valid_byte_fallback_share": round(n_byte / n_tok, 5),
           "tokens_per_word_valid_hindi": {"extended": tpw(tok, deva_texts), "baseline4000": tpw(base, deva_texts)},
           "tokens_per_word_valid_english": {"extended": tpw(tok, latin[:1000]), "baseline4000": tpw(base, latin[:1000]),
                                             "released_english": tpw(en_tok, latin[:1000])},
           "example": {"text": SAMPLE_SENTENCES[0], "extended": tok.encode(SAMPLE_SENTENCES[0]).tokens,
                       "baseline4000": base.encode(SAMPLE_SENTENCES[0]).tokens}}
    json.dump(res, open(f"{a.out}/tokenizer_stats.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_transplant(a):
    import torch, yaml
    from safetensors.torch import load_file, save_file
    from pocket_tts.utils.utils import download_if_necessary
    cfg = yaml.safe_load((POCKET / BASE_CFG).read_text())
    state = load_file(str(download_if_necessary(cfg["weights_path"])))
    key = "flow_lm.conditioner.embed.weight"
    if key not in state:
        raise SystemExit("embedding key not found; candidates: " + str([k for k in state if "embed" in k]))
    old = state[key].float()
    n_old = cfg["flow_lm"]["lookup_table"]["n_bins"]
    assert old.shape[0] == n_old + 1, old.shape
    n_new = len(json.loads(Path(a.tokenizer).read_text(encoding="utf-8"))["model"]["vocab"])
    g = torch.Generator().manual_seed(0)
    mu, sd = old[:n_old].mean(0), old[:n_old].std(0)
    new = torch.empty(n_new + 1, old.shape[1])
    new[:n_old] = old[:n_old]                                    # English rows, unchanged
    new[n_old:n_new] = mu + sd * torch.randn(n_new - n_old, old.shape[1], generator=g)
    new[n_new] = old[n_old]                                      # padding row stays last
    state[key] = new.to(state[key].dtype)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    save_file(state, a.out)
    print(json.dumps({"embedding": list(new.shape), "copied_rows": n_old, "new_rows": n_new - n_old,
                      "old_row_norm_mean": round(old[:n_old].norm(dim=1).mean().item(), 4),
                      "new_row_norm_mean": round(new[n_old:n_new].norm(dim=1).mean().item(), 4)}))
    return n_new


def cmd_configs(a):
    import yaml
    n_ext = len(json.loads(Path(f"{a.tok}/extended.json").read_text(encoding="utf-8"))["model"]["vocab"])
    common = {
        "model_config": BASE_CFG, "start_from_pretrained": True,
        "data": {"train_jsonl": f"{a.data}/train.jsonl", "valid_jsonl": f"{a.data}/valid.jsonl",
                 "max_duration_sec": 30.0, "max_voice_prompt_sec": 5.0},
        "flow": {"type": "lsd", "kwargs": {}}, "flow_batch_multiplier": 4, "eos_loss_weight": 0.1,
        "text_dropout": 0.2, "voice_dropout": 0.2,
        "optim": {"lr": 2e-4, "schedule": "constant", "warmup_steps": 1000},
        "batch_size": 64, "grad_accum_steps": 1, "max_steps": a.steps,
        "stats_ema_steps": 1000, "ema_decay": 0.999, "log_freq": 50,
        "valid_freq": 2500, "ckpt_freq": 2500, "num_ckpt_keep": 999,
        "sample_sentences": SAMPLE_SENTENCES, "sample_freq": 2500, "sample_cfg_coef": 2.0,
    }
    ours = common | {"run_dir": f"{a.runs}/extended", "reset_text_embedding": False,
                     "model_overrides": {"weights_path": a.weights,
                                         "flow_lm.lookup_table.tokenizer": "tokenizers",
                                         "flow_lm.lookup_table.tokenizer_path": f"{a.tok}/extended.json",
                                         "flow_lm.lookup_table.n_bins": n_ext}}
    # budget: the baseline runs to 10k steps; both are compared at the same step (10k)
    base = common | {"run_dir": f"{a.runs}/baseline", "reset_text_embedding": True, "max_steps": 10000,
                     "model_overrides": {"flow_lm.lookup_table.tokenizer": "tokenizers",
                                         "flow_lm.lookup_table.tokenizer_path": f"{a.tok}/mixed4000.json",
                                         "flow_lm.lookup_table.n_bins": 4000}}
    released = common | {"run_dir": f"{a.runs}/released_unused", "reset_text_embedding": False}
    os.makedirs(a.out, exist_ok=True)
    for name, c in [("extended", ours), ("baseline", base), ("released", released)]:
        Path(f"{a.out}/{name}.yaml").write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding="utf-8")
        # inference config: the model config with our tokenizer/size, weights filled in after training
        print(f"wrote {a.out}/{name}.yaml")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifests"); p.add_argument("--aligned"); p.add_argument("--map"); p.add_argument("--out")
    p = sub.add_parser("tokenizers"); p.add_argument("--data"); p.add_argument("--out"); p.add_argument("--deva_vocab", type=int, default=2000)
    p = sub.add_parser("transplant"); p.add_argument("--tokenizer"); p.add_argument("--out")
    p = sub.add_parser("configs"); p.add_argument("--data"); p.add_argument("--tok"); p.add_argument("--weights")
    p.add_argument("--runs"); p.add_argument("--out"); p.add_argument("--steps", type=int, default=20000)
    a = ap.parse_args()
    {"manifests": cmd_manifests, "tokenizers": cmd_tokenizers, "transplant": cmd_transplant, "configs": cmd_configs}[a.cmd](a)


if __name__ == "__main__":
    main()
