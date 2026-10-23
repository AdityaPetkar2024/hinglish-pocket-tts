"""Evaluate Hinglish TTS systems on held-out sentences and held-out voices.

Test set: eval_sentences.tsv, 40 sentences written for this eval (none occurs in the training
transcripts; checked below), in 6 categories: Hindi with one English word, Hindi with several,
English-matrix sentences with Hindi, pure Hindi, pure English, brand names.
Voices: 8 prompts from the validation split. IndicVoices-R and Svarah prompts are from speakers
never seen in training; the 2 IndicTTS speakers are seen, but these utterances are not.

Every system speaks every sentence in every voice (40 x 8 = 320 clips per system), same seed,
same sampler (temp 0.3, LSD 1 step, EOS threshold -1, Kyutai's eval settings), then:
  intelligibility   Whisper large-v3 transcribes each clip (language hi; en for pure English)
                    romanized CER: both texts transliterated to Latin (ITRANS) so script choice
                                   ("meeting" vs "मीटिंग") is not counted as an error
                    TA-WER: word error rate where a cross-script match counts as correct
                            (the metric from my Hinglish ASR project, analyze.py)
                    English-word recall: share of the English words in the text that the ASR
                            recovers (exactly or as a cross-script match): did the model say them?
  voice             WavLM x-vector cosine to a DIFFERENT real utterance of the prompt speaker
  quality           UTMOS (English-trained MOS predictor; indicative only for Hindi)
  failures          silent outputs, outputs that never stop (hit the 30 s cap)
ASR floor: the same ASR on the real validation recordings of these speakers, so the numbers
can be read against what Whisper does on real human speech.
95% CIs by bootstrap over sentences; paired difference for extended vs baseline.
"""
import argparse, collections, json, os, random, re, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root")
sys.path.insert(0, "/opt/pocket-tts")   # the `training` package is not part of the installed wheel
from metrics import norm, romanize, is_deva            # noqa: E402  (from the ASR project)
from analyze import utt_counts, cross_script_match     # noqa: E402
from xscript_phon import deva_phon, eng_phon, fold, lev  # noqa: E402  (loanword finder's phonetics)
import cmudict                                          # noqa: E402

_CMU = cmudict.dict()


def same_word(en_or_any, other):
    """True if two words are the same word, possibly in different scripts: exact, ASR-project
    skeleton match, or English pronunciation (CMU) within a small edit distance of the
    Devanagari word's phonetic reading (मीटिंग ~ meeting, बजट ~ budget, ऑफिस ~ office)."""
    if romanize(en_or_any) == romanize(other) or cross_script_match(en_or_any, other):
        return True
    e, d = (en_or_any, other) if LATIN.search(en_or_any) else (other, en_or_any)
    if not (LATIN.search(e) and is_deva(d)):
        return False
    dp = fold(deva_phon(d))
    for pron in _CMU.get(e.lower(), [])[:3]:
        ep = fold(eng_phon(pron))
        if lev(dp, ep) <= max(1, len(ep) // 4):
            return True
    return False

LATIN = re.compile(r"[A-Za-z]")


def read_sentences(path):
    rows = []
    for line in open(path, encoding="utf-8").read().splitlines()[1:]:
        i, cat, text, roman = line.split("\t")
        rows.append({"sid": i, "category": cat, "text": text, "roman": roman})
    return rows


def pick_voices(valid_jsonl, seed=0):
    """8 prompts: 2 IndicTTS (f/m), 4 IndicVoices-R (2f/2m), 2 Svarah; plus a different
    utterance of the same speaker as the speaker-similarity reference."""
    rng = random.Random(seed)
    by_spk = collections.defaultdict(list)
    for l in open(valid_jsonl, encoding="utf-8"):
        r = json.loads(l)
        by_spk[(r["dataset"], r["speaker"])].append(r)
    voices = []
    want = [("indictts_hi", None, 2), ("ivr_hi", None, 4), ("svarah_en", None, 2)]
    for ds, _, n in want:
        spks = sorted(k for k in by_spk if k[0] == ds and len(by_spk[k]) >= 3)
        rng.shuffle(spks)
        for key in spks[:n]:
            rs = sorted(by_spk[key], key=lambda r: abs(r["duration"] - 6.0))
            prompt = next((r for r in rs if 4.0 <= r["duration"] <= 9.0), rs[0])
            ref = next(r for r in rs if r["path"] != prompt["path"])
            voices.append({"vid": f"{ds}:{key[1][-6:]}", "dataset": ds, "speaker": key[1],
                           "prompt": prompt["path"], "ref": ref["path"],
                           "floor": [r for r in by_spk[key][:15]]})
    return voices


def english_words(text):
    return [w for w in norm(text).split() if LATIN.search(w)]


def english_recall(ref, hyp):
    """Of the English words in ref, how many does hyp contain at the aligned position
    (same word, or the same word in Devanagari)."""
    import jiwer
    R, H = norm(ref), norm(hyp)
    ew = [i for i, w in enumerate(R.split()) if LATIN.search(w)]
    if not ew:
        return None, 0
    if not H:
        return 0, len(ew)
    out = jiwer.process_words(R, H)
    rws, hws = out.references[0], out.hypotheses[0]
    ok_idx = set()
    for c in out.alignments[0]:
        for k in range(c.ref_end_idx - c.ref_start_idx):
            ri = c.ref_start_idx + k
            if c.type == "equal":
                ok_idx.add(ri)
            elif c.type == "substitute":
                hi = c.hyp_start_idx + k
                if same_word(rws[ri], hws[hi]):
                    ok_idx.add(ri)
    return sum(i in ok_idx for i in ew), len(ew)


def load_system(spec, device):
    """spec: {"name", "run_dir", "checkpoint"|None, "ema": bool} or {"name", "released": cfg}."""
    from training.args import load_args
    from training.eval.librispeech import load_run
    from training.modules.builders import build_models
    if "released" in spec:
        args = load_args(spec["args_yaml"])
        args.distill_cfg_coef = 0.0
        model, mimi, _ = build_models(args)
        model.to(device).eval(); mimi.to(device)
        return model, mimi, 0
    return load_run(spec["run_dir"], device, use_ema=spec.get("ema", True), checkpoint=spec.get("checkpoint"))


def bootstrap_ci(per_sent, B=2000, seed=0):
    """per_sent: list of (errors, total) per sentence -> 95% CI of sum(err)/sum(total)."""
    rng = random.Random(seed)
    n = len(per_sent)
    vals = []
    for _ in range(B):
        s = [per_sent[rng.randrange(n)] for _ in range(n)]
        vals.append(sum(e for e, _ in s) / max(1, sum(t for _, t in s)))
    vals.sort()
    return round(vals[int(0.025 * B)], 4), round(vals[int(0.975 * B) - 1], 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", required=True, help="json list of system specs")
    ap.add_argument("--sentences", default="/root/eval_sentences.tsv")
    ap.add_argument("--valid", default="/vol/data/valid.jsonl")
    ap.add_argument("--train", default="/vol/data/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--eos", type=float, default=-1.0)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--asr", default="openai/whisper-large-v3")
    a = ap.parse_args()
    device = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)
    import sphn
    from training.eval.librispeech import MIN_FRAMES, latents_to_wav, load_mono, load_16k

    sents = read_sentences(a.sentences)
    train_text = {norm(json.loads(l)["transcript"]) for l in open(a.train, encoding="utf-8")}
    leaks = [s["sid"] for s in sents if norm(s["text"]) in train_text]
    voices = pick_voices(a.valid)
    systems = json.loads(a.systems)

    # ---------------- generate
    items_all = {}
    for spec in systems:
        sdir = Path(a.out) / spec["name"]
        sdir.mkdir(parents=True, exist_ok=True)
        cache = sdir / "generated.json"
        if cache.exists():                  # generation finished on an earlier attempt: reuse it
            saved = json.load(open(cache, encoding="utf-8"))
            items_all[spec["name"]] = (saved["items"], saved["step"])
            print(f"[{spec['name']}] reusing {len(saved['items'])} generated clips", flush=True)
            continue
        model, mimi, step = load_system(spec, device)
        print(f"[{spec['name']}] loaded step {step}; generating", flush=True)
        tok = model.flow_lm.conditioner.tokenizer.encode
        items = [{"sid": s["sid"], "category": s["category"], "vid": v["vid"], "voice_ds": v["dataset"],
                  "ref_text": s["text"], "input": s["roman"] if spec.get("roman_input") else s["text"],
                  "prompt": v["prompt"], "spk_ref": v["ref"]} for v in voices for s in sents]
        torch.manual_seed(0)
        for i0 in range(0, len(items), a.batch):
            chunk = items[i0:i0 + a.batch]
            with torch.no_grad():
                vl = [mimi.encode_to_latent(load_mono(c["prompt"], mimi.sample_rate)[None, None].to(device))[0]
                      for c in chunk]
                outs = model.generate([torch.tensor(tok(c["input"]), dtype=torch.long) for c in chunk], vl,
                                      max_frames=375, temp=a.temp, n_steps=1, cfg_coef=spec.get("cfg", a.cfg), eos_threshold=a.eos)
            for c, lat in zip(chunk, outs):
                c["frames"] = int(lat.shape[0]); c["capped"] = int(lat.shape[0] >= 375)
                c["wav"] = None
                if lat.shape[0] >= MIN_FRAMES:
                    w = latents_to_wav(mimi, lat, device).float().cpu().numpy()
                    p = sdir / f'{c["sid"]}_{c["vid"].replace(":", "_")}.wav'
                    sphn.write_wav(str(p), w, int(mimi.sample_rate))
                    c["wav"] = str(p)
        items_all[spec["name"]] = (items, step)
        json.dump({"items": items, "step": step}, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[{spec['name']}] generated {sum(1 for c in items if c['wav'])}/{len(items)} clips", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---------------- score
    from transformers import AutoFeatureExtractor, WavLMForXVector, pipeline
    asr = pipeline("automatic-speech-recognition", model=a.asr, device=device, torch_dtype=torch.float16)
    def transcribe(paths, langs):
        out = []
        for lang in set(langs):
            idx = [i for i, l in enumerate(langs) if l == lang]
            audio = [load_16k(paths[i], device).cpu().numpy() for i in idx]
            res = asr([{"raw": x, "sampling_rate": 16000} for x in audio], batch_size=16,
                      generate_kwargs={"language": lang, "task": "transcribe"})
            for i, r in zip(idx, res):
                out.append((i, r["text"]))
        return [t for _, t in sorted(out)]
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    spk = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
    def xvec(path):
        x = load_16k(path, device).cpu().numpy()
        inp = fe([x], sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            return torch.nn.functional.normalize(spk(**inp).embeddings[0], dim=-1)
    try:
        from utmos_pytorch import UTMOSScoreTorch
        utmos = UTMOSScoreTorch(device="cuda")
    except Exception as e:  # noqa: BLE001
        print("UTMOS unavailable:", e); utmos = None
    ref_vec = {v["vid"]: xvec(v["ref"]) for v in voices}

    import jiwer
    summary = {"settings": {"cfg": a.cfg, "temp": a.temp, "eos_threshold": a.eos, "asr": a.asr,
                            "sentences": len(sents), "voices": [v["vid"] for v in voices],
                            "sentences_found_in_training_text": leaks}, "systems": {}}

    # ASR floor on real recordings of the same speakers
    floor = [r for v in voices for r in v["floor"]]
    hyp = transcribe([r["path"] for r in floor], ["en" if r["dataset"] == "svarah_en" else "hi" for r in floor])
    R, H = [romanize(r["transcript"]) for r in floor], [romanize(h) for h in hyp]
    summary["asr_floor_real_speech"] = {"utts": len(floor), "romanized_cer": round(jiwer.cer(R, H), 4),
                                        "ta_wer": round(sum(c["sub"] + c["ins"] + c["del"] for c in
                                                            [utt_counts(r["transcript"], h) for r, h in zip(floor, hyp)])
                                                        / sum(len(norm(r["transcript"]).split()) for r in floor), 4)}

    per_sys_sent = {}
    for name, (items, step) in items_all.items():
        ok = [c for c in items if c["wav"]]
        hyps = transcribe([c["wav"] for c in ok], ["en" if c["category"] == "english" else "hi" for c in ok])
        for c, h in zip(ok, hyps):
            c["hyp"] = h
            rr, hr = romanize(c["ref_text"]), romanize(h)
            if hr:
                pc = jiwer.process_characters(rr, hr)
                c["cer_err"] = pc.substitutions + pc.deletions + pc.insertions
            else:
                c["cer_err"] = len(rr)
            c["cer_len"] = len(rr)
            cnt = utt_counts(c["ref_text"], h)
            c["ta_err"], c["n_words"] = cnt["sub"] + cnt["ins"] + cnt["del"], cnt["n"]
            c["en_ok"], c["en_n"] = english_recall(c["ref_text"], h)
            c["sim"] = float(torch.dot(xvec(c["wav"]), ref_vec[c["vid"]]))
            if utmos is not None:
                w = load_16k(c["wav"], device)
                with torch.no_grad():
                    c["utmos"] = float(utmos.score(w[None, None]))
        for c in items:
            if not c["wav"]:
                c.update({"hyp": "", "cer_err": len(romanize(c["ref_text"])), "cer_len": len(romanize(c["ref_text"])),
                          "ta_err": len(norm(c["ref_text"]).split()), "n_words": len(norm(c["ref_text"]).split()),
                          "en_ok": 0 if english_words(c["ref_text"]) else None, "en_n": len(english_words(c["ref_text"]))})

        def agg(sel):
            sel = list(sel)
            by_sent = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
            for c in sel:
                b = by_sent[c["sid"]]
                b[0] += c["cer_err"]; b[1] += c["cer_len"]; b[2] += c["ta_err"]; b[3] += c["n_words"]
                b[4] += c["en_ok"] or 0; b[5] += c["en_n"]
            v = list(by_sent.values())
            sims = [c["sim"] for c in sel if "sim" in c]
            mos = [c["utmos"] for c in sel if "utmos" in c]
            res = {"clips": len(sel),
                   "romanized_cer": round(sum(x[0] for x in v) / sum(x[1] for x in v), 4),
                   "romanized_cer_ci95": bootstrap_ci([(x[0], x[1]) for x in v]),
                   "ta_wer": round(sum(x[2] for x in v) / sum(x[3] for x in v), 4),
                   "ta_wer_ci95": bootstrap_ci([(x[2], x[3]) for x in v]),
                   "speaker_sim": round(float(np.mean(sims)), 4) if sims else None,
                   "utmos": round(float(np.mean(mos)), 3) if mos else None,
                   "silent": sum(1 for c in sel if not c["wav"]), "no_eos": sum(c["capped"] for c in sel)}
            en_n = sum(x[5] for x in v)
            if en_n:
                res["english_word_recall"] = round(sum(x[4] for x in v) / en_n, 4)
                res["english_word_recall_ci95"] = tuple(round(1 - x, 4) for x in reversed(
                    bootstrap_ci([(x[5] - x[4], x[5]) for x in v if x[5]])))
            return res, by_sent

        overall, by_sent = agg(items)
        per_sys_sent[name] = by_sent
        summary["systems"][name] = {"step": step, "overall": overall,
                                    "by_category": {cat: agg(c for c in items if c["category"] == cat)[0]
                                                    for cat in sorted({c["category"] for c in items})},
                                    "by_voice_dataset": {ds: agg(c for c in items if c["voice_ds"] == ds)[0]
                                                         for ds in sorted({c["voice_ds"] for c in items})}}
        with open(Path(a.out) / f"{name}_items.jsonl", "w", encoding="utf-8") as f:
            for c in items:
                f.write(json.dumps({k: v for k, v in c.items() if k not in ("prompt", "spk_ref")}, ensure_ascii=False) + "\n")

    # paired bootstrap: TA-WER and romanized CER difference between the first two systems
    names = list(per_sys_sent)
    if len(names) >= 2:
        A, B_ = per_sys_sent[names[0]], per_sys_sent[names[1]]
        sids = sorted(set(A) & set(B_))
        rng = random.Random(0)
        for metric, (ei, ni) in {"ta_wer": (2, 3), "romanized_cer": (0, 1)}.items():
            d = []
            for _ in range(2000):
                s = [sids[rng.randrange(len(sids))] for _ in sids]
                d.append(sum(A[i][ei] for i in s) / sum(A[i][ni] for i in s) - sum(B_[i][ei] for i in s) / sum(B_[i][ni] for i in s))
            d.sort()
            summary.setdefault("paired", {})[f"{metric}: {names[0]} - {names[1]}"] = (round(d[50], 4), round(d[1949], 4))
    json.dump(summary, open(Path(a.out) / "summary.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
