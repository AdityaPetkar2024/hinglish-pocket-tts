"""Offline analysis of eval.py outputs (results/*_preds.jsonl). No audio or GPU needed.

1. corpus WER/CER with 95% bootstrap CIs (resampling utterances), plus paired differences
2. TA-WER: transliteration-aware WER, where a cross-script substitution counts as correct if
   both words reduce to the same consonant skeleton ("click" ~ "क्लिक", "loop" ~ "लूप")
3. word-error breakdown: cross-script matches, other substitutions, insertions, deletions,
   and the share of all errors that sit in looping utterances
usage: python analyze.py results_dir [more dirs...]  ->  writes analysis.json, prints tables
"""
import glob, json, os, random, re, sys
from collections import Counter

import jiwer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import norm, romanize, is_deva, has_loop


def skeleton(w):
    """Rough phonetic key: romanize, fold common English/ITRANS spelling variants, keep consonants."""
    r = romanize(w)
    r = re.sub(r"w$", "", r)                        # silent final w: window, show
    r = re.sub(r"m(?=[kgcjtdn])", "n", r)          # anusvara before a consonant: कमांड -> kamand
    for a, b in [("ck", "k"), ("ph", "f"), ("q", "k"), ("c", "k"), ("x", "ks"), ("w", "v"), ("z", "j")]:
        r = r.replace(a, b)
    r = re.sub(r"([kgcjtdpbs])h", r"\1", r)       # drop aspiration / sh-ch-th digraphs
    r = re.sub(r"[aeiouy]", "", r)                 # vowels and y are unreliable across scripts
    return re.sub(r"(.)\1+", r"\1", r)             # collapse doubles


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def cross_script_match(rw, pw):
    if is_deva(rw) == is_deva(pw):
        return False
    a, b = skeleton(rw), skeleton(pw)
    if min(len(a), len(b)) < 2:                    # too short to tell apart (new ~ नया)
        return False
    return lev(a, b) / max(len(a), len(b)) <= 0.34


def utt_counts(ref, pred):
    """Per-utterance edit counts on normalized text, with cross-script matches split out."""
    R, P = norm(ref), norm(pred)
    n = len(R.split())
    if not P:
        return {"n": n, "sub": 0, "ins": 0, "del": n, "xs": 0, "loop": False}
    out = jiwer.process_words(R, P)
    rws, pws = out.references[0], out.hypotheses[0]
    xs = sub = 0
    for c in out.alignments[0]:
        if c.type == "substitute":
            for k in range(c.ref_end_idx - c.ref_start_idx):
                if cross_script_match(rws[c.ref_start_idx + k], pws[c.hyp_start_idx + k]):
                    xs += 1
                else:
                    sub += 1
    return {"n": n, "sub": sub, "xs": xs, "ins": out.insertions, "del": out.deletions, "loop": has_loop(pred)}


def char_counts(ref, pred):
    R, P = norm(ref), norm(pred)
    if not P:
        return len(R), len(R)
    o = jiwer.process_characters(R, P)
    return o.substitutions + o.deletions + o.insertions, len(R)


def corpus(cs, ccs):
    N = sum(c["n"] for c in cs)
    err = sum(c["sub"] + c["xs"] + c["ins"] + c["del"] for c in cs)
    ta = sum(c["sub"] + c["ins"] + c["del"] for c in cs)
    return err / N, ta / N, sum(e for e, _ in ccs) / sum(n for _, n in ccs)


def bootstrap(cs, ccs, B=2000, seed=0):
    rng = random.Random(seed)
    idx = range(len(cs))
    w, t, c = [], [], []
    for _ in range(B):
        s = [rng.choice(idx) for _ in idx]
        a, b, cc = corpus([cs[i] for i in s], [ccs[i] for i in s])
        w.append(a); t.append(b); c.append(cc)
    q = lambda v: (round(sorted(v)[int(0.025 * B)], 4), round(sorted(v)[int(0.975 * B) - 1], 4))
    return q(w), q(t), q(c)


def paired_diff(csA, csB, B=2000, seed=0):
    """95% CI of WER(A) - WER(B) on the same utterances."""
    rng = random.Random(seed)
    idx = range(len(csA))
    d = []
    for _ in range(B):
        s = [rng.choice(idx) for _ in idx]
        NA = sum(csA[i]["n"] for i in s)
        eA = sum(csA[i]["sub"] + csA[i]["xs"] + csA[i]["ins"] + csA[i]["del"] for i in s)
        eB = sum(csB[i]["sub"] + csB[i]["xs"] + csB[i]["ins"] + csB[i]["del"] for i in s)
        d.append((eA - eB) / NA)
    d.sort()
    return round(d[int(0.025 * B)], 4), round(d[int(0.975 * B) - 1], 4)


def main(dirs):
    systems = {}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*_preds.jsonl")):
            tag = os.path.basename(f)[: -len("_preds.jsonl")]
            rows = [json.loads(l) for l in open(f, encoding="utf-8")]
            systems[tag] = {r["id"]: r for r in rows}
    ids = sorted(set.intersection(*[set(s) for s in systems.values()]))
    print(f"{len(systems)} systems, {len(ids)} shared utterances\n")

    res, counts = {}, {}
    for tag, rows in systems.items():
        cs = [utt_counts(rows[i]["ref"], rows[i]["pred"]) for i in ids]
        ccs = [char_counts(rows[i]["ref"], rows[i]["pred"]) for i in ids]
        counts[tag] = cs
        wer, ta, cer = corpus(cs, ccs)
        wci, taci, cci = bootstrap(cs, ccs)
        N = sum(c["n"] for c in cs)
        tot = {k: sum(c[k] for c in cs) for k in ["sub", "xs", "ins", "del"]}
        errs = sum(tot.values())
        loop_err = sum(c["sub"] + c["xs"] + c["ins"] + c["del"] for c in cs if c["loop"])
        res[tag] = {
            "wer": round(wer, 4), "wer_ci95": wci, "cer": round(cer, 4), "cer_ci95": cci,
            "ta_wer": round(ta, 4), "ta_wer_ci95": taci,
            "errors_per_ref_word": {k: round(v / N, 4) for k, v in tot.items()},
            "share_of_errors": {"cross_script": round(tot["xs"] / errs, 4) if errs else 0,
                                "insertions": round(tot["ins"] / errs, 4) if errs else 0,
                                "in_looping_utts": round(loop_err / errs, 4) if errs else 0},
            "looping_utts": sum(c["loop"] for c in cs), "ref_words": N,
        }

    print(f"{'system':22s} {'WER [95% CI]':24s} {'TA-WER':8s} {'CER [95% CI]':24s} {'xscript':>8s} {'ins':>6s} {'loops':>6s}")
    for tag, r in sorted(res.items(), key=lambda kv: kv[1]["wer"]):
        print(f"{tag:22s} {r['wer']:.3f} [{r['wer_ci95'][0]:.3f}, {r['wer_ci95'][1]:.3f}]   "
              f"{r['ta_wer']:.3f}    {r['cer']:.3f} [{r['cer_ci95'][0]:.3f}, {r['cer_ci95'][1]:.3f}]   "
              f"{r['share_of_errors']['cross_script']:7.1%} {r['share_of_errors']['insertions']:6.1%} {r['looping_utts']:6d}")

    pairs = [("ft_base_30h", "ft_small_30h"), ("ft_base_8h", "ft_base_30h"),
             ("whisper-base_nrng5", "ft_base_30h"), ("openai_whisper-base", "whisper-base_nrng5")]
    diffs = {}
    print("\npaired WER differences (A - B), 95% CI; an interval containing 0 means no detectable difference")
    for a, b in pairs:
        if a in counts and b in counts:
            ci = paired_diff(counts[a], counts[b])
            diffs[f"{a} - {b}"] = {"diff": round(res[a]["wer"] - res[b]["wer"], 4), "ci95": ci}
            print(f"  {a:22s} - {b:22s} = {res[a]['wer'] - res[b]['wer']:+.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    json.dump({"systems": res, "paired_differences": diffs, "n_utts": len(ids)},
              open("analysis.json", "w"), indent=2, ensure_ascii=False)
    return systems, counts, ids


if __name__ == "__main__":
    main(sys.argv[1:] or ["results"])
