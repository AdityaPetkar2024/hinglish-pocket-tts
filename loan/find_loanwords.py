"""Find English loanwords written in Devanagari in the Hindi transcripts (text only).

For each Devanagari word, compare a simple phonetic romanization against the pronunciations of
the 40k most frequent English words (CMU dictionary, ARPAbet mapped to how Hindi speakers adapt
English sounds: T->t, AY->ai, W->v, Z->j, ...). Close matches become CANDIDATES; every candidate
is then reviewed by hand before anything is rewritten.

usage: python find_loanwords.py deva_words.tsv candidates.tsv
"""
import re, sys
import cmudict
from wordfreq import top_n_list, zipf_frequency
from rapidfuzz import process as rf_process
from rapidfuzz.distance import Levenshtein

ARPA = {"AA": "a", "AE": "ai", "AH": "a", "AO": "o", "AW": "au", "AY": "ai", "EH": "e", "ER": "ar",
        "EY": "e", "IH": "i", "IY": "i", "OW": "o", "OY": "oy", "UH": "u", "UW": "u",
        "B": "b", "CH": "ch", "D": "d", "DH": "d", "F": "f", "G": "g", "HH": "h", "JH": "j", "K": "k",
        "L": "l", "M": "m", "N": "n", "NG": "ng", "P": "p", "R": "r", "S": "s", "SH": "sh", "T": "t",
        "TH": "th", "V": "v", "W": "v", "Y": "y", "Z": "j", "ZH": "j"}

CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
        "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
        "ध": "dh", "न": "n", "प": "p", "फ": "f", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
        "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
        "क़": "k", "ख़": "kh", "ग़": "g", "ज़": "j", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y"}
VOW = {"अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ए": "e", "ऐ": "ai", "ओ": "o",
       "औ": "au", "ऑ": "o", "ऍ": "ai", "ऋ": "ri"}
MATRA = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
         "ॉ": "o", "ॅ": "ai", "ृ": "ri"}
NUKTA, VIRAMA = "़", "्"


def deva_phon(w):
    """Phonetic-ish romanization with schwa deletion, e.g. टाइम -> taim, सर्विस -> sarvis."""
    import unicodedata
    w = unicodedata.normalize("NFC", w)
    out, i = [], 0
    chars = list(w)
    while i < len(chars):
        c = chars[i]
        if i + 1 < len(chars) and chars[i + 1] == NUKTA and c + NUKTA in CONS:
            c = c + NUKTA; i += 1
        if c in CONS:
            out.append(CONS[c])
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt in MATRA:
                out.append(MATRA[nxt]); i += 1
            elif nxt == VIRAMA:
                i += 1
            elif nxt in ("ं", "ँ"):
                out.append("a")
            else:
                out.append("a")  # inherent schwa; trimmed below
        elif c in VOW:
            out.append(VOW[c])
        elif c in ("ं", "ँ"):
            out.append("n")
        elif c == "ः":
            out.append("h")
        i += 1
    s = "".join(out)
    # schwa deletion: final, and medial in V C a C V contexts (कमरा -> kamra)
    s = re.sub(r"a$", "", s) if len(s) > 2 else s
    s = re.sub(r"(?<=[aeiou][^aeiou])a(?=[^aeiou][aeiou])", "", s)
    return s


def eng_phon(pron):
    return "".join(ARPA[re.sub(r"\d", "", p)] for p in pron)


def fold(s):
    s = s.replace("aa", "a").replace("ii", "i").replace("uu", "u")
    s = s.replace("kh", "k").replace("gh", "g").replace("chh", "ch").replace("jh", "j")
    s = s.replace("th", "t").replace("dh", "d").replace("bh", "b").replace("ph", "f")
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def main(src, dst):
    cmu = cmudict.dict()
    eng = {}
    for w in top_n_list("en", 40000):
        if not w.isalpha() or len(w) < 3 or w not in cmu:
            continue
        for pron in cmu[w][:2]:
            eng.setdefault(fold(eng_phon(pron)), []).append(w)
    by_len = {}
    for k in eng:
        by_len.setdefault(len(k), []).append(k)

    rows = []
    for line in open(src, encoding="utf-8").read().splitlines()[1:]:
        w, total, ivr, itts = line.split("\t")
        total, ivr, itts = int(total), int(ivr), int(itts)
        if ivr < 2:
            continue
        p = fold(deva_phon(w))
        if len(p) < 3:
            continue
        best = None
        maxd = 0 if len(p) <= 3 else 1 if len(p) <= 6 else 2
        keys = [k for L in range(len(p) - maxd, len(p) + maxd + 1) for k in by_len.get(L, [])]
        for k, d, _ in rf_process.extract(p, keys, scorer=Levenshtein.distance,
                                          score_cutoff=maxd, limit=20):
            cand = max(eng[k], key=lambda e: zipf_frequency(e, "en"))
            score = (d, -zipf_frequency(cand, "en"))
            if best is None or score < best[0]:
                best = (score, cand, k)
        if best:
            rows.append((total, ivr, itts, w, p, best[1], best[2], best[0][0]))
    rows.sort(reverse=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write("total\tivr\tindictts\tdeva\tdeva_phon\tenglish\teng_phon\tdist\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"{len(rows)} candidates written to {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
