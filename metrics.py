"""Metric helpers shared by eval.py, eval_cpp.py and analyze.py (light deps: jiwer, indic-transliteration)."""
import re, unicodedata
from collections import Counter

import jiwer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


def norm(t):
    t = unicodedata.normalize("NFC", t).lower().replace("।", " ")
    # strip punctuation/symbols but KEEP Devanagari vowel signs (category M);
    # Whisper's BasicTextNormalizer deletes them
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return " ".join(t.split())


def is_deva(w):
    return any("\u0900" <= c <= "\u097f" for c in w)


# ITRANS has no mapping for candra vowels (ॉ ऑ ॅ ऍ), which Hindi uses mainly for
# English loanwords ("ऑफिस"); left alone they'd be stripped below ("phisa").
DEVA_FIX = str.maketrans({"ऑ": "ओ", "ॉ": "ो", "ऍ": "ए", "ॅ": "े"})


def romanize(t):
    words = []
    for w in norm(t).split():
        if is_deva(w):
            w = transliterate(w.translate(DEVA_FIX), sanscript.DEVANAGARI, sanscript.ITRANS)
            # word-final schwa deletion: ITRANS writes the inherent vowel as lowercase
            # "a" (long आ is "A"), so "करना"->"karanA" keeps it, "ऑफिस"->"ophisa"->"ophis"
            if len(w) > 2 and w[-1] == "a" and w[-2].lower() not in "aeiou":
                w = w[:-1]
            w = w.lower()
        words.append(re.sub(r"[^a-z0-9]", "", w))
    return " ".join(w for w in words if w)


ARABIC = [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]


def script_of(t):
    c = Counter()
    for ch in t:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            c["devanagari"] += 1
        elif any(a <= o <= b for a, b in ARABIC):
            c["arabic"] += 1
        elif ch.isascii() and ch.isalpha():
            c["latin"] += 1
        elif ch.isalpha():
            c["other"] += 1
    return c.most_common(1)[0][0] if c else "empty"


# a 2-30 char unit repeated 4+ times in a row ("लिविविविवि", "click click click click click")
LOOP = re.compile(r"(.{2,30}?)\1{3,}")


def has_loop(t):
    return bool(LOOP.search(t))


def share(labels):
    c = Counter(labels)
    return {k: round(v / len(labels), 4) for k, v in c.most_common()}


def utt_wer(r, p):
    return 1.0 if not p else jiwer.wer(r, p)
