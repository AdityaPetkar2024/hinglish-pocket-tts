"""Phonetic readings for cross-script word matching (from loan/find_loanwords.py, which found
the loanwords; shared here so the evaluation uses the same phonetics)."""
import re


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
