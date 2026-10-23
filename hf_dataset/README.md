---
language:
- hi
- en
task_categories:
- text-to-speech
tags:
- hinglish
- code-switching
pretty_name: Hinglish TTS training set
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/valid-*.parquet
---

# Hinglish TTS training set

The exact data used to fine-tune `AdityaPetkar2006/pocket-tts-hinglish`: three public, human-transcribed
speech corpora, cleaned, force-aligned at word level, and with English loanwords in the Hindi
transcripts rewritten in Latin script.

| Split | Utterances | Hours | Code-mixed utterances | Code-mixed hours |
|---|---|---|---|---|
| train | 41,589 | 102.11 | 12,754 | 36.91 |
| validation | 1,269 | 2.91 | 270 | 0.78 |

| Source | Train utts | Train h | Valid utts | Valid h | Words rewritten to Latin (train) |
|---|---|---|---|---|---|
| IndicTTS Hindi | 11,522 | 24.37 | 174 | 0.37 | 4,148 |
| IndicVoices-R Hindi | 26,102 | 71.16 | 581 | 1.55 | 22,036 |
| Svarah (English) | 3,965 | 6.58 | 514 | 0.99 | 0 |

"Code-mixed" means the transcript contains at least one English word in Latin script after
preprocessing.

## Sources

| Dataset | Link | What it is | License |
|---|---|---|---|
| IndicTTS Hindi | [SPRINGLab/IndicTTS-Hindi](https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi) (Hugging Face copy of the IndicTTS database recorded by IIT Madras) | studio read speech, 2 speakers | IndicTTS database terms (IIT Madras) |
| IndicVoices-R | [ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r), Hindi subset | read and conversational speech, many speakers, human transcripts (verbatim and normalized) | CC BY 4.0 |
| Svarah | [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | English from 117 Indian speakers across 19 states, read and conversational | CC BY 4.0 |

Each source keeps its own license; this repo adds no rights. Please cite the original papers:
IndicVoices-R ("IndicVoices-R: Unlocking a Massive Multilingual Multi-speaker Speech Corpus for
Scaling Indian TTS") and Svarah ("Svarah: Evaluating English ASR Systems on Indian Accents").

## Columns

| Column | Type | Content |
|---|---|---|
| `id` | string | utterance id (source dataset + original file) |
| `split` | string | `train` or `valid` |
| `source_dataset` | string | `indictts_hi`, `ivr_hi` or `svarah_en` |
| `speaker` | string | speaker id from the source |
| `duration` | float | seconds |
| `transcript` | string | training text (after the steps below) |
| `words` | string (JSON) | `[{"word", "start", "end"}, ...]`, word timings in seconds |
| `audio` | Audio | 24 kHz mono FLAC |

```python
from datasets import load_dataset
ds = load_dataset("AdityaPetkar2006/hinglish-tts-data", token="hf_...")   # private repo
```

## Preprocessing

1. **Audio.** Resampled to 24 kHz mono (the Mimi codec's rate) with soxr VHQ from 48 kHz
   (IndicTTS, IndicVoices-R) or 16 kHz (Svarah), loudness normalized to -20 dBFS RMS with a -1 dBFS
   peak cap, stored as FLAC.
2. **Text.** Unicode NFC; danda (।, ॥) to "."; zero-width joiners removed; spaces before punctuation
   removed; terminal punctuation added where missing (Pocket TTS appends "." at inference).
   IndicVoices-R provides a verbatim and a normalized transcript: the normalized one (standard
   spelling, what users type) is the training text.
3. **Filters.** Dropped: duration outside 1 to 30 s; transcripts with digits (number readings are
   ambiguous and were not guessed); characters outside Devanagari, Latin and basic punctuation;
   Svarah fragments under 3 words.
4. **Split.** Speaker-disjoint validation for IndicVoices-R and Svarah. IndicTTS has only 2 speakers,
   so 1.5% of each speaker's utterances are held out instead.
5. **Word alignment.** Pocket TTS training needs word timings. Kyutai's aligner
   (`training/scripts/align_data.py` at commit a922b91, CTC Viterbi) with
   `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200` for Hindi and
   `facebook/wav2vec2-large-960h-lv60-self` for English. For IndicVoices-R the verbatim transcript is
   aligned when it has the same word count as the normalized one, and the timings are copied word by
   word. Utterances were dropped when the aligner failed, more than 20% of words had no timing, a word
   was longer than 3 s or shorter than 15 ms, or the speech started more than 3 s late or ended more
   than 3 s early.
6. **Loanword rescripting.** Hindi transcripts spell English words in Devanagari (टाइम, स्कूल,
   फ़ोन). Candidates were found by comparing a phonetic romanization of every Devanagari word with
   CMUdict pronunciations of frequent English words, then **reviewed by hand**; the resulting map
   (`loanword_map.tsv`, 678 entries) rewrites those words in Latin script (time, school, phone).
   Only the text changes, the audio is untouched, so the model learns Indian pronunciations of English
   words written in English. Most frequent rewrites: time (467), school (376), phone (354), order
   (325), media (322).

## Files

* `data/*.parquet`: the dataset
* `loanword_map.tsv`: the reviewed Devanagari to Latin map
* `manifest_stats.json`, `tokenizer_stats.json` and the other `*.json` files: counts written by the
  preprocessing scripts (kept and dropped utterances, alignment audit, tokenizer checks)
