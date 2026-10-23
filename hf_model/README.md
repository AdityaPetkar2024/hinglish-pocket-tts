---
language:
- hi
- en
library_name: pocket-tts
pipeline_tag: text-to-speech
base_model: kyutai/pocket-tts
tags:
- hinglish
- code-switching
- text-to-speech
- voice-cloning
---

# Pocket TTS Hinglish

A fine-tune of Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) (24-layer English model,
316M trainable parameters in the language model) that speaks Hindi-English code-switched text with an
Indian accent. Input can be Devanagari, Roman, or both in the same sentence:

```
मुझे कल office जाना है, लेकिन मेरी car खराब हो गई है.
I think यह plan काफ़ी अच्छा है, let's start करते हैं.
```

The voice (speaker, accent, recording conditions) is cloned from a short reference recording.

## Files

| File | What it is |
|---|---|
| `model.safetensors` | flow language model + Mimi codec, EMA weights of step 5,000 |
| `tokenizer.json` | the original English tokenizer with Devanagari pieces appended (5,713 tokens) |
| `pocket_tts_hinglish.yaml` | config for the `pocket_tts` package, points at the two files above |
| `infer.py` | command-line and Python inference (both sampling modes below) |
| `summary.json` | full evaluation output (all systems, per category, confidence intervals) |

## Usage

The repo is private: set a token with read access first (`export HF_TOKEN=hf_...`).

**Stock pocket-tts package (CPU is fine, no guidance).** Install the pinned version:
`pip install "git+https://github.com/kyutai-labs/pocket-tts@a922b912dfa366975294afb568d815b305acde5d"`

```python
from pocket_tts import TTSModel
model = TTSModel.load_model(config="hf://AdityaPetkar2006/pocket-tts-hinglish/pocket_tts_hinglish.yaml")
voice = model.get_state_for_audio_prompt("reference.wav")      # 4 to 10 s of clean speech
audio = model.generate_audio(voice, "आज की meeting में हम new project का budget discuss करेंगे.", max_tokens=120)
```

**With classifier-free guidance (coefficient 2, the setting all numbers below were measured with):**

```bash
git clone https://github.com/kyutai-labs/pocket-tts && cd pocket-tts
git checkout a922b912dfa366975294afb568d815b305acde5d && uv sync
uv pip install soundfile
uv run python /path/to/infer.py --cfg 2 --voice reference.wav --out out.wav \
    --text "Please अपना password किसी के साथ share मत करना."
```

`infer.py --cfg 1` uses the stock package path instead.

## How it was trained

* **Data.** 102.1 hours of training speech (41,589 utterances) plus 2.9 hours held out (1,269
  utterances), all public and human-transcribed:

  | Source | Train | Of which code-mixed |
  |---|---|---|
  | IndicTTS Hindi (IIT Madras), studio read speech | 24.4 h | 5.5 h |
  | IndicVoices-R Hindi (AI4Bharat), read and conversational | 71.2 h | 31.4 h |
  | Svarah (AI4Bharat), Indian-accented English | 6.6 h | |

  36.9 hours of the training audio contain at least one English word in Latin script. The exact
  training set is in the private dataset repo `AdityaPetkar2006/hinglish-tts-data`, with the
  preprocessing described in its card.
* **Code-switching from Hindi data.** Hindi transcripts write English words in Devanagari
  (ऑफिस, मीटिंग). A reviewed map of 678 such words rewrites them in Latin script (office, meeting)
  while the audio stays the same, so the model learns how Indian speakers pronounce English words
  inside Hindi sentences when they are typed in Latin script. Rarer English words that were not in
  the map stayed in Devanagari, so the model has also seen English words written that way.
* **Tokenizer.** Instead of training a new tokenizer from scratch (Kyutai's recipe for a new
  language), the released English tokenizer is kept unchanged and Devanagari pieces are appended.
  English text tokenizes exactly as before (checked on 6,000 strings), and the English embedding
  rows keep their pretrained values; the 1,713 new rows start from the statistics of the English
  rows. On held-out English text this needs 1.79 tokens per word against 2.24 for a fresh 4,000-piece
  tokenizer, at a small cost for Hindi (1.50 against 1.41).
* **Alignment.** Word timings by CTC forced alignment (needed by the Pocket TTS training loader);
  utterances with implausible alignments were dropped.
* **Training.** Pocket TTS training code at commit a922b91, LSD flow objective, lr 2e-4,
  batch 64, 1,000 warmup steps, text and voice dropout 0.2, on one H100 via Modal. Checkpoints were
  kept every 2,500 steps and the final one was chosen on the evaluation below, not on training loss.

## Evaluation

40 test sentences (none appear in the training text) in 6 categories: Hindi with one English word,
Hindi with several English words, English-matrix sentences with Hindi words, pure Hindi, pure
English, brand names. Each is spoken in 8 voices taken from the validation split, 320 clips per
system: 4 IndicVoices-R and 2 Svarah speakers never seen in training, and the 2 IndicTTS speakers
(whose prompt recordings are held out, though the speakers are in training). Speech is transcribed with Whisper large-v3.

* **Romanized CER**: both transcript and reference converted to Roman script (ITRANS) before
  character error rate, so writing an English word in Devanagari or Latin script does not count as
  an error.
* **TA-WER**: word error rate that treats a word in either script as correct if it is the same word.
* **English-word recall**: share of the English words in the input that are heard in the output
  (matched across scripts by spelling skeleton and CMUdict pronunciation).
* **Speaker similarity**: WavLM x-vector cosine to a different recording of the reference speaker.
* **UTMOS**: automatic naturalness score (trained on English, so indicative only).

| System | Romanized CER | TA-WER | English-word recall | Speaker sim. | UTMOS |
|---|---|---|---|---|---|
| **This model** (extended tokenizer, 5k steps) | **11.5%** [8.9, 14.4] | **10.5%** [7.8, 13.4] | **93.8%** [90.1, 96.9] | 0.925 | 3.59 |
| Kyutai recipe (new 4k tokenizer), 5k steps | 13.0% | 13.6% | 89.6% | 0.926 | 3.63 |
| Extended tokenizer, 10k steps | 12.2% | 12.7% | 93.5% | 0.923 | 3.52 |
| Extended tokenizer, 15k steps | 13.5% | 15.0% | 89.6% | 0.925 | 3.48 |
| Released English model, romanized input | 20.7% | 44.1% | 83.5% | 0.933 | 3.80 |

95% bootstrap confidence intervals over sentences in brackets. Paired bootstrap, this model minus the
Kyutai recipe at the same step: TA-WER difference [-5.2, -0.9] points, CER difference [-2.8, -0.3]
points, both intervals entirely below zero. No silent outputs and no generations that failed to stop,
in any system.

For reference, the same ASR on 120 real recordings of the evaluation speakers gives CER 7.1% and TA-WER
19.0%: part of the remaining error is the recognizer (for example Hindi spelling variants such as
जाओंगा for जाऊँगा), so these numbers are upper bounds on the model's own error.

Longer training made things worse (10k and 15k steps), consistent with overfitting a dataset of this
size, which is why the 5k checkpoint is released.

## Limitations

* Hindi words should be written in Devanagari. Fully romanized Hindi ("kal main office jaunga") is
  outside the training distribution and was not evaluated.
* Digits were removed from training. Write numbers as words (`तीन सौ`, `three hundred`).
* The weakest category is Hindi sentences with several English words (CER about 19%).
* UTMOS is an English-trained predictor; no listening test was run.
* The voice prompt matters: noisy or reverberant prompts are reproduced faithfully.
* Do not use it to imitate a real person without their consent.

## License

Base model and training code: Kyutai Pocket TTS (see its repository for terms). Training data:
see the dataset card; each source keeps its own license.
