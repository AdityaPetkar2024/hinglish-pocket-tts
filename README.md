# Hinglish Pocket TTS

Fine-tuning Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) to speak Hindi-English
code-switched text with a natural Indian accent, from Devanagari, Latin, or mixed-script input.

```
मुझे कल office जाना है, लेकिन मेरी car खराब हो गई है.
I think यह plan काफ़ी अच्छा है, let's start करते हैं.
```

* Model (private): https://huggingface.co/AdityaPetkar2006/pocket-tts-hinglish
* Training data (private): https://huggingface.co/datasets/AdityaPetkar2006/hinglish-tts-data
* Demo video: see the submission email

## Results

40 held-out test sentences (Hindi with one or several English words, English with Hindi words, pure
Hindi, pure English, brand names) x 8 voices = 320 clips per system, transcribed by Whisper large-v3.

| System | Romanized CER | TA-WER | English-word recall | Speaker sim. | UTMOS |
|---|---|---|---|---|---|
| **This model** (extended tokenizer, 5k steps) | **11.5%** [8.9, 14.4] | **10.5%** [7.8, 13.4] | **93.8%** [90.1, 96.9] | 0.925 | 3.59 |
| Kyutai's new-language recipe, 5k steps | 13.0% | 13.6% | 89.6% | 0.926 | 3.63 |
| Extended tokenizer, 10k steps | 12.2% | 12.7% | 93.5% | 0.923 | 3.52 |
| Extended tokenizer, 15k steps | 13.5% | 15.0% | 89.6% | 0.925 | 3.48 |
| Released English model, romanized input | 20.7% | 44.1% | 83.5% | 0.933 | 3.80 |

* 95% bootstrap confidence intervals in brackets. Paired bootstrap against Kyutai's recipe at the same
  step: TA-WER [-5.2, -0.9] points, CER [-2.8, -0.3] points, both entirely below zero.
* **Romanized CER** converts both sides to Roman script first, so writing a word in either script is
  not an error. **TA-WER** (transliteration-aware WER) counts a word as correct in either script.
  **English-word recall** checks that each English word of the input is actually spoken.
* ASR floor: the same pipeline on 120 real recordings of the evaluation speakers gives CER 7.1% and
  TA-WER 19.0%, so part of the remaining error belongs to the recognizer.
* More training hurt (10k, 15k), so the 5k checkpoint is released. Full output:
  `summary.json` in the model repo.

## Data

Three public, human-transcribed corpora, 102.1 h of training speech (41,589 utterances) and 2.9 h
held out (1,269 utterances):

| Dataset | Link | Train | License |
|---|---|---|---|
| IndicTTS Hindi (IIT Madras) | [SPRINGLab/IndicTTS-Hindi](https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi) | 24.4 h, studio read speech | IndicTTS terms |
| IndicVoices-R Hindi (AI4Bharat) | [ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r) | 71.2 h, read and conversational | CC BY 4.0 |
| Svarah (AI4Bharat) | [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | 6.6 h, Indian-accented English | CC BY 4.0 |

Preprocessing (details in the dataset card):

1. Audio to 24 kHz mono FLAC, loudness normalized to -20 dBFS.
2. Text: NFC, danda to ".", terminal punctuation; IndicVoices-R normalized transcripts (what users
   type) for training, verbatim transcripts for alignment.
3. Dropped: clips outside 1 to 30 s, transcripts with digits or unsupported characters.
4. Speaker-disjoint validation split (utterance-level for the 2 IndicTTS speakers).
5. Word-level CTC forced alignment with Kyutai's aligner, plus an audit that drops implausible
   alignments.
6. **Loanword rescripting**: Hindi transcripts spell English words in Devanagari (टाइम, स्कूल). A
   phonetic matcher against CMUdict proposed candidates, which were reviewed by hand into a 678-entry
   map (`loan/loanword_map.tsv`) that rewrites them in Latin script (time, school). The audio is
   unchanged, so 36.9 h of Hindi speech became code-mixed training data in which English words are
   written in English and pronounced the way Indian speakers say them.

## Method

* **Extended tokenizer.** Kyutai's recipe for a new language trains a fresh 4,000-piece tokenizer and
  resets the text embeddings, which throws away everything the model knows about English text. Here
  the released English tokenizer is kept exactly (verified on 6,000 strings) and 1,713 Devanagari
  pieces are appended. English rows of the embedding keep their pretrained values; new rows start
  from the English rows' mean and standard deviation. Both approaches were trained with identical
  settings, and the comparison is the second row of the table.
* **Training.** Pocket TTS training code at commit a922b91 (24-layer model), LSD flow objective,
  lr 2e-4, batch 64, 1k warmup steps, text and voice dropout 0.2, EMA weights, one H100 on Modal.
* **Inference.** Voice cloned from a short prompt; classifier-free guidance coefficient 2,
  temperature 0.3.

## Code

| File | Step |
|---|---|
| `modal_01_inspect.py`, `modal_02_more_data.py` | download and inspect the source datasets |
| `modal_03_prep.py` | audio and text cleaning, filters, splits |
| `loan/find_loanwords.py` | loanword candidates for manual review (`loan/loanword_map.tsv` = result) |
| `modal_04_align.py` | word-level forced alignment and audit |
| `tts_tools.py` | manifests, both tokenizers, embedding transplant, training configs |
| `modal_tts.py` | all later stages on Modal: build, pack, train, evaluate, release, dataset, speak |
| `tts_eval.py`, `metrics.py`, `analyze.py`, `xscript_phon.py`, `eval_sentences.tsv` | evaluation |
| `tts_publish.py` | export the checkpoint, smoke tests, upload model and dataset |
| `infer.py` | inference (command line or Python) |

Reproduce (Modal account, HF token as Modal secret `huggingface`):

```bash
modal run modal_01_inspect.py && modal run modal_02_more_data.py && modal run modal_03_prep.py
modal run modal_04_align.py
modal run modal_tts.py::build
modal run --detach modal_tts.py::train --run extended     # precomputes latents, then trains
modal run modal_tts.py::pack                              # bundle data for fast reads, then train again
modal run --detach modal_tts.py::train --run baseline --max-steps 5000
modal run modal_tts.py::evaluate
modal run modal_tts.py::release
modal run modal_tts.py::dataset
```

## Inference

```bash
export HF_TOKEN=hf_...          # read access to the private model repo
pip install soundfile "git+https://github.com/kyutai-labs/pocket-tts@a922b912dfa366975294afb568d815b305acde5d"
python infer.py --cfg 1 --voice reference.wav --out out.wav \
    --text "आज की meeting में हम new project का budget discuss करेंगे."
```

`--cfg 2` (guidance, the evaluated setting) also needs Kyutai's training code: clone
https://github.com/kyutai-labs/pocket-tts at commit a922b91, `uv sync`, and run `infer.py` from
inside it. Or on Modal: `modal run modal_tts.py::speak --file demo.txt --voice me.wav`.

## Limitations

* Hindi words should be in Devanagari; fully romanized Hindi was not part of training or evaluation.
* Write numbers as words (digits were filtered out of training).
* Weakest category: Hindi sentences with several English words (CER about 19%).
* UTMOS is trained on English; no human listening test was run.
