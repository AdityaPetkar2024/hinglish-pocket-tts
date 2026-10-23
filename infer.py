"""Hinglish text-to-speech with the fine-tuned Pocket TTS model.

Text can be Devanagari, Roman (English words), or mixed:
    मुझे कल office जाना है, लेकिन मेरी car खराब हो गई है.

Two sampling paths, same weights:
  --cfg 1   the stock `pocket_tts` package (pip install pocket-tts). Runs on CPU, streams.
  --cfg 2   classifier-free guidance (coefficient 2), as used for every number in the evaluation.
            Needs Kyutai's training code: clone https://github.com/kyutai-labs/pocket-tts at commit
            a922b912 and run from inside that folder (or put it on PYTHONPATH). GPU recommended.

The model repo is private, so set a token with read access first:
    export HF_TOKEN=hf_...
    python infer.py --text "आज की meeting में हम new project का budget discuss करेंगे." \
                    --voice my_voice.wav --out out.wav --cfg 2

--voice is a 4-10 s clean recording of the target speaker (wav/flac/mp3). Speaker identity, accent
and recording conditions are copied from it.
"""
import argparse, os, sys, unicodedata

DEFAULT_REPO = "AdityaPetkar2006/pocket-tts-hinglish"
TEMP, EOS = 0.3, -1.0            # the settings used in the evaluation
# Text is cut into sentence chunks of at most this many tokens. pocket-tts uses 50 for English; a
# Hindi sentence needs more tokens for the same duration, and the model was trained on utterances
# of up to 30 s, so whole sentences are kept together (the evaluation used whole sentences).
MAX_TOKENS = 120
MAXF = 375                        # 30 s per chunk, the longest training utterance
_cache = {}


def _config(source: str) -> str:
    if os.path.isdir(source):     # an exported folder (pocket_tts_hinglish_local.yaml points at it)
        return os.path.join(source, "pocket_tts_hinglish_local.yaml")
    return f"hf://{source}/pocket_tts_hinglish.yaml"


def _clean(text: str) -> str:
    """Same normalisation as the training transcripts: NFC, no zero-width joiners."""
    text = unicodedata.normalize("NFC", text).replace("‌", "").replace("‍", "")
    return " ".join(text.split())


def load(source: str = DEFAULT_REPO, cfg: float = 2.0, device: str = "cpu"):
    from pocket_tts import TTSModel
    key = (source, cfg != 1.0, device)
    if key not in _cache:
        tts = TTSModel.load_model(config=_config(source), temp=TEMP, eos_threshold=EOS)
        if cfg != 1.0:
            from training.modules.model import TrainableTTS
            from training.modules.samplers import build_flow
            model = TrainableTTS(tts.flow_lm, build_flow("lsd"), args=None)
            model.to(device).eval()
            tts.mimi.to(device)
            _cache[key] = (tts, model)
        else:
            _cache[key] = (tts, None)
    return _cache[key]


def synthesize(text: str, voice: str, cfg: float = 2.0, source: str = DEFAULT_REPO, device: str | None = None,
               seed: int = 0):
    """Returns (waveform as float32 numpy array, sample rate)."""
    import numpy as np, torch
    torch.manual_seed(seed)
    text = _clean(text)
    if cfg == 1.0:
        tts, _ = load(source, 1.0, "cpu")
        state = tts.get_state_for_audio_prompt(voice)
        wav = tts.generate_audio(state, text, max_tokens=MAX_TOKENS)
        return wav.reshape(-1).float().numpy(), tts.sample_rate

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    from pocket_tts.models.text_chunking import prepare_text_prompt, split_into_best_sentences
    from training.eval.librispeech import latents_to_wav, load_mono
    tts, model = load(source, cfg, device)
    tok = tts.flow_lm.conditioner.tokenizer
    mimi = tts.mimi
    opts = (tts.pad_with_spaces_for_short_inputs, tts.remove_semicolons, tts.append_terminal_punctuation,
            tts.capitalize_first_letter, tts.replace_characters)
    with torch.no_grad():
        v = mimi.encode_to_latent(load_mono(voice, mimi.sample_rate)[None, None].to(device))[0]
        pieces = []
        for chunk in split_into_best_sentences(tok, text, MAX_TOKENS, *opts):
            prompt, _ = prepare_text_prompt(chunk, *opts)
            ids = torch.tensor(tok.encode(prompt), dtype=torch.long)
            lat = model.generate([ids], [v], max_frames=MAXF, temp=TEMP, n_steps=1, cfg_coef=cfg,
                                 eos_threshold=EOS)[0]
            w = latents_to_wav(mimi, lat, device)
            if w is not None:
                pieces += [w.float().cpu().numpy(), np.zeros(int(0.12 * mimi.sample_rate), np.float32)]
    if not pieces:
        raise RuntimeError("the model produced no audio for this text")
    return np.concatenate(pieces[:-1]), int(mimi.sample_rate)


def main():
    global TEMP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="text to speak (or use --file)")
    ap.add_argument("--file", help="UTF-8 text file, one utterance per line -> out_001.wav, out_002.wav, ...")
    ap.add_argument("--voice", required=True, help="reference recording of the speaker to clone")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--cfg", type=float, default=2.0, help="1 = stock pocket_tts sampler, 2 = guided (default)")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF model repo or a local exported folder")
    ap.add_argument("--seed", type=int, default=0, help="different seeds give different takes")
    ap.add_argument("--temp", type=float, default=0.3, help="0.3 = evaluated; lower is steadier, higher livelier")
    a = ap.parse_args()
    TEMP = a.temp
    if not (a.text or a.file):
        ap.error("give --text or --file")
    import soundfile as sf
    lines = [a.text] if a.text else [l for l in open(a.file, encoding="utf-8").read().splitlines() if l.strip()]
    stem, ext = os.path.splitext(a.out)
    for i, line in enumerate(lines, 1):
        wav, sr = synthesize(line, a.voice, cfg=a.cfg, source=a.repo, seed=a.seed)
        path = a.out if len(lines) == 1 else f"{stem}_{i:03d}{ext or '.wav'}"
        sf.write(path, wav, sr)
        print(f"{path}  {len(wav) / sr:.1f}s  {line}", file=sys.stderr)


if __name__ == "__main__":
    main()
