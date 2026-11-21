"""
voicelab_utils.py  --  starter helpers for the VoiceLab assignment (AIML ZG520).

These helpers only cover the routine work: reading your config, loading audio, auditing your
recordings, fixing random seeds and writing results.json. The speech processing code in the
parts marked [Critical] must be written by you.
"""
import json
import os
import random

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


def set_seed(cfg):
    seed = int(cfg["speaker_seed"])
    random.seed(seed)
    np.random.seed(seed)
    return seed


def load_wav(path, sr=None):
    """Read a wav file as mono float32 in [-1, 1]. Resample to sr if given."""
    fs, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr is not None and fs != sr:
        g = np.gcd(int(fs), int(sr))
        x = resample_poly(x, int(sr) // g, int(fs) // g).astype(np.float32)
        fs = sr
    return x, fs


def audit_file(path, frame_ms=25):
    """Duration, peak, clipping %, estimated SNR (dB) and silence fraction of one file."""
    x, fs = load_wav(path)
    n = max(int(fs * frame_ms / 1000), 1)
    m = len(x) // n
    frames = x[: m * n].reshape(m, n)
    e = (frames ** 2).mean(axis=1) + 1e-12
    srt = np.sort(e)
    noise = srt[: max(int(0.1 * m), 1)].mean()
    speech = srt[int(0.7 * m):].mean() if m > 3 else srt.mean()
    snr = 10 * np.log10(speech / noise)
    silence = float((e < 10 * noise).mean())
    return {"file": os.path.basename(path), "duration_s": round(len(x) / fs, 2),
            "peak": round(float(np.abs(x).max()), 3),
            "clipping_pct": round(float((np.abs(x) > 0.99).mean() * 100), 3),
            "snr_db": round(float(snr), 1), "silence_frac": round(silence, 2)}


def audit_folder(folder, skip=("noisy", "blocks", "original")):
    """Audit every wav under folder, except the long blocks, the originals and the noisy mixtures."""
    rows = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in skip)
        for fn in sorted(files):
            if fn.lower().endswith(".wav"):
                rows.append(audit_file(os.path.join(root, fn)))
    return rows


def print_table(rows):
    if not rows:
        print("no rows")
        return
    keys = list(rows[0])
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))


def mix_at_snr(x, noise, snr_db):
    """Return x + g*n with exact SNR over the active region of x (Part 6, item 4 of the brief)."""
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    noise = noise[: len(x)]
    active = np.abs(x) > 0.1 * np.abs(x).max()
    px = (x[active] ** 2).mean()
    pn = (noise ** 2).mean() + 1e-12
    g = np.sqrt(px / (pn * 10 ** (snr_db / 10)))
    return x + g * noise


class Results:
    """Collects check values and writes results.json (merges with what is already there)."""

    def __init__(self, path="results.json"):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)

    def set(self, key, value):
        self.data[key] = float(value) if isinstance(value, (int, float, np.floating)) else value
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)


def plot_wave_spec(x, fs, title="", n_fft=512, hop=128):
    """Quick waveform plus log spectrogram. Replace with your own STFT in Part 1 if you wish."""
    import matplotlib.pyplot as plt
    from scipy.signal import stft
    t = np.arange(len(x)) / fs
    f, tt, Z = stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(t, x, lw=0.6, color="black")
    ax[0].set_ylabel("amplitude")
    ax[0].set_title(title)
    ax[1].pcolormesh(tt, f, 20 * np.log10(np.abs(Z) + 1e-8), cmap="gray_r", shading="auto")
    ax[1].set_ylabel("Hz")
    ax[1].set_xlabel("time (s)")
    plt.tight_layout()
    return fig


_MODELS = {}
# English, plain transcription: no language guessing, and no 'forced_decoder_ids' notice.
WHISPER_EN = {"language": "en", "task": "transcribe"}


def get_device():
    """'cuda' when a GPU is available, else 'cpu'. Set VOICELAB_DEVICE=cpu (or run a command with --cpu)
    to force the CPU. The results are the same up to rounding; a GPU is only faster."""
    if os.environ.get("VOICELAB_DEVICE", "").lower() == "cpu":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _asr(model):
    """Load a Whisper pipeline once and reuse it."""
    dev = get_device()
    if ("asr", model, dev) not in _MODELS:
        from transformers import pipeline
        try:                      # hide harmless deprecation notices from the library
            from transformers.utils import logging as _hf_log
            _hf_log.set_verbosity_error()
        except Exception:
            pass
        _MODELS[("asr", model, dev)] = pipeline("automatic-speech-recognition", model=model,
                                                device=0 if dev == "cuda" else -1)
    return _MODELS[("asr", model, dev)]


def first_seconds_transcript(path, seconds=8, model="openai/whisper-base"):
    """Transcribe the first seconds of a file with Whisper to check that your code is spoken."""
    x, fs = load_wav(path, sr=16000)
    return _asr(model)({"raw": x[: int(16000 * seconds)], "sampling_rate": 16000},
                       generate_kwargs=WHISPER_EN)["text"]


# ---------------------------------------------------------------- model helpers (need internet and the libraries)
def wav2vec2_logprobs(path):
    """Frame-level log posteriors from wav2vec2-base-960h. Returns (T x V array, list of tokens)."""
    import torch
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    dev = get_device()
    if ("w2v", dev) not in _MODELS:
        _MODELS[("w2v", dev)] = (Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h"),
                                 Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").eval().to(dev))
    proc, model = _MODELS[("w2v", dev)]
    x, _ = load_wav(path, sr=16000)
    inp = proc(x, sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        logits = model(inp.input_values.to(dev)).logits[0].float().cpu()
    vocab = proc.tokenizer.convert_ids_to_tokens(list(range(logits.shape[-1])))
    return torch.log_softmax(logits, dim=-1).numpy(), vocab


def whisper_transcribe(path, size="base"):
    x, _ = load_wav(path, sr=16000)
    return _asr(f"openai/whisper-{size}")({"raw": x, "sampling_rate": 16000},
                                          generate_kwargs=WHISPER_EN)["text"].strip()


def tts_synthesise(text, pipeline_name):
    """Pretrained Tacotron2 + WaveRNN from torchaudio. Returns (waveform array, sample rate)."""
    import torch
    import torchaudio
    bundle = getattr(torchaudio.pipelines, pipeline_name)
    processor = bundle.get_text_processor()
    dev = get_device()
    tacotron2 = bundle.get_tacotron2().eval().to(dev)
    vocoder = bundle.get_vocoder().eval().to(dev)
    with torch.inference_mode():
        processed, lengths = processor(text)
        spec, spec_lengths, _ = tacotron2.infer(processed.to(dev), lengths.to(dev))
        wav, _ = vocoder(spec, spec_lengths)
    rate = getattr(vocoder, "sample_rate", None) or getattr(bundle, "sample_rate", 22050)
    return wav[0].float().cpu().numpy(), int(rate)


def stoi_score(reference, degraded, fs):
    from pystoi import stoi
    n = min(len(reference), len(degraded))
    return float(stoi(reference[:n], degraded[:n], fs, extended=False))


def to_16k_mono(src, dst):
    """Convert a phone recording (wav, or m4a/mp3 if ffmpeg is installed) to mono 16 bit 16 kHz WAV.
    Keep the original file untouched in data/original/."""
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000",
                        "-sample_fmt", "s16", dst], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        import soundfile as sf
        x, fs = sf.read(src)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = resample_poly(x, 16000, fs) if fs != 16000 else x
        sf.write(dst, x, 16000, subtype="PCM_16")


# ---------------------------------------------------------------- plumbing helpers (no speech processing in here)
def savefig(name, folder="figures", dpi=150):
    """Save the current matplotlib figure as figures/<name>.png (the report builder looks there)."""
    import matplotlib.pyplot as plt
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name if name.endswith(".png") else name + ".png")
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def compare(label, mine, reference, tol=1e-4):
    """Print PASS or FAIL for your value against a library value (verification only)."""
    a, b = np.asarray(mine, dtype=float), np.asarray(reference, dtype=float)
    ok = bool(np.allclose(a, b, rtol=tol, atol=1e-9))
    err = float(np.max(np.abs(a - b) / (np.abs(b) + 1e-12))) if a.shape == b.shape else float("nan")
    print(f"{'PASS' if ok else 'FAIL'}  {label}  (largest relative difference {err:.2e}, tolerance {tol})")
    return ok


def brown_sentences(lowercase=True):
    """The Brown corpus (NLTK) as a list of token lists, words only. Choose and state your own split."""
    import nltk
    try:
        from nltk.corpus import brown
        brown.words()
    except LookupError:
        nltk.download("brown", quiet=True)
        from nltk.corpus import brown
    out = []
    for sent in brown.sents():
        toks = [w.lower() if lowercase else w for w in sent if w.isalpha() or "'" in w]
        if toks:
            out.append(toks)
    return out


def cmudict_dict():
    """CMUdict as {word: [phone lists]}. Words not in it need the G2P fallback (g2p_en)."""
    import nltk
    try:
        from nltk.corpus import cmudict
        d = cmudict.dict()
    except LookupError:
        nltk.download("cmudict", quiet=True)
        from nltk.corpus import cmudict
        d = cmudict.dict()
    return d


def load_transcripts(path="cache/transcripts.json"):
    """Whisper transcripts saved by 'python voicelab.py precompute' (key: clip name, e.g. sent_0_clean)."""
    with open(path) as f:
        return json.load(f)


def load_posteriors(name, folder="cache"):
    """wav2vec2 log posteriors saved by 'python voicelab.py precompute'. Returns (T x V array, tokens)."""
    logp = np.load(os.path.join(folder, f"w2v_{name}.npy"))
    with open(os.path.join(folder, "w2v_tokens.json")) as f:
        tokens = json.load(f)
    return logp, tokens


# ---------------------------------------------------------------- recording blocks
def recording_script(cfg, path="recording_script.txt"):
    """Write and print the read-aloud script for your three recording blocks."""
    code = " ".join(cfg["code"])
    lines = ["BLOCK 1: WORDS  (record as data/original/words_block.wav)",
             "Silence 1 s, say the code, pause 2 s. Then each line below: say the word once, pause 2 s.",
             f"  CODE: {code}"]
    for w in cfg["vocab"]:
        for r in range(3):
            lines.append(f"  {w}   (repetition {r})")
    lines += ["", "BLOCK 2: SENTENCES  (record as data/original/sentences_block.wav)",
              "Silence 1 s, say the code, pause 2 s. Then each sentence once, pause 2 s after each.",
              f"  CODE: {code}"]
    for i, t in enumerate(cfg["sentences_spoken"]):
        lines.append(f"  sentence {i}: {t}")
    lines += ["", "BLOCK 3: ROOM NOISE  (record as data/original/noise_block.wav)",
              "Silence 1 s, say the code, then stay completely silent for about 11 s and stop.",
              f"  CODE: {code}"]
    text = "\n".join(lines)
    open(path, "w").write(text)
    print(text)
    return text


def find_segments(x, fs, min_silence_s=1.2, min_len_s=0.15, margin_db=12.0, pad_s=0.12):
    """Split a recording into speech segments separated by pauses of at least min_silence_s."""
    hop = int(0.010 * fs)
    win = int(0.020 * fs)
    m = max((len(x) - win) // hop, 1)
    e = np.array([(x[i * hop: i * hop + win] ** 2).mean() for i in range(m)]) + 1e-12
    db = 10 * np.log10(e)
    thr = np.percentile(db, 10) + margin_db
    act = db > thr
    segs, start, last = [], None, None
    gap = int(min_silence_s / 0.010)
    for i, a in enumerate(act):
        if a:
            if start is None:
                start = i
            last = i
        elif start is not None and i - last > gap:
            segs.append((start, last))
            start = None
    if start is not None:
        segs.append((start, last))
    out = []
    for s0, s1 in segs:
        if (s1 - s0) * 0.010 >= min_len_s:
            out.append((max(int(s0 * hop - pad_s * fs), 0), min(int(s1 * hop + win + pad_s * fs), len(x))))
    return out


def _gaps(segs):
    return [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]


def split_block(src, names, out_dir, min_silence_s=None, margin_db=None, verbose=True):
    """Cut a block recording (16 kHz mono wav) into clips, with automatic repair.

    The first segment is your spoken code, the next len(names) segments are your clips in script
    order. The function
      1. tries several pause and loudness settings until it finds exactly 1 + len(names) segments;
      2. if none works but there are too many segments (a cough, a breath, a long pause inside a
         sentence), keeps the 1 + len(names) segments separated by the longest silences and says so;
      3. if there are too few (words run together), stops and tells you what to fix;
      4. prints a table of every clip with its length and any warning, and saves split_report.csv.
    Give min_silence_s and margin_db to force one setting instead of the automatic search."""
    import csv
    import soundfile as sf
    x, fs = load_wav(src, sr=16000)
    need = 1 + len(names)
    if min_silence_s is not None or margin_db is not None:
        grid = [(min_silence_s or 1.2, margin_db or 12.0)]
    else:
        grid = [(m, d) for d in (12.0, 9.0, 15.0, 18.0, 6.0) for m in (1.2, 1.0, 0.8, 1.5, 2.0, 0.6)]
    segs, note = None, ""
    for m, d in grid:
        cand = find_segments(x, fs, min_silence_s=m, margin_db=d)
        if len(cand) == need:
            segs = cand
            if (m, d) != (1.2, 12.0):
                note = f"settings adjusted automatically (min_silence_s={m}, margin_db={d})"
            break
    if segs is None:
        best = None
        for d in (12.0, 9.0, 15.0, 18.0, 6.0):
            cand = find_segments(x, fs, min_silence_s=0.3, margin_db=d)
            if len(cand) >= need:
                best = cand
                break
        if best is None:
            top = max(len(find_segments(x, fs, min_silence_s=0.3, margin_db=d)) for d in (6.0, 9.0, 12.0, 15.0, 18.0))
            raise ValueError(
                f"Found at most {top} separate pieces of speech but need {need} (the code plus "
                f"{len(names)} clips). Either an item was skipped, or words or sentences are running "
                f"together because the pauses between them are too short. Record the block again, "
                f"saying every item and pausing at least 2 seconds after each.")
        gaps = _gaps(best)
        cut = sorted(np.argsort(gaps)[-(need - 1):])
        segs, start = [], 0
        for c in cut:
            segs.append((best[start][0], best[c][1]))
            start = c + 1
        segs.append((best[start][0], best[-1][1]))
        note = (f"repaired automatically: found {len(best)} pieces of speech, joined the pieces "
                f"separated by the shortest pauses to get {need}. Check the flagged clips.")
    rows, flags = [], 0
    code_len = (segs[0][1] - segs[0][0]) / fs
    lens = np.array([(b - a) / fs for a, b in segs[1:]])
    med = float(np.median(lens)) if len(lens) else 0.0
    if not 1.5 <= code_len <= 7.0:
        flags += 1
    os.makedirs(out_dir, exist_ok=True)
    # the spoken code goes next to the block, not among the clips (so data/words holds exactly your 30 clips)
    sf.write(os.path.join(os.path.dirname(src) or ".", "code_" + os.path.basename(src)),
             x[segs[0][0]:segs[0][1]], fs, subtype="PCM_16")
    for (a0, b0), nm, ln in zip(segs[1:], names, lens):
        warn = ""
        if ln < 0.2:
            warn = "very short"
        elif len(names) > 6 and ln > max(2.5 * med, 1.0):
            warn = "much longer than the others"
        elif len(names) > 6 and ln < 0.4 * med and ln < 0.4:
            warn = "much shorter than the others"
        if warn:
            flags += 1
        sf.write(os.path.join(out_dir, nm), x[a0:b0], fs, subtype="PCM_16")
        rows.append({"clip": nm, "start_s": round(a0 / fs, 2), "length_s": round(ln, 2), "warning": warn})
    with open(os.path.join(out_dir, "split_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "start_s", "length_s", "warning"])
        w.writeheader()
        w.writerows(rows)
    if verbose:
        print(f"{os.path.basename(src)}: wrote {len(names)} clips to {out_dir}. Code length {code_len:.1f} s.")
        if note:
            print("Note:", note)
        if not 1.5 <= code_len <= 7.0:
            print(f"WARNING: the code part is {code_len:.1f} s long (expected about 2 to 5 s). Check the start of the block.")
        for r in rows:
            if r["warning"]:
                print(f"WARNING: {r['clip']} ({r['length_s']} s) is {r['warning']}. Listen to it.")
        if not flags and not note:
            print("All clips look normal.")
    return rows


def split_noise_block(src, out_path, margin_s=0.7, min_len_s=8.0):
    """Keep the silent part of the noise block: everything after your spoken code."""
    import soundfile as sf
    x, fs = load_wav(src, sr=16000)
    segs = find_segments(x, fs, min_silence_s=1.2)
    if not segs:
        raise ValueError("no spoken code found at the start of the noise block")
    start = segs[0][1] + int(margin_s * fs)
    noise = x[start: len(x) - int(0.3 * fs)]
    if len(noise) < min_len_s * fs:
        raise ValueError(f"only {len(noise) / fs:.1f} s of noise found, need at least {min_len_s} s")
    sf.write(out_path, noise, fs, subtype="PCM_16")
    return len(noise) / fs
