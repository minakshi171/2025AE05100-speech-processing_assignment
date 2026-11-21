"""
voicelab.py  --  one command-line tool for the routine work of the VoiceLab assignment.

Run it from your project folder (the one with your <ROLLNO>_config.json). In Colab put '!' in front.

    python voicelab.py setup            first-time set-up: libraries, folders, git, notebook name, corpora
    python voicelab.py script           print your read-aloud recording script
    python voicelab.py data             recordings -> 16 kHz blocks -> clips -> audit -> spoken-code check
    python voicelab.py precompute       noisy sentences, wav2vec2 posteriors and Whisper text, saved in cache/
    python voicelab.py toy              print your hand-calculation numbers in a readable layout
    python voicelab.py save "message"   git add and commit everything
    python voicelab.py log 2 "text"     add a line to the Part 2 entry of design_log.md
    python voicelab.py ai               fill in ai_statement.md by answering four questions
    python voicelab.py handwork         photos in handwork/ -> handwork.pdf (ordered, compressed)
    python voicelab.py report           report.md -> report.pdf (config and results tables filled in)
    python voicelab.py package          final checks, then <ROLLNO>_voicelab.zip
    python voicelab.py progress         what is done, what to do next  (add --summary for one line per part)

None of this does the speech processing for you: the [Critical] parts and every hand calculation
remain yours. Nothing here checks whether your answers are right or reveals an expected value.
"""
import argparse
import csv
import datetime
import glob
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
import zipfile

# Harmless library chatter (for example triton's "Failed to find CUDA") is hidden. Nothing here needs a GPU.
warnings.filterwarnings("ignore", module="triton")
warnings.filterwarnings("ignore", message=".*CUDA.*")
warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LIBS = {  # import name -> pip name
    "numpy": "numpy", "scipy": "scipy", "matplotlib": "matplotlib", "soundfile": "soundfile",
    "nltk": "nltk", "jiwer": "jiwer", "g2p_en": "g2p-en", "librosa": "librosa", "pystoi": "pystoi",
    "torch": "torch", "torchaudio": "torchaudio", "transformers": "transformers",
    "PIL": "pillow", "markdown": "markdown", "pypdf": "pypdf",
}
FOLDERS = ["data/original", "data/blocks", "data/words", "data/sentences", "data/noise",
           "figures", "handwork"]
REQUIRED = ["config.json", "voicelab.ipynb", "run_all.sh", "design_log.md", "ai_statement.md",
            "results.json", "report.pdf", "handwork.pdf"]
ALSO_PACK = ["voicelab_utils.py", "wfst_starter.py", "voicelab.py", "progress.py", "report.md",
             "recording_script.txt"]
SKIP_DIRS = {"cache", "data/noisy", "__pycache__", ".ipynb_checkpoints", ".venv", "venv"}
SKIP_FILES = {".DS_Store"}
NUMWORDS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "to": "2", "too": "2", "three": "3",
            "four": "4", "for": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}


# ---------------------------------------------------------------- small helpers
def say(msg=""):
    print(msg, flush=True)


def find_config():
    """You download <ROLLNO>_config.json. The kit works with a copy called config.json."""
    if os.path.exists("config.json"):
        return
    found = sorted(glob.glob("*_config.json"))
    if len(found) == 1:
        shutil.copyfile(found[0], "config.json")
        say(f"Copied {found[0]} to config.json (the name the kit uses).")
    elif len(found) > 1:
        sys.exit("More than one *_config.json here: " + ", ".join(found) +
                 ". Keep only the file with your own roll number. Using any other configuration is an integrity breach.")


def load_cfg():
    find_config()
    if not os.path.exists("config.json"):
        sys.exit("Your <ROLLNO>_config.json was not found here. Run this from your project folder, next to that file.")
    with open("config.json") as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def have_git():
    return shutil.which("git") is not None


def ensure_git_identity():
    for key, val in (("user.name", "student"), ("user.email", "student@example.com")):
        if not run(["git", "config", key]).stdout.strip():
            run(["git", "config", key, val])


def audio_duration(path):
    import soundfile as sf
    try:
        return sf.info(path).duration
    except Exception:
        out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                   "default=nw=1:nk=1", path]).stdout.strip()
        return float(out) if out else 0.0


# ---------------------------------------------------------------- setup
def gpu_note():
    """One line on the GPU. The assignment runs on a CPU; a GPU only makes the model steps faster."""
    has_card = bool(shutil.which("nvidia-smi"))
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU: PyTorch can use {torch.cuda.get_device_name(0)}. It is used automatically for the model steps (add --cpu to force the CPU). Not required."
        built = getattr(torch.version, "cuda", None)
    except Exception:
        return "GPU: not required. Everything in this assignment runs on a CPU."
    if has_card and not built:
        return ("GPU: an NVIDIA card was found, but the PyTorch installed here is the CPU-only build (pip installs that "
                "on Windows by default), so it runs on the CPU. This is fine: a GPU is not required. If you want the "
                "speed-up, install the CUDA build from pytorch.org and rerun set-up.")
    if has_card:
        return ("GPU: an NVIDIA card was found, but PyTorch cannot use it (driver or CUDA version mismatch), so it runs "
                "on the CPU. This is fine: a GPU is not required.")
    return "GPU: none found. This is fine: a GPU is not required, everything runs on a CPU."


IGNORABLE = """Messages you can safely ignore
  - "Failed to find CUDA" / triton warnings: the GPU is optional; the CPU is enough.
  - "script jiwer.exe is installed in ... which is not on PATH": the kit never calls that program directly.
  - "A new release of pip is available": no need to upgrade.
  - "Device set to use cuda:0" (or cpu): just tells you where the model runs. Both are fine.
  - Whisper or transformers notices about "deprecated", "forced_decoder_ids" or "language detection": harmless.
  - Any other UserWarning or FutureWarning printed by a library: the run is fine unless it stops with an Error."""


def cmd_setup(a):
    cfg = load_cfg()
    say(f"VoiceLab set-up for roll {cfg.get('roll', '?')}")
    say(f"Python {sys.version.split()[0]}")
    libs = dict(LIBS)
    missing = []
    for mod, pipname in libs.items():
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pipname)
    if missing and not a.no_install:
        say("Installing: " + " ".join(missing))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-warn-script-location",
                        "--disable-pip-version-check", *missing])
        if "g2p-en" in missing:       # g2p-en pulls in a dependency that sometimes fails to build
            try:
                importlib.import_module("g2p_en")
            except Exception:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-warn-script-location",
                                "--disable-pip-version-check", "--no-deps", "g2p-en", "inflect"])
        still = []
        for mod, pipname in libs.items():
            try:
                importlib.import_module(mod)
            except Exception:
                still.append(pipname)
        missing = still
    say("Libraries: " + ("all present" if not missing else "still missing: " + " ".join(missing)))
    say(gpu_note())
    say("ffmpeg: " + ("found" if shutil.which("ffmpeg") else
                      "not found (needed only for m4a/mp3 recordings; record WAV or install ffmpeg)"))
    for d in FOLDERS:
        os.makedirs(d, exist_ok=True)
    say("Folders created: " + ", ".join(FOLDERS))

    stamp = ".config.sha256"
    if os.path.exists(stamp):
        same = open(stamp).read().strip() == sha("config.json")
        say("config.json: " + ("unchanged since first set-up" if same else
                               "WARNING it has changed since first set-up. Restore the file you were issued."))
    else:
        open(stamp, "w").write(sha("config.json"))
        say("config.json: fingerprint saved. Do not edit this file.")

    if os.path.exists("voicelab_starter.ipynb") and not os.path.exists("voicelab.ipynb"):
        os.rename("voicelab_starter.ipynb", "voicelab.ipynb")
        say("Notebook renamed to voicelab.ipynb")

    if have_git():
        if not os.path.isdir(".git"):
            run(["git", "init", "-q"])
            say("git repository created")
        ensure_git_identity()
        gi = ".gitignore"
        if not os.path.exists(gi):
            open(gi, "w").write("cache/\ndata/noisy/\n__pycache__/\n.ipynb_checkpoints/\n.DS_Store\n")
        if run(["git", "rev-parse", "HEAD"]).returncode != 0:
            run(["git", "add", "-A"])
            run(["git", "commit", "-q", "-m", "VoiceLab: first commit"])
            say("first commit made")
    else:
        say("git not found: install git, then rerun setup.")

    try:
        import nltk
        for pkg in ("cmudict", "brown"):
            nltk.download(pkg, quiet=True)
        say("NLTK data (CMUdict, Brown) downloaded")
    except Exception:
        say("NLTK data not downloaded (install nltk first)")

    if a.models:
        say("Downloading pretrained models (a few minutes, needs internet)...")
        from voicelab_utils import _asr
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            _asr("openai/whisper-base")
            _asr(f"openai/whisper-{cfg['decoding']['whisper_size']}")
            import torchaudio
            b = getattr(torchaudio.pipelines, cfg["tts"]["torchaudio_pipeline"])
            b.get_tacotron2()
            b.get_vocoder()
            say("Models downloaded")
        except Exception as e:
            say(f"Model download failed ({e}). Check the internet connection and run again.")
    else:
        say("Pretrained models are downloaded on first use. To fetch them now: python voicelab.py setup --models")

    from voicelab_utils import recording_script
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        recording_script(cfg)
    say("\nYour recording script is saved as recording_script.txt")
    say("Next: record the three blocks (see Part 0 of the brief), save them in data/original/ as")
    say("      words_block, sentences_block and noise_block, then run: python voicelab.py data")
    say("\n" + IGNORABLE)


# ---------------------------------------------------------------- script
def cmd_script(a):
    from voicelab_utils import recording_script
    recording_script(load_cfg())


# ---------------------------------------------------------------- data
def pick_originals():
    files = [f for f in sorted(glob.glob("data/original/*"))
             if os.path.isfile(f) and not os.path.basename(f).startswith(".")]
    roles = {}
    for f in files:
        n = os.path.basename(f).lower()
        for role, keys in (("words", ("word",)), ("sentences", ("sent",)), ("noise", ("noise", "room"))):
            if any(k in n for k in keys):
                roles.setdefault(role, []).append(f)
    if all(len(roles.get(r, [])) == 1 for r in ("words", "sentences", "noise")):
        return {r: roles[r][0] for r in roles}, False
    if len(files) == 3:
        by_len = sorted(files, key=audio_duration, reverse=True)
        return {"words": by_len[0], "sentences": by_len[1], "noise": by_len[2]}, True
    sys.exit("Could not tell which file is which in data/original/. Name them words_block.*, "
             "sentences_block.* and noise_block.* (any audio extension).")


def spoken_digits(text):
    toks = re.findall(r"[a-z]+|\d", text.lower())
    out = []
    for t in toks:
        if t.isdigit():
            out.append(t)
        elif t in NUMWORDS:
            out.append(NUMWORDS[t])
    return "".join(out)


def code_match(heard, code):
    """Number of digits of the code that line up with what was heard (order kept)."""
    import difflib
    m = difflib.SequenceMatcher(None, heard, code, autojunk=False)
    return sum(b.size for b in m.get_matching_blocks())


class Tee:
    """Echo everything printed inside the block and also keep the lines."""
    def __enter__(self):
        self.lines, self._out, self._buf = [], sys.stdout, ""
        outer = self

        class _W:
            def write(self, t):
                outer._out.write(t)
                outer._buf += t
                while "\n" in outer._buf:
                    ln, outer._buf = outer._buf.split("\n", 1)
                    outer.lines.append(ln)
                return len(t)

            def flush(self):
                outer._out.flush()
        sys.stdout = _W()
        return self

    def __exit__(self, *exc):
        sys.stdout = self._out
        if self._buf:
            self.lines.append(self._buf)
        return False


def cmd_data(a):
    from voicelab_utils import (to_16k_mono, split_block, split_noise_block, audit_folder,
                                first_seconds_transcript)
    cfg = load_cfg()
    code = "".join(cfg["code"].split()) if isinstance(cfg["code"], str) else "".join(cfg["code"])
    for d in FOLDERS:
        os.makedirs(d, exist_ok=True)
    problems, notes = [], []
    split_warn = {}

    blocks = {r: f"data/blocks/{r}_block.wav" for r in ("words", "sentences", "noise")}
    if a.skip_existing and all(os.path.exists(b) for b in blocks.values()):
        say("Blocks already converted: skipping conversion.")
    else:
        found, guessed = pick_originals()
        if guessed:
            say("File names do not say which block is which, so I used length: longest = words, "
                "middle = sentences, shortest = noise. Rename them if that is wrong.")
        for role, src in found.items():
            say(f"Converting {os.path.basename(src)} -> {blocks[role]}")
            to_16k_mono(src, blocks[role])

    word_names = [f"{w}_{r}.wav" for w in cfg["vocab"] for r in range(3)]
    sent_names = [f"sent_{i}.wav" for i in range(len(cfg["sentences_spoken"]))]
    done_words = all(os.path.exists(f"data/words/{n}") for n in word_names)
    done_sents = all(os.path.exists(f"data/sentences/{n}") for n in sent_names)
    done_noise = os.path.exists("data/noise/room_noise.wav")
    for label, ok_already, call in (
        ("words", done_words, lambda: split_block(blocks["words"], word_names, "data/words")),
        ("sentences", done_sents, lambda: split_block(blocks["sentences"], sent_names, "data/sentences")),
        ("noise", done_noise, lambda: split_noise_block(blocks["noise"], "data/noise/room_noise.wav")),
    ):
        if a.skip_existing and ok_already:
            say(f"{label}: clips already present, skipping the split.")
            continue
        say(f"\n--- splitting the {label} block")
        try:
            with Tee() as tee:
                call()
            for line in tee.lines:
                if line.strip().startswith("WARNING"):
                    split_warn.setdefault(label, []).append(line.strip())
        except (SystemExit, Exception) as e:
            msg = str(e) if str(e) else type(e).__name__
            problems.append(f"{label} block: {msg}")
            say(f"STOPPED: {msg}")

    say("\n--- audit")
    rows = audit_folder("data")
    flagged = []
    for r in rows:
        why = []
        if r["clipping_pct"] > 0.5:
            why.append("clipping (record a little quieter or further away)")
        if r["snr_db"] < 15 and "noise" not in r["file"]:
            why.append("low SNR (noisy room or too far from the phone)")
        if r["silence_frac"] > 0.85 and "noise" not in r["file"]:
            why.append("mostly silence")
        if r["duration_s"] < 0.2:
            why.append("very short")
        if why:
            flagged.append((r["file"], "; ".join(why)))
    if rows:
        with open("data/audit_report.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    say(f"{len(rows)} files audited (details in data/audit_report.csv).")
    for name, why in flagged:
        say(f"  CHECK {name}: {why}")
    if not flagged:
        say("  no clip was flagged")

    if a.no_whisper:
        say("\nSpoken-code check skipped (--no-whisper).")
    else:
        say("\n--- spoken-code check (Whisper listens to the first 8 s of each block)")
        for role, b in blocks.items():
            if not os.path.exists(b):
                continue
            try:
                text = first_seconds_transcript(b, seconds=8)
                digits = spoken_digits(text)
                hit = code_match(digits, code)
                ok = hit >= len(code) - 2          # a small speech recogniser may mishear a digit or two
                say(f"  {role}: heard '{text.strip()}' -> {hit} of {len(code)} digits match "
                    f"({'code accepted' if ok else 'code NOT recognised'})")
                if not ok:
                    notes.append(f"{role} block: the spoken code was not recognised. Listen to the start of "
                                 f"the recording; if you did not say your code clearly, record the block again.")
            except Exception as e:
                say(f"  {role}: could not run Whisper here ({str(e)[:80]}). Rerun with internet or use --no-whisper.")
                break

    say("")
    for n_ in notes:
        say("CHECK " + n_)
    if problems:
        say("Things to fix:")
        for p in problems:
            say("  - " + p)
        sys.exit(1)
    review = []   # (block, what to listen to)
    for label in ("words", "sentences", "noise"):
        for w_ in split_warn.get(label, []):
            review.append((label, w_))
    for name, why in flagged:
        base = os.path.basename(str(name))
        blk = "sentences" if base.startswith("sent_") else ("noise" if "noise" in str(name) else "words")
        review.append((blk, f"CHECK {name}: {why}"))
    for n_ in notes:
        review.append((n_.split(" block")[0], "CHECK " + n_))
    say("=" * 70)
    say("PLEASE CONFIRM YOUR RECORDINGS")
    if review:
        say("Listen to these before you go on:")
        for blk, txt in review:
            say(f"  [{blk}] {txt}")
    else:
        say("No clip or block was flagged. Still, listen to a few clips to be sure.")
    say("")
    say("  All recordings OK?    -> continue:  python voicelab.py save \"Part 0 data\"")
    say("                                      python voicelab.py precompute")
    say("  Something not OK?     -> record that block again (say every item, pause at least 2 seconds")
    say("                           after each), save it over the file in data/original/ and rerun:")
    say("                           python voicelab.py data")
    say("  A clip that is only a little long, with a breath or extra silence at its edges, is fine.")
    say("=" * 70)


# ---------------------------------------------------------------- precompute
def cmd_precompute(a):
    from voicelab_utils import load_wav, mix_at_snr, wav2vec2_logprobs, whisper_transcribe
    import numpy as np
    from scipy.io import wavfile
    cfg = load_cfg()
    from voicelab_utils import get_device
    say(f"Model steps run on: {get_device()}" + ("  (use --cpu to force the CPU)" if get_device() == "cuda" else ""))
    os.makedirs("cache", exist_ok=True)
    os.makedirs("data/noisy", exist_ok=True)
    tpath = "cache/transcripts.json"
    transcripts = json.load(open(tpath)) if os.path.exists(tpath) else {}
    noise, fs = load_wav("data/noise/room_noise.wav", sr=16000)
    n_sent = len(cfg["sentences_spoken"])
    jobs = []
    for i in range(n_sent):
        jobs.append((f"sent_{i}_clean", f"data/sentences/sent_{i}.wav"))
        x, _ = load_wav(f"data/sentences/sent_{i}.wav", sr=16000)
        for snr in cfg["noise"]["snr_db"]:
            p = f"data/noisy/sent_{i}_snr{snr}.wav"
            y = mix_at_snr(x, noise, snr)
            y = y / max(1.0, float(np.abs(y).max()) / 0.99)
            wavfile.write(p, 16000, (y * 32767).astype(np.int16))
            jobs.append((f"sent_{i}_snr{snr}", p))
    say(f"Mixed noisy versions of your {n_sent} sentences at SNR {cfg['noise']['snr_db']} dB (data/noisy/).")
    tokens_saved = os.path.exists("cache/w2v_tokens.json")
    for name, path in jobs:
        if a.if_missing and os.path.exists(f"cache/w2v_{name}.npy") and name in transcripts:
            continue
        say(f"  {name}: wav2vec2 and Whisper ({cfg['decoding']['whisper_size']})")
        logp, vocab = wav2vec2_logprobs(path)
        np.save(f"cache/w2v_{name}.npy", logp.astype(np.float32))
        if not tokens_saved:
            json.dump(vocab, open("cache/w2v_tokens.json", "w"))
            tokens_saved = True
        transcripts[name] = whisper_transcribe(path, cfg["decoding"]["whisper_size"])
        json.dump(transcripts, open(tpath, "w"), indent=1)
    say("\nSaved in cache/: log posteriors (load_posteriors('sent_0_clean')) and Whisper text "
        "(load_transcripts()). They are the model outputs only; WER, grouping and beam search are yours.")


# ---------------------------------------------------------------- toy
def cmd_toy(a):
    cfg = load_cfg()
    t = cfg["toy"]
    lines = [f"Hand-calculation values for roll {cfg['roll']}  (copy onto your pages; do not retype from memory)", ""]
    lines.append("Part 1 item 1: DFT")
    lines.append("  x[n], n = 0.." + str(len(t["dft_frame"]) - 1) + ": " + ", ".join(str(v) for v in t["dft_frame"]))
    lines += ["", "Part 2 item 1: HMM"]
    for k, v in t["hmm"].items():
        lines.append(f"  {k}: {v}")
    lines += ["  Model: 3 states, left to right, start in state 1. State i stays with probability A_self[i]",
              "  and otherwise moves to state i+1; state 3 stays with probability 1. B_sym1[i] is the",
              "  probability of symbol 1 in state i (symbol 2 has the rest). obs is the observed sequence.",
              "  P(O|model) sums over every final state."]
    lines += ["", "Part 3 item 1: CTC"]
    c = t["ctc"]
    lines.append(f"  labels: {c['labels']}  (posterior columns follow this order)")
    lines.append(f"  target: {c['target']}")
    lines.append("  posteriors, one row per frame (columns follow label order 0, 1, 2, ...):")
    for i, row in enumerate(c["posteriors"]):
        lines.append(f"    t={i + 1}: " + "  ".join(f"{p:.3f}" if isinstance(p, float) else str(p) for p in row))
    lines += ["", "Part 4 item 1: language model"]
    lm = t["lm"]
    for i, s in enumerate(lm["mini_corpus"]):
        lines.append(f"  sentence {i + 1}: {s}")
    lines.append(f"  query bigram: {lm['query_bigram']}")
    lines.append(f"  Kneser-Ney discount: {cfg['language_model']['kn_discount']}")
    lines += ["  Conventions: pad each sentence with <s> and </s>; V = distinct words plus </s>;",
              "  Kneser-Ney continuation counts are over bigram types (<s> counts as a left context)."]
    lines += ["", "Part 5 item 1: WFST"]
    w = t["wfst"]
    for p in w["paths_phones_to_word"]:
        lines.append(f"  phones {p['phones']} -> word {p['word']}, cost {p['cost']}")
    lines.append(f"  semiring: {w['semiring']}")
    lines += ["  Build L as three linear paths from one start state. Every arc is phone:eps at cost 0",
              "  except the last arc of a path, which is phone:word/cost. Count every state (start,",
              "  internal and final)."]
    text = "\n".join(lines)
    open("toy_values.txt", "w").write(text)
    say(text)
    say("\n(also saved as toy_values.txt)")


# ---------------------------------------------------------------- save, log, ai
def cmd_save(a):
    if not have_git() or not os.path.isdir(".git"):
        sys.exit("No git repository here. Run: python voicelab.py setup")
    ensure_git_identity()
    msg = a.message or "work in progress " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    run(["git", "add", "-A"])
    r = run(["git", "commit", "-q", "-m", msg])
    if r.returncode == 0:
        n = run(["git", "rev-list", "--count", "HEAD"]).stdout.strip()
        say(f"Committed: '{msg}'  ({n} commits so far)")
    else:
        say("Nothing new to commit.")


def cmd_log(a):
    if not os.path.exists("design_log.md"):
        sys.exit("design_log.md not found here.")
    lines = open("design_log.md").read().split("\n")
    pat = re.compile(r"^\s*-\s*Part\s*%d\s*:(.*)$" % a.part)
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if m:
            old = m.group(1).strip()
            lines[i] = f"- Part {a.part}: " + (old + "; " if old else "") + a.text.strip()
            open("design_log.md", "w").write("\n".join(lines))
            say(lines[i])
            return
    lines.append(f"- Part {a.part}: {a.text.strip()}")
    open("design_log.md", "w").write("\n".join(lines))
    say(lines[-1])


def cmd_ai(a):
    qs = [("Tools used (name and version)", "Tools used (name and version):"),
          ("Parts where you used them (for example debugging Part 2)",
           "Parts where I used them (for example debugging Part 2, explaining CTC in Part 3):"),
          ("What you wrote yourself", "What I wrote myself:"),
          ("What you verified yourself, and how", "What I verified myself, and how:")]
    say("AI use statement. If you used no AI tools, write: none. Answer each line in a sentence or two.")
    out = ["# AI use statement", ""]
    for prompt, label in qs:
        ans = input(f"{prompt}: ").strip()
        out.append(f"- {label} {ans}")
    open("ai_statement.md", "w").write("\n".join(out) + "\n")
    say("ai_statement.md written. Edit it by hand any time.")


# ---------------------------------------------------------------- handwork
def cmd_handwork(a):
    from PIL import Image, ImageOps
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic")
    files = sorted(f for f in glob.glob(os.path.join(a.folder, "*")) if f.lower().endswith(exts))
    if not files:
        sys.exit(f"No images in {a.folder}/. Put one photo or scan per page there, named so that they sort in "
                 "part order, for example 1_dft.jpg, 2_hmm.jpg, 3_ctc.jpg, 4_lm.jpg, 5_wfst.jpg.")
    say(f"{len(files)} pages, in this order:")
    for i, f in enumerate(files, 1):
        say(f"  page {i}: {os.path.basename(f)}")
    if len(files) < 5:
        say(f"NOTE: five hand-written items are expected (DFT, HMM, CTC, language model, WFST); you have {len(files)}.")
    imgs = []
    for f in files:
        try:
            im = ImageOps.exif_transpose(Image.open(f)).convert("L" if a.gray else "RGB")
        except Exception as e:
            sys.exit(f"Could not open {f}: {e} (for iPhone HEIC photos, export as JPEG first)")
        imgs.append(im)
    for size, q in ((2000, 80), (1800, 70), (1600, 60), (1400, 50), (1200, 45)):
        pages = []
        for im in imgs:
            im2 = im.copy()
            im2.thumbnail((size, size))
            pages.append(im2.convert("RGB"))
        pages[0].save("handwork.pdf", save_all=True, append_images=pages[1:], quality=q)
        mb = os.path.getsize("handwork.pdf") / 1e6
        if mb < 9.5:
            break
    say(f"\nhandwork.pdf written: {len(pages)} pages, {mb:.1f} MB.")
    say("Open it and check that every page is upright, readable, and shows your roll number, "
        "part, item and your toy values.")
    if mb >= 9.5:
        say("Still above 9.5 MB: retake the photos at a lower resolution or use --gray.")


# ---------------------------------------------------------------- report
REPORT_TEMPLATE = """# VoiceLab report: {{roll}}

*At most 4 pages. Delete the italic hints when you write. Add figures with
`![caption](figures/name.png)`; save them from the notebook with `savefig('name')`.*

## 1. Configuration

{{config_table}}

## 2. Results

{{results_table}}

*Add the key plots and tables here: spectrogram (Part 1), HMM log-likelihood and confusion matrix
(Part 2), perplexities (Part 4), WER for both recognisers and both beam widths (Part 6), WER against
SNR (Part 6), Griffin-Lim convergence and STOI (Part 7). One sentence under each figure.*

## 3. Findings

*Answer only what your own data can answer, with numbers:*

- Which minimal pair failed most often, and what does that say about your voice?
- Which SNR hurt more, and which recogniser broke first?
- Where did the language model help or hurt you?

## 4. Notes on process

*Two or three sentences: what went wrong first, what you changed, anything you would do differently.*
"""


def cmd_report(a):
    cfg = load_cfg()
    if a.init and os.path.exists("report.md"):
        sys.exit("report.md already exists; delete it first if you really want a fresh template.")
    if not os.path.exists("report.md"):
        open("report.md", "w").write(REPORT_TEMPLATE)
        say("report.md created. Write your findings in it, then run: python voicelab.py report")
        return
    text = open("report.md").read()
    fe = cfg["front_end"]
    cfg_tab = ["| Item | Your value |", "|---|---|",
               f"| Roll | {cfg['roll']} |",
               f"| Vocabulary | {', '.join(cfg['vocab'])} |",
               f"| Front end | {fe['frame_ms']} ms frames, {fe['hop_ms']} ms hop, {fe['window']}, "
               f"{fe['n_mels']} mel, {fe['n_mfcc']} MFCC, pre-emphasis {fe['preemph']}, lifter {fe['lifter']}, "
               f"delta N {fe['delta_N']} |",
               f"| HMM | {cfg['hmm']['states']} states, held-out repetition {cfg['hmm']['heldout_rep']} |",
               f"| Kneser-Ney discount | {cfg['language_model']['kn_discount']} |",
               f"| SNR levels (dB) | {cfg['noise']['snr_db']} |",
               f"| Beam widths | {cfg['decoding']['beam_widths']} |",
               f"| Whisper size | {cfg['decoding']['whisper_size']} |",
               f"| Seed | {cfg['speaker_seed']} |"]
    res = json.load(open("results.json")) if os.path.exists("results.json") else {}
    rtab = ["| Check value | Saved value |", "|---|---|"]
    for k, v in res.items():
        rtab.append(f"| {k} | {v:.6g} |" if isinstance(v, (int, float)) else f"| {k} | {v} |")
    text = (text.replace("{{roll}}", str(cfg["roll"])).replace("{{config_table}}", "\n".join(cfg_tab))
            .replace("{{results_table}}", "\n".join(rtab) if len(rtab) > 2 else "*(no check values saved yet)*"))
    import markdown
    body = markdown.markdown(text, extensions=["tables"])
    css = ("@page{size:A4;margin:1.6cm}body{font-family:Georgia,serif;font-size:10pt;line-height:1.3;color:#000}"
           "h1{font-size:15pt}h2{font-size:12pt;margin:10pt 0 3pt}table{border-collapse:collapse;font-size:9pt}"
           "td,th{border:1px solid #000;padding:2px 5px}img{max-width:100%;max-height:7cm}"
           "em{color:#333}p,li{margin:3pt 0}")
    html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
    open("report.html", "w").write(html)
    made = False
    try:
        from weasyprint import HTML
        HTML(string=html, base_url=os.getcwd()).write_pdf("report.pdf")
        made = True
    except Exception:
        pass
    if not made:  # a browser that is already on most machines: Edge or Chrome, headless
        cands = [shutil.which(n) for n in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser")]
        for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
            if base:
                cands += [os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                          os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")]
        for exe in [c for c in cands if c and os.path.exists(c)]:
            if os.path.exists("report.pdf"):
                os.remove("report.pdf")
            url = "file:///" + os.path.abspath("report.html").replace("\\", "/").lstrip("/")
            run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                 "--print-to-pdf=" + os.path.abspath("report.pdf"), url])
            if os.path.exists("report.pdf") and os.path.getsize("report.pdf") > 0:
                made = True
                break
    if not made and shutil.which("pandoc"):
        made = run(["pandoc", "report.md", "-o", "report.pdf"]).returncode == 0
    if not made:
        say("Could not make a PDF automatically. report.html was written: open it in a browser, press Ctrl+P and "
            "save as PDF named report.pdf in this folder. (Optional: pip install weasyprint, which needs extra "
            "libraries on Windows.)")
        return
    from progress import pdf_pages
    n = pdf_pages("report.pdf")
    say(f"report.pdf written ({n} pages)." if n else "report.pdf written.")
    if n and n > 4:
        say("WARNING: more than 4 pages. Shorten the text or shrink the figures.")


# ---------------------------------------------------------------- package
def cmd_package(a):
    from progress import pdf_pages, finite, KEYS
    cfg = load_cfg()
    roll = str(cfg["roll"])
    fails, warns = [], []
    for f in REQUIRED:
        if not os.path.exists(f):
            fails.append(f"missing file: {f}")
    for f in ("voicelab_utils.py", "wfst_starter.py"):
        if not os.path.exists(f):
            fails.append(f"missing file: {f} (the notebook imports it)")
    if not os.path.isdir("data") or not glob.glob("data/**/*.wav", recursive=True):
        fails.append("data/ has no recordings")
    if not os.path.isdir(".git"):
        fails.append("no git repository (.git folder)")
    stamp = ".config.sha256"
    if os.path.exists(stamp) and open(stamp).read().strip() != sha("config.json"):
        fails.append("config.json has been changed. Restore the file you were issued.")
    if os.path.exists("results.json"):
        res = json.load(open("results.json"))
        lacking = [k for ks in KEYS.values() for k in ks if not finite(res.get(k))]
        if lacking:
            warns.append("results.json has no value for: " + ", ".join(lacking))
    if os.path.exists("report.pdf"):
        n = pdf_pages("report.pdf")
        if n and n > 4:
            fails.append(f"report.pdf has {n} pages (limit 4)")
    if os.path.exists("handwork.pdf"):
        if os.path.getsize("handwork.pdf") > 10 * 1024 * 1024:
            fails.append("handwork.pdf is over 10 MB (run: python voicelab.py handwork --gray)")
        n = pdf_pages("handwork.pdf")
        if n is not None and n < 5:
            warns.append(f"handwork.pdf has {n} pages; five hand-written items are expected")
    if os.path.exists("design_log.md"):
        empty = [m.group(1) for m in re.finditer(r"^\s*-\s*Part\s*(\d)\s*:\s*$", open("design_log.md").read(), re.M)]
        if empty:
            warns.append("design_log.md has no line yet for Part " + ", ".join(empty))
    if os.path.exists("ai_statement.md"):
        body = [ln for ln in open("ai_statement.md").read().splitlines() if ln.strip().startswith("-")]
        if any(len(re.sub(r"^[^:]*:", "", ln).strip()) < 3 for ln in body):
            warns.append("ai_statement.md still has an empty line (write 'none' if that is the truth)")
    if os.path.exists("wfst_starter.py"):
        body = open("wfst_starter.py").read().split("def compose", 1)[-1].split("\ndef ", 1)[0]
        if "raise NotImplementedError" in body:
            warns.append("compose() in wfst_starter.py still raises NotImplementedError (Part 5 item 2)")
    if os.path.exists("voicelab.ipynb"):
        nb = json.load(open("voicelab.ipynb"))
        if not any(c.get("outputs") for c in nb.get("cells", []) if c["cell_type"] == "code"):
            warns.append("the notebook has no saved outputs; run it and keep the outputs")
    if have_git() and os.path.isdir(".git"):
        if run(["git", "status", "--porcelain"]).stdout.strip():
            if a.commit:
                ensure_git_identity()
                run(["git", "add", "-A"])
                run(["git", "commit", "-q", "-m", "VoiceLab: final state before packaging"])
                say("Uncommitted changes committed.")
            else:
                warns.append("uncommitted changes (run: python voicelab.py save \"final\", or package --commit)")

    for f in fails:
        say("FIX   " + f)
    for w in warns:
        say("CHECK " + w)
    if fails:
        sys.exit("\nNot packaged. Fix the lines marked FIX and run again.")
    if warns and not a.force:
        sys.exit("\nNot packaged yet. Look at the lines marked CHECK; if they are fine, run again with --force.")

    zname = f"{roll}_voicelab.zip"
    count = 0
    with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk("."):
            rel_root = os.path.relpath(root, ".")
            dirs[:] = [d for d in dirs if os.path.normpath(os.path.join(rel_root, d)).replace("\\", "/") not in SKIP_DIRS
                       and d not in SKIP_DIRS]
            for fn in files:
                p = os.path.normpath(os.path.join(rel_root, fn))
                top = p.split(os.sep)[0]
                keep = (top in ("data", ".git", "figures", "handwork") or p in REQUIRED or p in ALSO_PACK
                        or p in (".gitignore", ".config.sha256", "toy_values.txt", "report.html"))
                if keep and fn not in SKIP_FILES and not p.endswith(".zip"):
                    z.write(p, os.path.join(f"{roll}_voicelab", p))
                    count += 1
    say(f"\nWrote {zname}: {count} files, {os.path.getsize(zname) / 1e6:.1f} MB.")
    say("Before you submit: test run_all.sh in a fresh Colab session. Submission date: 28 November.")


# ---------------------------------------------------------------- progress
def cmd_progress(a):
    import progress
    sys.argv = ["progress.py"] + (["--summary"] if a.summary else [])
    progress.main()


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(prog="voicelab.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--no-install", action="store_true", help="do not pip install missing libraries")
    s.add_argument("--models", action="store_true", help="also download the pretrained models now")
    s.set_defaults(fn=cmd_setup)
    sub.add_parser("script").set_defaults(fn=cmd_script)
    cpu = argparse.ArgumentParser(add_help=False)
    cpu.add_argument("--cpu", action="store_true", help="force the CPU even if a GPU is available")
    s = sub.add_parser("data", parents=[cpu])
    s.add_argument("--no-whisper", action="store_true", help="skip the spoken-code check")
    s.add_argument("--skip-existing", action="store_true", help="reuse blocks and clips already made")
    s.set_defaults(fn=cmd_data)
    s = sub.add_parser("precompute", parents=[cpu])
    s.add_argument("--if-missing", action="store_true", help="only do what is not in cache/ yet")
    s.set_defaults(fn=cmd_precompute)
    sub.add_parser("toy").set_defaults(fn=cmd_toy)
    s = sub.add_parser("save")
    s.add_argument("message", nargs="?", default="")
    s.set_defaults(fn=cmd_save)
    s = sub.add_parser("log")
    s.add_argument("part", type=int)
    s.add_argument("text")
    s.set_defaults(fn=cmd_log)
    sub.add_parser("ai").set_defaults(fn=cmd_ai)
    s = sub.add_parser("handwork")
    s.add_argument("folder", nargs="?", default="handwork")
    s.add_argument("--gray", action="store_true", help="grey-scale pages (smaller file)")
    s.set_defaults(fn=cmd_handwork)
    s = sub.add_parser("report")
    s.add_argument("--init", action="store_true", help="create report.md from the template")
    s.set_defaults(fn=cmd_report)
    s = sub.add_parser("package")
    s.add_argument("--commit", action="store_true", help="commit pending changes first")
    s.add_argument("--force", action="store_true", help="package even if there are CHECK lines")
    s.set_defaults(fn=cmd_package)
    s = sub.add_parser("progress")
    s.add_argument("--summary", action="store_true", help="one line per part only")
    s.add_argument("--detail", action="store_true", help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_progress)
    a = p.parse_args()
    if getattr(a, "cpu", False):
        os.environ["VOICELAB_DEVICE"] = "cpu"
    a.fn(a)


if __name__ == "__main__":
    main()
