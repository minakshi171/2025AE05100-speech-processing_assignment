"""
progress.py  --  your VoiceLab progress checker.

Run it from the folder that holds config.json, data/ and voicelab.ipynb:

    python progress.py            # every part with its items ticked or not, and your next step
    python progress.py --summary  # one line per part only

What it does: it looks for the files, recordings, saved check values and notes that the checklist
(Appendix E of the brief) asks for, and tells you which step to do next.

What it does NOT do: it never judges whether your answers are correct, never shows or hints at an
expected value, and is not part of the marking. A tick here only means the evidence is present,
so verify your own work as the brief describes. It uses only the Python standard library.
"""
import csv
import datetime
import glob
import json
import math
import os
import re
import subprocess
import sys
import wave

DEADLINE = datetime.date(2026, 11, 28)
NOTEBOOKS = ["voicelab.ipynb", "voicelab_starter.ipynb"]
KEYS = {
    1: ["mfcc_sum_first_clip"],
    2: ["toy_hmm_forward", "toy_hmm_viterbi"],
    3: ["toy_ctc_prob"],
    4: ["toy_add1_bigram", "toy_kn_bigram"],
    5: ["wfst_det_states"],
    6: ["wer_sentence1_wav2vec2", "wer_widest_beam"],
    7: ["griffinlim_convergence_30"],
}
TITLES = {
    1: "Part 1. Signal analysis and features",
    2: "Part 2. HMM acoustic model",
    3: "Part 3. CTC",
    4: "Part 4. Language model and lexicon",
    5: "Part 5. WFSTs",
    6: "Part 6. Recognisers and decoding",
    7: "Part 7. Speech synthesis",
}
HAND = {1: "DFT", 2: "HMM forward and Viterbi", 3: "CTC probability",
        4: "add-one and Kneser-Ney", 5: "determinise and minimise"}


# ---------------------------------------------------------------- helpers
def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def wav_info(path):
    try:
        with wave.open(path) as w:
            return w.getframerate(), w.getnchannels(), w.getsampwidth() * 8, w.getnframes() / w.getframerate()
    except Exception:
        return None


def git_commits():
    try:
        out = subprocess.run(["git", "log", "--format=%ct"], capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            return None
        return [int(x) for x in out.stdout.split()]
    except Exception:
        return None


def pdf_pages(path):
    """Page count of a PDF (uses pypdf if installed, else reads the file)."""
    try:
        from pypdf import PdfReader
        return len(PdfReader(path).pages)
    except Exception:
        pass
    try:
        data = open(path, "rb").read()
        n = len(re.findall(rb"/Type\s*/Page[^s]", data))
        if n:
            return n
        counts = [int(c) for c in re.findall(rb"/Count\s+(\d+)", data)]
        return max(counts) if counts else None
    except Exception:
        return None


def notebook_sections():
    """Return {part: (code_lines, cells_with_output)} counting only your own lines of code."""
    nbpath = next((p for p in NOTEBOOKS if os.path.exists(p)), None)
    nb = read_json(nbpath) if nbpath else None
    if not nb:
        return nbpath, {}
    part, out = None, {}
    for c in nb.get("cells", []):
        src = "".join(c.get("source", []))
        if c["cell_type"] == "markdown":
            m = re.search(r"^#+\s*Part\s*(\d)", src, re.M)
            if m:
                part = int(m.group(1))
                out.setdefault(part, [0, 0])
            continue
        if part is None or src.lstrip().startswith("# [starter]"):
            continue
        starter = ("brown_sentences(", "cmudict_dict(", "load_transcripts(", "load_posteriors(")
        lines = [ln for ln in src.splitlines() if ln.strip() and not ln.strip().startswith(("#", "!"))
                 and not any(k in ln for k in starter)]
        out[part][0] += len(lines)
        if c.get("outputs"):
            out[part][1] += 1
    return nbpath, out


def log_line_filled(part):
    try:
        for ln in open("design_log.md"):
            m = re.match(r"\s*-\s*Part\s*%d\s*:\s*(.*)" % part, ln)
            if m:
                return len(m.group(1).strip()) >= 15
    except Exception:
        pass
    return False


def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


# ---------------------------------------------------------------- checks
# Each check returns (status, items, next_hint); items are (ok, text) lines.
def check_setup(cfg):
    items = []
    items.append((cfg is not None and "roll" in cfg and "vocab" in cfg,
                  "your configuration file is present and readable (do not edit it)"))
    nbpath = next((p for p in NOTEBOOKS if os.path.exists(p)), None)
    items.append((nbpath == "voicelab.ipynb", "notebook is renamed to voicelab.ipynb"))
    commits = git_commits()
    items.append((bool(commits), "git repository exists with at least one commit"))
    hint = None
    if not items[0][0]:
        hint = "Put the <ROLLNO>_config.json you were issued next to this script, then run: python voicelab.py setup"
    elif not items[1][0]:
        hint = "Run: python voicelab.py setup (it renames the notebook and prepares the folders)."
    elif not items[2][0]:
        hint = "Run: python voicelab.py setup (starts git), then python voicelab.py save \"message\" at the end of each work session."
    return items, hint


def check_part0(cfg):
    items = []
    orig = [f for f in glob.glob("data/original/*") if os.path.isfile(f)]
    items.append((len(orig) >= 3, "originals kept in data/original/ (3 files: words, sentences, room noise)"))
    blocks = glob.glob("data/blocks/*.wav")
    bad = [b for b in blocks if (wav_info(b) or (0, 0, 0, 0))[:3] != (16000, 1, 16)]
    items.append((len(blocks) >= 3 and not bad,
                  "3 converted block files in data/blocks/ (16 kHz, mono, 16-bit)"
                  + (f"  [{len(bad)} not in that format]" if bad else "")))
    words_ok = sent_ok = False
    if cfg:
        want = [f"{w}_{r}.wav" for w in cfg.get("vocab", []) for r in range(3)]
        have = set(os.listdir("data/words")) if os.path.isdir("data/words") else set()
        missing = [w for w in want if w not in have]
        words_ok = bool(want) and not missing
        items.append((words_ok, f"30 word clips in data/words/ named word_rep.wav"
                      + (f"  [{len(missing)} missing]" if missing else "")))
        have = set(os.listdir("data/sentences")) if os.path.isdir("data/sentences") else set()
        missing = [f"sent_{i}.wav" for i in range(4) if f"sent_{i}.wav" not in have]
        sent_ok = not missing
        items.append((sent_ok, "4 sentence clips sent_0.wav to sent_3.wav in data/sentences/"
                      + (f"  [{len(missing)} missing]" if missing else "")))
    noise = wav_info("data/noise/room_noise.wav")
    items.append((bool(noise) and noise[3] >= 8, "room noise clip data/noise/room_noise.wav (at least 8 s)"))
    clips = glob.glob("data/words/*.wav") + glob.glob("data/sentences/*.wav")
    odd = [c for c in clips if (wav_info(c) or (0, 0, 0, 0))[:3] != (16000, 1, 16)]
    items.append((bool(clips) and not odd, "every clip is 16 kHz, mono, 16-bit"
                  + (f"  [{len(odd)} clips differ]" if odd else "")))
    items.append((os.path.exists("data/words/split_report.csv") and os.path.exists("data/sentences/split_report.csv"),
                  "split reports written (read them for WARNING lines and listen to any clip named)"))
    hint = None
    for ok, text in items:
        if not ok:
            if "originals" in text:
                hint = ("Run python voicelab.py script, record the three blocks (each starting with your spoken code), "
                        "save them in data/original/ as words_block, sentences_block, noise_block, then run: python voicelab.py data")
            elif "converted" in text:
                hint = "Run: python voicelab.py data (converts the blocks and splits the clips)."
            elif "split" in text or "clips" in text or "word clips" in text or "sentence" in text:
                hint = ("Run: python voicelab.py data. If it stops with an error, re-record that block with longer pauses.")
            else:
                hint = text.split(" (")[0].capitalize() + ": see Part 0 in the brief."
            break
    return items, hint


def check_part(n, results, sections):
    items = []
    code_lines, outputs = sections.get(n, (0, 0))
    items.append((n not in HAND or None, None))  # placeholder removed below
    items = []
    if n in HAND:
        items.append((None, f"hand-written page for {HAND[n]} (checked by you and in handwork.pdf, see Finish)"))
    items.append((code_lines >= 3, "your code is in the notebook cells for this part"))
    items.append((outputs >= 1, "the notebook cells have been run and the outputs are kept"))
    if n == 5:
        try:
            src = open("wfst_starter.py").read()
            done = "raise NotImplementedError" not in src.split("def compose", 1)[1].split("\ndef ", 1)[0]
        except Exception:
            done = False
        items.append((done, "compose() in wfst_starter.py is written (it no longer raises NotImplementedError)"))
    for k in KEYS[n]:
        v = (results or {}).get(k)
        items.append((finite(v), f"check value '{k}' saved with res.set(...) in results.json"))
    items.append((log_line_filled(n), f"design_log.md has a line for Part {n}"))
    hint = None
    for ok, text in items:
        if ok is False:
            if "your code" in text:
                hint = f"Start the {TITLES[n]} items in the notebook, in the order listed in the brief (Section 3)."
            elif "outputs" in text:
                hint = "Run the notebook cells for this part so the plots and printed results are saved."
            elif "check value" in text:
                hint = "After your own verification, record it: " + text.split(" saved")[0].replace("check value ", "res.set(") + ", value)."
            elif "compose" in text:
                hint = "Write compose() in wfst_starter.py and test it on small examples."
            else:
                hint = f'Add a line: python voicelab.py log {n} "what you tried, what failed, what you changed"'
            break
    return items, hint


def check_finish(results):
    items = []
    keys = sum(KEYS.values(), [])
    got = [k for k in keys if finite((results or {}).get(k))]
    items.append((len(got) == len(keys), f"check values saved in results.json ({len(got)} of {len(keys)})"))
    hp = pdf_pages("handwork.pdf")
    big = os.path.exists("handwork.pdf") and os.path.getsize("handwork.pdf") > 10 * 1024 * 1024
    items.append((bool(hp) and hp >= 5 and not big,
                  "handwork.pdf: one scanned PDF, at least 5 pages (one per hand item), under 10 MB"
                  + ("  [file is over 10 MB]" if big else "")))
    rp = pdf_pages("report.pdf")
    items.append((bool(rp) and rp <= 4, "report.pdf exists and is at most 4 pages"
                  + (f"  [now {rp} pages]" if rp and rp > 4 else "")))
    try:
        ai = open("ai_statement.md").read()
        filled = all(len(re.sub(r"^\W*[^:]*:", "", ln).strip()) >= 3
                     for ln in ai.splitlines() if ln.strip().startswith("-"))
    except Exception:
        filled = False
    items.append((filled, "ai_statement.md is complete (write 'none' if you used no AI tools, with a reason)"))
    items.append((os.path.exists("run_all.sh"), "run_all.sh is present (test it in a fresh Colab session: bash run_all.sh)"))
    hint = None
    for ok, text in items:
        if not ok:
            if text.startswith("check"):
                hint = "Save the remaining check values from the notebook with res.set(...)."
            elif text.startswith("handwork"):
                hint = "Put one photo per hand-written page in handwork/ (named 1_..., 2_..., in part order), then run: python voicelab.py handwork"
            elif text.startswith("report"):
                hint = "Run: python voicelab.py report (creates report.md the first time). Write your findings in it, run it again for report.pdf."
            elif text.startswith("ai_statement"):
                hint = "Run: python voicelab.py ai (or edit ai_statement.md by hand)."
            else:
                hint = "Restore run_all.sh from the starter kit."
            break
    return items, hint


# ---------------------------------------------------------------- main
def main():
    detail = "--summary" not in sys.argv
    cfg = read_json("config.json")
    if cfg is None:
        import glob
        found = glob.glob("*_config.json")
        if len(found) == 1:
            cfg = read_json(found[0])
    results = read_json("results.json") or {}
    nbpath, sections = notebook_sections()

    groups = [("Set up", *check_setup(cfg)), ("Part 0. Dataset", *check_part0(cfg))]
    for n in range(1, 8):
        groups.append((TITLES[n], *check_part(n, results, sections)))
    groups.append(("Finish", *check_finish(results)))

    print("VoiceLab progress" + (f"  (roll {cfg['roll']})" if cfg and "roll" in cfg else ""))
    days = (DEADLINE - datetime.date.today()).days
    print(f"Submission: 28 November ({days} days left)" if days >= 0 else "Submission date has passed.")
    print("This shows whether the evidence is present, not whether your work is correct.")
    if detail:
        print("[x] evidence found   [ ] not yet   [-] you confirm this yourself (not counted)")
    print()

    next_step = None
    total = done_n = 0
    for name, items, hint in groups:
        checks = [ok for ok, _ in items if ok is not None]
        d = sum(1 for ok in checks if ok)
        total += len(checks)
        done_n += d
        status = "done" if d == len(checks) else ("not started" if d == 0 else "in progress")
        print(f"[{'x' if status == 'done' else ' '}] {name:<42} {d}/{len(checks)}  {status}")
        if detail:
            for ok, text in items:
                print(f"      [{'x' if ok else (' ' if ok is not None else '-')}] {text}")
            print()
        if next_step is None and status != "done":
            next_step = (name, hint)

    print(f"\nOverall: {done_n} of {total} checks have evidence.")
    if next_step:
        print(f"\nNext step: {next_step[0]}\n  {next_step[1]}")
    else:
        print("\nEverything has evidence. Test run_all.sh in a fresh Colab session, then zip as "
              "<ROLLNO>_voicelab.zip and submit.")

    commits = git_commits()
    if commits:
        age = (datetime.datetime.now().timestamp() - max(commits)) / 86400
        if age > 5:
            print(f"\nReminder: your last git commit was {int(age)} days ago. Commit your current work.")
    if not detail:
        print("\nRun 'python progress.py' without --summary to see every item.")


if __name__ == "__main__":
    main()
