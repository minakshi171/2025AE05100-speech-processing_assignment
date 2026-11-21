# VoiceLab starter kit

Download your own <ROLLNO>_config.json (use only the file with your roll number) and put it next to these files. `setup` copies it to config.json, the name the kit uses. Then:

    python voicelab.py setup        first-time set-up (libraries, folders, git, notebook name, corpora)
    python voicelab.py script       your read-aloud recording script
    python voicelab.py data         recordings -> 16 kHz blocks -> clips -> audit -> spoken-code check
    python voicelab.py precompute   noisy sentences, wav2vec2 posteriors, Whisper text (cache/)
    python voicelab.py toy          your hand-calculation numbers, readable
    python voicelab.py save "msg"   git commit
    python voicelab.py log 2 "text" add to the Part 2 line of design_log.md
    python voicelab.py ai           fill in ai_statement.md
    python voicelab.py handwork     photos in handwork/ -> handwork.pdf
    python voicelab.py report       report.md -> report.pdf
    python voicelab.py package      final checks, then <ROLLNO>_voicelab.zip
    python voicelab.py progress     every item ticked or not, and the next step (--summary for one line per part)

In Colab put ! in front of each command.

The commands are written in full (`python voicelab.py data`). If you want to type just `vl data`, define it once per terminal window. Windows PowerShell: `function vl { python voicelab.py @args }`. Mac/Linux: `alias vl='python voicelab.py'`.

Files
- voicelab.py: the commands above
- progress.py: the progress checker (also run by voicelab.py progress). It checks that evidence exists, never whether your answers are right
- voicelab_utils.py: helpers for the notebook (config, audio, audit, splitting, noise mixing, results.json, model outputs, corpora loaders, savefig, compare)
- wfst_starter.py: FST container and shortest path for Part 5. You write compose().
- voicelab_starter.ipynb: one section per part (setup renames it voicelab.ipynb)
- run_all.sh: the one-command run
- design_log.md, ai_statement.md: templates

The speech processing code in the parts marked [Critical] and every hand calculation are yours to do.

Notes
- The spoken-code check in `data` is advisory. It listens to the first 8 seconds of each block and prints a CHECK note if the digits you said do not match your code (up to two wrong digits are accepted). It never stops the run. If it keeps complaining, re-record that block, or carry on and mention it in your design log.
- `results.json` values for your own recordings (for example `mfcc_sum_first_clip`) must come from your own code. The evaluator reruns `run_all.sh` and compares with your report, so make sure the one-command run reproduces them.

Messages you can ignore: "Failed to find CUDA" (a GPU is not required; the CPU is enough), pip's "not on PATH" and "new release of pip" notices, and library UserWarning or FutureWarning lines. Act only when a command stops with an error.

GPU: if PyTorch can use a GPU, the model steps use it automatically. Add --cpu to `data` or `precompute` to force the CPU (or set the environment variable VOICELAB_DEVICE=cpu for the notebook helpers). Results agree up to rounding.
