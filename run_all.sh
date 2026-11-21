#!/usr/bin/env bash
# One command that regenerates every figure and results.json from data/ and config.json.
# Test it in a fresh Colab session before you submit:  bash run_all.sh
set -e
pip install -q numpy scipy matplotlib nltk jiwer librosa transformers torch torchaudio soundfile pystoi pillow markdown pypdf nbconvert ipykernel
# g2p-en sometimes fails to build one of its dependencies; the second line installs it without that dependency
pip install -q g2p-en || pip install -q --no-deps g2p-en inflect
python voicelab.py setup --no-install        # folders, corpora, notebook name (safe to repeat)
python voicelab.py precompute --if-missing   # noisy sentences, wav2vec2 posteriors, Whisper text
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3000 voicelab.ipynb
python voicelab.py progress
echo "Done. See results.json and the figures folder."
