# Design log

One or two lines per part: what you tried, what failed, what you changed. Include at least one failure.

- Part 0:
- Part 1: Implemented MFCC extraction using pre-emphasis, framing, Hann window, power spectrum, manually constructed Mel filter bank, log Mel energies and DCT. Tested on bad_0.wav and obtained 56 frames with 12 coefficients. MFCC sum: -2407.322836948214.; Implemented F0 estimation using autocorrelation with 40 ms frames, 10 ms hop, Hann window and a 70-400 Hz pitch search range. For sentence 0, detected 546 voiced frames out of 1030, with average estimated F0 of 267.11 Hz. Saved the F0 contour figure.
- Part 2: Implemented diagonal Gaussian HMM emissions in the log domain, Forward and Viterbi algorithms, and segmental k-means training. Trained 10 word models with 4 states and 5 iterations each. Evaluated on held-out repetition 1 and achieved 20% accuracy (2/10). Saved training likelihood and confusion matrix figures. Limited training data may contribute to word confusions.
- Part 3: Implemented CTC forward algorithm in log space using blank-expanded target labels. Calculated toy CTC probability as 0.1752 and log probability as -1.7418271. Verified against PyTorch CTCLoss with absolute difference 4.05e-10.
- Part 4: Built a 39-word pronunciation lexicon using CMUdict and G2P for missing words. Added three weighted pronunciation alternatives for Anirudh, Kochi and Lucknow.; Built a 39-word pronunciation lexicon using CMUdict and G2P for missing words. Added three weighted pronunciation alternatives for Anirudh, Kochi and Lucknow.
- Part 5:
- Part 6:
- Part 7:
