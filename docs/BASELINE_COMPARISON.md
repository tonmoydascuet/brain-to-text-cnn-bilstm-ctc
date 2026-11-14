# Baseline comparison: what changed, what was retained

This document separates **implemented differences** from **unverified comparative claims**, based on the uploaded project notebook.

## What remains baseline-informed

- **Input:** 512 pre-extracted neural features, with 20-ms time bins (not raw neural electrode voltages).
- **Gaussian smoothing:** σ = 2 bins and a 100-element depthwise temporal kernel.
- **Training augmentation:** white noise with standard deviation 1.0, trial-constant feature offset with standard deviation 0.2, and randomized removal of 0–2 initial bins.
- **Evaluation:** training-time noise, offset, and time cutting are disabled on validation/test features; smoothing is retained.

The preprocessing choices are **adapted** from the original open-source configuration; implementation details such as boundary handling, application order, and batching are not asserted to be exact copies.

## What is different

1. **Architecture:** two stride-2 1-D convolutional layers precede a three-layer bidirectional LSTM (512 hidden units *per direction*). This is not the authors' phoneme-oriented GRU network.
2. **Labels:** the model uses **characters** extracted from training transcripts, with index 0 as a CTC blank; it does not use a phoneme lexicon.
3. **Loss/decoding:** character-level CTC training with greedy collapse (remove repeated consecutive labels and blanks), without an external language model.
4. **Training setup:** in this notebook, 100 epochs, batch size 64, AdamW, OneCycleLR, dropout 0.5 and gradient clipping at 5.0.
5. **Monitoring:** validation WER/CER, BLEU, ROUGE and exact-match accuracy; best checkpoint selected by validation WER.

## Claims not established

- No controlled result establishes that CNN–BiLSTM beats the paper's original baseline.
- No ablation identifies which component causes lower WER.
- The validation WER is not a directly comparable competition leaderboard or blinded test WER.
- This code is not a direct EEG-to-text model; the competition provides **intracortical neural features**.

## Suggested project contribution statement

> We implement a CNN–BiLSTM–CTC character decoder for pre-extracted intracortical neural features, with augmentation and smoothing choices adapted from a public brain-to-text baseline. Unlike the baseline's phoneme-based GRU and language-model pipeline, the implemented model produces transcripts through direct greedy character decoding. The trained model achieves 26.26% WER and 13.48% CER on the provided validation split, without asserting superiority over the published baseline.
