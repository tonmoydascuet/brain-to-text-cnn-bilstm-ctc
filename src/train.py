#!/usr/bin/env python3
"""Kaggle-oriented training entry point extracted from the accompanying notebook.

The model, preprocessing, training loop, metrics, and test decoding match
notebooks/brain_to_text_cnn_bilstm_ctc.ipynb. This script is not a reimplementation
of the phoneme-GRU baseline. See README.md and docs/REPRODUCIBILITY.md.
"""

"""
Brain-to-Text '25: CNN-BiLSTM character CTC (no oscillator).
Paper-aligned neural preprocessing follows Card et al. public PyTorch baseline:
pre-extracted 20 ms 512-feature inputs, Gaussian smoothing; training-only
white noise, constant offset and small temporal cuts.
This is NOT a reproduction of the paper's five-layer phoneme GRU/LM decoder.
"""

# ============================================================================
# 1. SETUP & IMPORTS
# ============================================================================
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
# Install metric dependencies if unavailable
import sys
import subprocess
import importlib.util
required = {"jiwer": "jiwer", "rouge_score": "rouge-score"}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing]) 

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import warnings
import random

# Reduce irrelevant warnings
warnings.filterwarnings('ignore')

# Konfiguracja
CONFIG = {
    'data_dir': '/kaggle/input/competitions/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 64,       # Efektywny rozmiar wsadu
    'num_epochs': 100,       # Zwiększono, bo SpecAugment wymaga dłuższego treningu
    'hidden_dim': 512,      # Wymiar LSTM
    'num_layers': 3,        # Liczba warstw LSTM
    'dropout': 0.5,         # Silny dropout
    'learning_rate': 1e-3,  # Standardowy LR dla LSTM
    'seed': 42,
    'eval_every': 5,
    # Values from the authors' publicly released rnn_args.yaml
    'smooth_sigma_bins': 2.0,
    'smooth_kernel_size': 100,
    'white_noise_std': 1.0,
    'constant_offset_std': 0.2,
    'random_cut': 3,  # random 0, 1, or 2 bins; training only
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


seed_everything(CONFIG['seed'])
print(f"Seed: {CONFIG['seed']}")
print(f"Device: {CONFIG['device']}")
print(f"PyTorch version: {torch.__version__}")


# ============================================================================
# 2. DATA LOADING & DATASET
# ============================================================================
def load_split(data_dir, split='train'):
    """Load the competition's 512 pre-extracted, 20-ms-bin neural features."""
    from glob import glob
    
    pattern = f'{data_dir}/**/data_{split}.hdf5'
    files = sorted(glob(pattern, recursive=True))
    
    print(f"\nLoading {split} split...")
    
    all_data = {k: [] for k in ['neural', 'n_steps', 'sentence']}
    
    for filepath in tqdm(files):
        with h5py.File(filepath, 'r') as f:
            for trial_key in f.keys():
                trial = f[trial_key]
                # Dane neuronowe
                neural = trial['input_features'][:]
                n_steps = trial.attrs['n_time_steps']
                # Etykieta tekstowa
                sentence = trial.attrs.get('sentence_label')
                if sentence and isinstance(sentence, bytes):
                    sentence = sentence.decode('utf-8')
                
                all_data['neural'].append(neural)
                all_data['n_steps'].append(n_steps)
                all_data['sentence'].append(sentence)
    
    print(f"✓ Loaded {len(all_data['neural'])} samples")
    return all_data

class BrainToTextDataset(Dataset):
    def __init__(self, data, char2idx=None):
        self.neural = data['neural']
        self.n_steps = data['n_steps']
        self.sentences = data['sentence']
        
        if char2idx is None:
            self.char2idx = self._build_vocab()
        else:
            self.char2idx = char2idx
        
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
    
    def _build_vocab(self):
        chars = set()
        for sent in self.sentences:
            if sent: chars.update(sent.lower())
        chars = sorted(list(chars))
        char2idx = {'<BLANK>': 0}
        for i, ch in enumerate(chars, start=1):
            char2idx[ch] = i
        return char2idx
    
    def __len__(self): return len(self.neural)
    
    def __getitem__(self, idx):
        neural = self.neural[idx][:self.n_steps[idx]]
        # Already extracted neural features: no per-trial global Z-score.
        neural = np.asarray(neural, dtype=np.float32)
        
        sentence = self.sentences[idx] if self.sentences[idx] else ""
        target = [self.char2idx.get(ch.lower(), 0) for ch in sentence]
        
        return {
            'neural': torch.FloatTensor(neural),
            'target': torch.LongTensor(target),
            'length': len(neural),
            'target_length': len(target),
            'sentence': sentence
        }

def collate_fn(batch):
    # Sortowanie malejąco dla pack_padded_sequence
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    neurals = [item['neural'] for item in batch]
    targets = [item['target'] for item in batch]
    
    neural_padded = pad_sequence(neurals, batch_first=True)
    target_padded = pad_sequence(targets, batch_first=True)
    lengths = torch.LongTensor([item['length'] for item in batch])
    target_lengths = torch.LongTensor([item['target_length'] for item in batch])
    
    return {
        'neural': neural_padded,
        'target': target_padded,
        'lengths': lengths,
        'target_lengths': target_lengths,
        'sentences': [item['sentence'] for item in batch]
    }


# ============================================================================
# 3. CNN-BiLSTM-CTC (damped oscillator and SpecAugment removed)
# ============================================================================
import torch.nn.functional as F

class ImprovedBrainCTCModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_layers=3,
                 vocab_size=50, dropout=0.5):
        super().__init__()
        # Unchanged convolutional and recurrent backbone from the uploaded notebook.
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(256, hidden_dim, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(2 * hidden_dim, vocab_size)

    def forward(self, x, lengths):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        cnn_lengths = ((lengths.cpu() + 3) // 4).clamp(min=1)
        assert torch.all(cnn_lengths <= x.size(1)), 'CNN output-length mismatch'
        x = pack_padded_sequence(x, cnn_lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        log_probs = torch.log_softmax(self.fc(x), dim=-1)
        return log_probs.transpose(0, 1), cnn_lengths


# Authors' PyTorch baseline preprocessing (not their GRU architecture):
# 512 pre-extracted neural features at 20-ms resolution, smoothed with
# a Gaussian sigma=2 bins, kernel_size=100. Add noise/offset and crop 0-2
# initial bins during training only. No per-trial global Z-score.
def paper_neural_preprocessing(features, lengths, config, training=False):
    # features: [batch, time, 512]; zero-padded by collate_fn.
    if training:
        if config['white_noise_std']:
            features = features + torch.randn_like(features) * config['white_noise_std']
        if config['constant_offset_std']:
            features = features + torch.randn((features.size(0), 1, features.size(2)),
                           device=features.device, dtype=features.dtype) * config['constant_offset_std']
        if config['random_cut']:
            cut = int(np.random.randint(0, config['random_cut']))
            if cut:
                features = features[:, cut:, :]
                lengths = lengths - cut

    size = config['smooth_kernel_size']
    sigma = config['smooth_sigma_bins']
    # Even-sized Gaussian window; asymmetry is negligible at +/-25 sigma.
    offsets = torch.arange(size, device=features.device, dtype=features.dtype) - (size // 2)
    kernel = torch.exp(-0.5 * (offsets / sigma)**2)
    kernel = kernel / kernel.sum()
    channels = features.shape[-1]
    weight = kernel.view(1, 1, size).expand(channels, 1, size).contiguous()
    x = features.transpose(1, 2)
    # Same-length zero-boundary convolution. Data were padded with zeros above.
    x = F.conv1d(F.pad(x, (size // 2, size - 1 - size // 2)),
                 weight, groups=channels)
    return x.transpose(1, 2), lengths


model_check = ImprovedBrainCTCModel(input_dim=512, hidden_dim=512,
                                    num_layers=3, vocab_size=38, dropout=0.5)
print('CNN-BiLSTM-CTC parameters:', sum(p.numel() for p in model_check.parameters()))


# ============================================================================
# 4. OBJECTIVE MATRICS IMPLEMENTATIONS
# ============================================================================

import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu
from jiwer import wer, cer
from rouge_score import rouge_scorer

OUT = Path("/kaggle/working/brain_to_text_results")
OUT.mkdir(parents=True, exist_ok=True)


def evaluate_full_metrics(model, val_loader, idx2char, device, config):
    model.eval()
    refs, preds = [], []
    example = None

    with torch.inference_mode():
        for batch in val_loader:
            neural = batch["neural"].to(device)
            lengths = batch["lengths"]

            neural, lengths = paper_neural_preprocessing(neural, lengths, config, training=False)
            log_probs, out_lengths = model(neural, lengths)
            tokens = log_probs.argmax(dim=-1).cpu()

            for b in range(neural.size(0)):
                sequence = tokens[:int(out_lengths[b]), b].tolist()
                decoded, previous = [], None

                for token in sequence:
                    if token != 0 and token != previous:
                        decoded.append(idx2char.get(token, ""))
                    previous = token

                predicted = "".join(decoded)
                actual = batch["sentences"][b] or ""

                preds.append(predicted)
                refs.append(actual)

                if example is None:
                    signal = batch["neural"][b, :int(lengths[b])].cpu().numpy()
                    example = (signal, predicted, actual)

    # WER, CER, Exact Match
    wer_score = 100 * wer(
        [r.lower() for r in refs],
        [p.lower() for p in preds]
    )
    cer_score = 100 * cer(
        [r.lower() for r in refs],
        [p.lower() for p in preds]
    )
    exact = 100 * np.mean([
        r.lower().strip() == p.lower().strip()
        for r, p in zip(refs, preds)
    ])

    # BLEU and ROUGE
    def tokenize(s):
        return re.findall(
            r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]",
            s.lower()
        )

    rt = [tokenize(s) for s in refs]
    pt = [tokenize(s) for s in preds]

    bleu = {}
    for n in range(1, 5):
        bleu[f"BLEU-{n}"] = 100 * corpus_bleu(
            [[r] for r in rt],
            pt,
            weights=tuple([1 / n] * n)
        )

    # Standard ROUGE-1 and ROUGE-L (rather than custom token-overlap ROUGE-F).
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    rouge1 = [scorer.score(r, p)["rouge1"] for r, p in zip(refs, preds)]
    rougeL = [scorer.score(r, p)["rougeL"] for r, p in zip(refs, preds)]

    metrics = {
        "WER": wer_score,
        "CER": cer_score,
        **bleu,
        "ROUGE-1 P": 100 * np.mean([v.precision for v in rouge1]),
        "ROUGE-1 R": 100 * np.mean([v.recall for v in rouge1]),
        "ROUGE-1 F1": 100 * np.mean([v.fmeasure for v in rouge1]),
        "ROUGE-L P": 100 * np.mean([v.precision for v in rougeL]),
        "ROUGE-L R": 100 * np.mean([v.recall for v in rougeL]),
        "ROUGE-L F1": 100 * np.mean([v.fmeasure for v in rougeL]),
        "Exact Match": exact
    }

    comparison = pd.DataFrame({
        "Actual Text": refs,
        "Predicted Text": preds
    })

    return metrics, comparison, example


def save_best_analysis(metrics, comparison, example):
    # Save numerical results
    pd.DataFrame([{
        "Model": "CNN-BiLSTM-CTC (Card-style preprocessing)", **metrics
    }]).to_csv(
        OUT / "complete_objective_metrics.csv",
        index=False
    )

    comparison.to_csv(
        "/kaggle/working/validation_predictions.csv",
        index=False
    )

    # Figure 1: BLEU and ROUGE
    names = [
        "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
        "ROUGE-1 P", "ROUGE-1 R", "ROUGE-1 F1", "ROUGE-L F1"
    ]
    values = [metrics[k] for k in names]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, values, edgecolor="black")

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{v:.2f}",
            ha="center", fontsize=9
        )

    ax.set_title("Brain-to-Text: BLEU and ROUGE Performance")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, max(values + [5]) * 1.2)
    ax.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        OUT / "BLEU_ROUGE_metrics.png",
        dpi=300, bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)

    # Figure 2: Neural Signal -> Predicted -> Actual
    signal, predicted, actual = example

    sample_wer = 100 * wer(actual.lower(), predicted.lower())

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4.5),
        gridspec_kw={"width_ratios": [2.1, 1.2, 1.2]}
    )

    display_limit = max(float(np.percentile(np.abs(signal), 99)), 1e-6)
    im = axes[0].imshow(
        signal.T, aspect="auto", origin="lower",
        cmap="RdBu_r", vmin=-display_limit, vmax=display_limit,
        extent=[0, len(signal) * 0.02, 0, signal.shape[1]]
    )

    axes[0].set_title("1. Neural Signal")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Neural Feature Index")
    fig.colorbar(im, ax=axes[0])

    for ax, title, sentence, color in [
        (axes[1], "2. Predicted Text", predicted, "#eef4ff"),
        (axes[2], "3. Actual Text", actual, "#eff8ef")
    ]:
        ax.axis("off")
        ax.set_title(title, fontweight="bold")
        ax.text(
            0.5, 0.5, textwrap.fill(sentence, 30),
            ha="center", va="center", fontsize=12,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.7", fc=color)
        )

    fig.suptitle(
        f"Neural-to-Text Decoding | Sample WER: {sample_wer:.2f}%",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    plt.savefig(
        OUT / "Neural_to_Text_Decoding.png",
        dpi=300, bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)

    print("Best-model results and figures saved:", OUT)


def evaluate_wer_cer(model, loader, idx2char, device, config):
    """Greedy CTC WER/CER measured without training-time augmentation."""
    model.eval()
    refs, preds = [], []
    with torch.inference_mode():
        for batch in loader:
            neural = batch['neural'].to(device)
            neural, lengths = paper_neural_preprocessing(neural, batch['lengths'], config, training=False)
            log_probs, out_lengths = model(neural, lengths)
            tokens = log_probs.argmax(dim=-1).cpu()
            for b in range(neural.size(0)):
                previous, chars = None, []
                for token in tokens[:int(out_lengths[b]), b].tolist():
                    if token != 0 and token != previous:
                        chars.append(idx2char.get(token, ''))
                    previous = token
                preds.append(''.join(chars).lower())
                refs.append((batch['sentences'][b] or '').lower())
    return {'WER': 100 * wer(refs, preds), 'CER': 100 * cer(refs, preds)}


def plot_train_validation_curves(history):
    df = pd.DataFrame(history)
    if df.empty:
        raise ValueError('No metric checkpoints were recorded.')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, metric in enumerate(('WER', 'CER')):
        ax[j].plot(df['Epoch'], df[f'Train {metric}'], marker='o', label='Training')
        ax[j].plot(df['Epoch'], df[f'Val {metric}'], marker='s', label='Validation')
        ax[j].set(xlabel='Epoch', ylabel=f'{metric} (%)', title=f'Training vs Validation {metric}')
        ax[j].grid(alpha=0.25)
        ax[j].legend()
    plt.tight_layout()
    plt.savefig(OUT / 'training_validation_WER_CER.png', dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print('Training/validation curves:', OUT / 'training_validation_WER_CER.png')


# ============================================================================
# 5. TRAINING LOOP
# ============================================================================
def train_model(train_loader, train_eval_loader, val_loader, char2idx, config):
    model = ImprovedBrainCTCModel(
        input_dim=512,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        vocab_size=len(char2idx),
        dropout=config['dropout']
    ).to(config['device'])
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    
    # Preserve the original OneCycleLR training schedule
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['learning_rate'],
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader)
    )
    
    idx2char = {v: k for k, v in char2idx.items()}
    best_wer = float('inf')
    metrics_history = []
    curves_history = []
    
    print("\nStarting training (CNN-BiLSTM-CTC + Card-style neural preprocessing)...")
    
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}'):
            neural = batch['neural'].to(config['device'])
            target = batch['target'].to(config['device'])
            lengths = batch['lengths']
            target_lengths = batch['target_lengths']
            
            # Paper-aligned training preprocessing: noise, offset, jitter, Gaussian smoothing.
            neural, lengths = paper_neural_preprocessing(neural, lengths, config, training=True)
            # Forward
            log_probs, cnn_lengths = model(neural, lengths)
            
            # Loss
            loss = criterion(log_probs, target, cnn_lengths, target_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        
        # Matched, unaugmented train/validation error rates at regular epochs.
        if (epoch + 1) % config['eval_every'] == 0 or (epoch + 1) == config['num_epochs']:
            train_errors = evaluate_wer_cer(model, train_eval_loader, idx2char,
                                            config['device'], config)
            val_errors = evaluate_wer_cer(model, val_loader, idx2char,
                                          config['device'], config)
            curves_history.append({
                'Epoch': epoch + 1, 'Train WER': train_errors['WER'],
                'Val WER': val_errors['WER'], 'Train CER': train_errors['CER'],
                'Val CER': val_errors['CER']
            })
            pd.DataFrame(curves_history).to_csv(OUT / 'train_validation_WER_CER.csv', index=False)
            print(f"Epoch {epoch+1}: Train WER {train_errors['WER']:.2f}%, "
                  f"Val WER {val_errors['WER']:.2f}%, "
                  f"Train CER {train_errors['CER']:.2f}%, "
                  f"Val CER {val_errors['CER']:.2f}%")

            metrics, comparison, example = evaluate_full_metrics(
                model, val_loader, idx2char, config["device"], config
            )

            val_wer = metrics["WER"]

            metrics_history.append({
                "Epoch": epoch + 1,
                **metrics
            })

            pd.DataFrame(metrics_history).to_csv(
                OUT / "metrics_by_epoch.csv",
                index=False
            )

            print(f"\n=== EPOCH {epoch + 1} VALIDATION ===")
            print(f"WER: {metrics['WER']:.2f}%")
            print(f"CER: {metrics['CER']:.2f}%")
            print(f"BLEU-1: {metrics['BLEU-1']:.2f}%")
            print(f"BLEU-4: {metrics['BLEU-4']:.2f}%")
            print(f"ROUGE-1 F1: {metrics['ROUGE-1 F1']:.2f}%")
            print(f"ROUGE-L F1: {metrics['ROUGE-L F1']:.2f}%")
            print(f"Exact Match: {metrics['Exact Match']:.2f}%")

            # Save results only when validation WER improves
            if val_wer < best_wer:
                best_wer = val_wer

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "char2idx": char2idx,
                    "config": config,
                    "wer": val_wer
                }, "/kaggle/working/best_model.pt")

                save_best_analysis(
                    metrics, comparison, example
                )

                print(
                    f"New best checkpoint saved | "
                    f"WER: {best_wer:.2f}%"
                )

    plot_train_validation_curves(curves_history)
    return model, best_wer, curves_history


# ============================================================================
# 6. SUBMISSION GENERATION (Greedy Decoding)
# ============================================================================
def load_test_data_for_submission(data_dir):
    from glob import glob
    print("\nLoading test data...")
    pattern = f'{data_dir}/**/data_test.hdf5'
    files = sorted(glob(pattern, recursive=True))
    all_samples = []
    sample_id = 0
    for filepath in tqdm(files):
        with h5py.File(filepath, 'r') as f:
            trial_keys = [k for k in f.keys() if 'trial' in k.lower()]
            for trial_key in trial_keys:
                trial = f[trial_key]
                if 'input_features' not in trial: continue
                features = trial['input_features'][:]
                n_steps = trial.attrs['n_time_steps']
                features = features[:n_steps]
                features = np.asarray(features, dtype=np.float32)  # raw pre-extracted 20-ms features
                all_samples.append({'id': sample_id, 'features': torch.FloatTensor(features)})
                sample_id += 1
    return all_samples

def generate_predictions(model, test_samples, idx2char, device, config, batch_size=64):
    model.eval()
    predictions = []
    print("\nGenerating predictions...")
    with torch.no_grad():
        for i in tqdm(range(0, len(test_samples), batch_size)):
            batch = test_samples[i:i+batch_size]
            features = [s['features'] for s in batch]
            lengths = torch.LongTensor([len(f) for f in features])
            features_padded = pad_sequence(features, batch_first=True).to(device)
            sorted_lengths, sorted_idx = lengths.sort(descending=True)
            features_sorted = features_padded[sorted_idx]
            
            features_sorted, sorted_lengths = paper_neural_preprocessing(
                features_sorted, sorted_lengths, config, training=False)
            log_probs, cnn_lengths = model(features_sorted, sorted_lengths)
            
            _, max_indices = log_probs.max(dim=-1)
            batch_preds = []
            for b in range(max_indices.size(1)):
                valid_len = cnn_lengths[b].item()
                seq = max_indices[:valid_len, b].cpu().numpy()
                decoded = []
                prev = None
                for token in seq:
                    if token != 0 and token != prev:
                        decoded.append(idx2char.get(token, ''))
                    prev = token
                batch_preds.append(''.join(decoded))
            
            unsorted_preds = [''] * len(batch_preds)
            for i, pred in zip(sorted_idx.tolist(), batch_preds):
                unsorted_preds[i] = pred
            predictions.extend(unsorted_preds)
    return predictions


def verify_ctc_alignments(dataset, label):
    """Check that the CNN provides enough timesteps to align each CTC target."""
    invalid = []
    for i, (steps, sentence) in enumerate(zip(dataset.n_steps, dataset.sentences)):
        ids = [dataset.char2idx.get(ch.lower(), 0) for ch in (sentence or "")]
        repeats = sum(a == b for a, b in zip(ids, ids[1:]))
        minimum = len(ids) + repeats
        # Training augmentation can cut up to random_cut - 1 starting bins.
        cut = CONFIG['random_cut'] - 1 if label == 'Train' else 0
        cnn_steps = (int(steps) - cut + 3) // 4
        if cnn_steps < minimum:
            invalid.append((i, cnn_steps, minimum))
    print(f"{label}: CTC-alignment check: {len(dataset) - len(invalid)}/{len(dataset)} valid")
    if invalid:
        raise ValueError(
            f"{label}: {len(invalid)} examples have insufficient CTC timesteps; "
            f"first invalid sample (index, available, required) = {invalid[0]}. "
            "Review the data/sequence lengths rather than silently zeroing their loss."
        )


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================
def main():
    print("="*80)
    print("BRAIN-TO-TEXT '25 - CNN-BiLSTM-CTC (CARD-STYLE PREPROCESSING)")
    print("="*80)
    
    # 1. Load Data
    train_data = load_split(CONFIG['data_dir'], 'train')
    val_data = load_split(CONFIG['data_dir'], 'val')
    
    if not train_data['neural'] or not val_data['neural']:
        raise FileNotFoundError('Training or validation HDF5 files not found at CONFIG["data_dir"].')

    # 2. Datasets
    train_dataset = BrainToTextDataset(train_data)
    val_dataset = BrainToTextDataset(val_data, char2idx=train_dataset.char2idx)
    
    # Verify CTC feasibility using exact CNN output lengths and repeated-label requirements.
    verify_ctc_alignments(train_dataset, "Train")
    verify_ctc_alignments(val_dataset, "Validation")

    train_eval_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                                   shuffle=False, collate_fn=collate_fn, num_workers=2,
                                   worker_init_fn=seed_worker)

    train_gen = torch.Generator().manual_seed(CONFIG['seed'])
    val_gen = torch.Generator().manual_seed(CONFIG['seed'] + 1)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, collate_fn=collate_fn, num_workers=2,
                              worker_init_fn=seed_worker, generator=train_gen)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, collate_fn=collate_fn, num_workers=2,
                            worker_init_fn=seed_worker, generator=val_gen)

    # 3. Train (original architecture and training schedule unchanged)
    model, best_wer, history = train_model(
        train_loader, train_eval_loader, val_loader, train_dataset.char2idx, CONFIG)
    print(f"\nTraining complete | Best validation WER: {best_wer:.2f}%")

    # 4. Reload BEST checkpoint, not the final-epoch weights.
    checkpoint_path = "/kaggle/working/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'], weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    idx2char = {v: k for k, v in checkpoint['char2idx'].items()}

    best_metrics, best_predictions, _ = evaluate_full_metrics(
        model, val_loader, idx2char, CONFIG['device'], CONFIG
    )
    best_predictions.to_csv('/kaggle/working/validation_predictions.csv', index=False)
    print(f"Best-checkpoint validation: WER {best_metrics['WER']:.2f}%, "
          f"CER {best_metrics['CER']:.2f}%, ROUGE-L F1 {best_metrics['ROUGE-L F1']:.2f}%")
    print(best_predictions.head(5).to_string(index=False))

    # 5. Run test inference if the competition's test HDF5 files are mounted.
    test_samples = load_test_data_for_submission(CONFIG['data_dir'])
    if test_samples:
        predictions = generate_predictions(model, test_samples, idx2char,
                                           CONFIG['device'], CONFIG)
        df = pd.DataFrame({'id': [s['id'] for s in test_samples], 'text': predictions})
        df = df.sort_values('id')
        df.to_csv('/kaggle/working/submission.csv', index=False)
        print(f"\nSaved {len(df)} test predictions to /kaggle/working/submission.csv")
        print(df.head(5).to_string(index=False))
        print("Test WER/CER not available: competition test labels are not provided here.")
        print("Note: IDs follow the original notebook's sequential indexing; "
              "verify competition submission ID mapping before upload.")
    else:
        print("No data_test.hdf5 files found; validation evaluation completed, "
              "but held-out test predictions cannot be generated without test data.")

if __name__ == "__main__":
    main()

