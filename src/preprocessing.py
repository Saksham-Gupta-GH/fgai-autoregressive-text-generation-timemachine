import re
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


class Vocabulary:
    """Vocabulary mapping for word-level tokenization."""
    def __init__(self, pad_token="<pad>", unk_token="<unk>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {}
        self.idx2word = {}
        
        # Reserved tokens
        self.pad_idx = self.add_word(self.pad_token)
        self.unk_idx = self.add_word(self.unk_token)

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            return idx
        return self.word2idx[word]

    def build_from_words(self, words, min_freq=2, max_vocab_size=6000):
        counter = Counter(words)
        # Sort words by frequency
        most_common = counter.most_common(max_vocab_size - len(self.word2idx))
        for word, count in most_common:
            if count >= min_freq:
                self.add_word(word)

    def __len__(self):
        return len(self.word2idx)

    def word_to_idx(self, word):
        return self.word2idx.get(word, self.unk_idx)

    def idx_to_word(self, idx):
        return self.idx2word.get(idx, self.unk_token)

    def encode(self, words):
        return [self.word_to_idx(w) for w in words]

    def decode(self, indices):
        return [self.idx_to_word(idx) for idx in indices]


def strip_gutenberg_headers(text):
    """Strips Gutenberg license and legal preamble/postamble."""
    start_match = re.search(r"\*\*\* START OF [^\*]*\*\*\*", text, re.IGNORECASE)
    if start_match:
        text = text[start_match.end():]
    end_match = re.search(r"\*\*\* END OF [^\*]*\*\*\*", text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]
    return text.strip()


def clean_and_tokenize(text):
    """
    Cleans text and extracts word-level tokens while preserving essential punctuation
    as distinct tokens to maintain poetic and dialogue structure.
    """
    text = strip_gutenberg_headers(text)
    text = text.lower()
    tokens = re.findall(r"\b\w+(?:'\w+)?|[.,!?:;]", text)
    return tokens


def get_dataset_stats(raw_text, tokens, vocab):
    """
    Computes statistical characteristics of the textual dataset.
    """
    raw_text = strip_gutenberg_headers(raw_text)
    lines = raw_text.splitlines()
    total_chars = len(raw_text)
    total_words = len(tokens)
    total_lines = len(lines)
    unique_words = len(set(tokens))
    vocab_size = len(vocab)
    avg_words_per_line = total_words / max(1, total_lines)
    
    # Calculate word frequency distribution for Zipf analysis
    counter = Counter(tokens)
    top_10 = counter.most_common(10)
    
    stats = {
        "total_characters": total_chars,
        "total_tokens": total_words,
        "total_lines": total_lines,
        "unique_raw_tokens": unique_words,
        "vocab_size": vocab_size,
        "avg_words_per_line": avg_words_per_line,
        "top_10_words": top_10
    }
    return stats


class TextSequenceDataset(Dataset):
    """
    Autoregressive sliding-window dataset:
    Given a sequence of length N: [w_1, w_2, ..., w_N],
    predict target next word w_{N+1}.
    """
    def __init__(self, token_indices, seq_length=8):
        self.seq_length = seq_length
        self.inputs = []
        self.targets = []
        
        # Build sliding windows
        for i in range(len(token_indices) - seq_length):
            self.inputs.append(token_indices[i : i + seq_length])
            self.targets.append(token_indices[i + seq_length])
            
        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def prepare_data(file_path, seq_length=8, max_vocab_size=3000, min_freq=1, val_split=0.1, test_split=0.1, batch_size=64, max_samples=None, seed=42):
    """
    Full pipeline to load, preprocess, tokenize, and create train/val/test DataLoaders.
    Uses chronological contiguous splitting (80% Train, 10% Validation, 10% Test)
    and fits vocabulary strictly on the training set to prevent data leakage.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokens = clean_and_tokenize(raw_text)
    if max_samples is not None and max_samples < len(tokens):
        tokens = tokens[:max_samples]
        
    total_tokens = len(tokens)
    val_count = int(total_tokens * val_split)
    test_count = int(total_tokens * test_split)
    train_count = total_tokens - val_count - test_count
    
    train_tokens = tokens[:train_count]
    val_tokens = tokens[train_count : train_count + val_count]
    test_tokens = tokens[train_count + val_count :]
    
    # Fit vocabulary strictly on training tokens (prevents vocabulary leakage)
    vocab = Vocabulary()
    vocab.build_from_words(train_tokens, min_freq=min_freq, max_vocab_size=max_vocab_size)
    
    # Encode splits
    train_indices = vocab.encode(train_tokens)
    val_indices = vocab.encode(val_tokens)
    test_indices = vocab.encode(test_tokens)
    
    # Construct independent sliding-window sequence datasets
    train_dataset = TextSequenceDataset(train_indices, seq_length=seq_length)
    val_dataset = TextSequenceDataset(val_indices, seq_length=seq_length)
    test_dataset = TextSequenceDataset(test_indices, seq_length=seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    stats = get_dataset_stats(raw_text, tokens, vocab)
    stats["train_tokens"] = len(train_tokens)
    stats["val_tokens"] = len(val_tokens)
    stats["test_tokens"] = len(test_tokens)
    stats["train_sequences"] = len(train_dataset)
    stats["val_sequences"] = len(val_dataset)
    stats["test_sequences"] = len(test_dataset)
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "vocab": vocab,
        "stats": stats,
        "tokens": tokens,
        "raw_text": raw_text
    }
