import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter


def sample_with_temperature(logits, temperature=1.0):
    """
    Applies temperature scaling to logits and draws a token from the resulting multinomial distribution.
    - Low T (< 0.5): Greedy-leaning, confident, conservative, prone to repetition.
    - Moderate T (~ 0.7 - 1.0): Balanced creativity and coherence.
    - High T (> 1.2): High diversity, high randomness, higher risk of syntactic breakdown.
    """
    if temperature <= 1e-4:
        return torch.argmax(logits, dim=-1).item()
    
    # Scale logits
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    
    # Sample from multinomial distribution
    sampled_idx = torch.multinomial(probabilities, num_samples=1).item()
    return sampled_idx


def generate_text(model, vocab, seed_text, max_new_tokens=30, seq_length=8, temperature=1.0, device="cpu"):
    """
    Autoregressively generates text token-by-token given a seed sequence.
    Maintains a sliding window of length `seq_length`.
    """
    model.eval()
    from src.preprocessing import clean_and_tokenize
    
    seed_tokens = clean_and_tokenize(seed_text)
    if len(seed_tokens) == 0:
        seed_tokens = ["to", "be"]
        
    generated_tokens = list(seed_tokens)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Take the most recent `seq_length` tokens
            context = generated_tokens[-seq_length:]
            indices = vocab.encode(context)
            
            # If context is shorter than seq_length, pad on the left
            if len(indices) < seq_length:
                indices = [vocab.pad_idx] * (seq_length - len(indices)) + indices
                
            input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
            logits = model(input_tensor) # (1, vocab_size)
            logits = logits.squeeze(0)   # (vocab_size)
            
            # Disallow generating <pad> or <unk> directly
            logits[vocab.pad_idx] = -float("inf")
            logits[vocab.unk_idx] = -float("inf")
            
            next_idx = sample_with_temperature(logits, temperature=temperature)
            next_word = vocab.idx_to_word(next_idx)
            generated_tokens.append(next_word)
            
    # Format generated output nicely
    # Reconstruct text respecting punctuation
    formatted_words = []
    for token in generated_tokens:
        if token in {".", ",", "!", "?", ":", ";"} and len(formatted_words) > 0:
            formatted_words[-1] = formatted_words[-1] + token
        else:
            formatted_words.append(token)
            
    return " ".join(formatted_words), generated_tokens


def calculate_generation_metrics(tokens, num_seed_tokens=0):
    """
    Calculates quantitative metrics strictly on NEWLY GENERATED tokens (excluding seed prompt):
    - Distinct-1: Ratio of unique unigrams to total generated tokens
    - Distinct-2: Ratio of unique bigrams to total generated bigrams
    - Trigram Repetition Rate: Percentage of repeated trigrams (3-grams appearing > 1 time)
    - Longest Loop: Length of repeating pattern (if looping detected)
    """
    gen_tokens = tokens[num_seed_tokens:] if num_seed_tokens < len(tokens) else tokens
    if len(gen_tokens) == 0:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "trigram_repeat_rate": 0.0, "longest_loop": 0}
        
    unigrams = gen_tokens
    distinct_1 = len(set(unigrams)) / len(unigrams)
    
    bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams) - 1)]
    distinct_2 = len(set(bigrams)) / max(1, len(bigrams))
    
    # Trigram repetition rate
    trigrams = [f"{unigrams[i]}_{unigrams[i+1]}_{unigrams[i+2]}" for i in range(len(unigrams) - 2)]
    if len(trigrams) > 0:
        trigram_counts = Counter(trigrams)
        repeated_trigrams = sum(count - 1 for count in trigram_counts.values() if count > 1)
        trigram_repeat_rate = repeated_trigrams / len(trigrams)
    else:
        trigram_repeat_rate = 0.0
        
    # Longest repeated loop (find repeated token n-grams)
    longest_loop = 0
    for length in range(2, min(8, len(unigrams) // 2 + 1)):
        for i in range(len(unigrams) - 2 * length + 1):
            chunk1 = unigrams[i : i + length]
            chunk2 = unigrams[i + length : i + 2 * length]
            if chunk1 == chunk2:
                longest_loop = max(longest_loop, length)
                
    return {
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "trigram_repeat_rate": round(trigram_repeat_rate, 4),
        "longest_loop": longest_loop
    }
