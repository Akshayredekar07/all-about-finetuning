# **Tokenization Fundamentals**
 Deep-dive reference: Tokens · Token IDs · Attention Mask · Token Type IDs · Encoding · Decoding  


## Table of Contents
1. [What is a Token?](#1-what-is-a-token)
2. [Vocabulary & Token IDs](#2-vocabulary--token-ids)
3. [Encoding — Text → Tensor](#3-encoding--text--tensor)
4. [Attention Mask — What and Why](#4-attention-mask--what-and-why)
5. [Token Type IDs — What and Why](#5-token-type-ids--what-and-why)
6. [Decoding — Tensor → Text](#6-decoding--tensor--text)
7. [Special Tokens Deep-Dive](#7-special-tokens-deep-dive)
8. [From Scratch in Pure PyTorch](#8-from-scratch-in-pure-pytorch)
9. [HuggingFace Transformers — Full Pipeline](#9-huggingface-transformers--full-pipeline)
10. [Batching — Padding & Truncation](#10-batching--padding--truncation)
11. [Quick Reference](#11-quick-reference)


## 1. What is a Token?

A **token** is the atomic unit of text that a model sees. It is **not** a character, **not** a word — it is a **subword chunk** produced by a tokenization algorithm.

### Mental model

```
Raw Text  →  [Tokenizer]  →  Tokens  →  [Vocab Lookup]  →  Token IDs  →  [Model]
```

### Why subwords and not whole words?

| Approach | Problem |
|---|---|
| **Character-level** | Sequences become huge; no semantic meaning per unit |
| **Word-level** | Vocabulary explodes (millions of words); OOV (out-of-vocabulary) for rare words |
| **Subword (BPE/Unigram/WordPiece)** | ✅ Compact vocab + handles any word via splits |

### Concrete example with GPT-2

```
Input text : "Hello, I am learning Transformers!"

Tokenization result:
┌─────┬──────────────┬──────────────────────────────────┐
│ Pos │   Token      │   What it represents              │
├─────┼──────────────┼──────────────────────────────────┤
│  0  │  Hello       │  The word Hello (no leading space)│
│  1  │  ,           │  Punctuation                      │
│  2  │  ĠI          │  " I" (Ġ = space before the word) │
│  3  │  Ġam         │  " am"                            │
│  4  │  Ġlearning   │  " learning"                      │
│  5  │  ĠTransform  │  " Transform"                     │
│  6  │  ers         │  "ers" (continuation of Transform)│
│  7  │  !           │  Punctuation                      │
└─────┴──────────────┴──────────────────────────────────┘
```

> **Tip:** "Transformers" is split into `ĠTransform` + `ers` because the BPE merge table does not have a single rule merging the full word `Transformers` as one token — it stops at `Transform` + `ers`.

### The Ġ (GPT-2) and ▁ (T5/Llama) markers explained

Both are **space markers** — they encode whitespace as part of the token rather than a separate character.

```
GPT-2   uses   Ġ  (Unicode: U+0120)  →  prefix = "there was a space before me"
T5/Llama uses  ▁  (Unicode: U+2581)  →  same semantic meaning, different char

"Hello world"
  GPT-2  → ['Hello', 'Ġworld']       # first word has no Ġ
  T5     → ['▁Hello', '▁world']      # first word gets ▁ too (SP always marks)
```

---

## 2. Vocabulary & Token IDs

### What is the vocabulary?

The vocabulary is a **fixed dictionary**: `{ token_string → integer_id }`.  
It is built **once** during tokenizer training and frozen. The model only ever sees integer IDs, never raw strings.

```python
# Peeking at the GPT-2 vocabulary
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")

print(tok.vocab_size)          # 50257
print(tok.get_vocab()['Hello']) # 15496
print(tok.get_vocab()['Ġam'])   # 716

# Reverse lookup: ID → token string
inv_vocab = {v: k for k, v in tok.get_vocab().items()}
print(inv_vocab[15496])         # 'Hello'
```

### Vocabulary as a lookup table (what the model physically holds)

Inside every Transformer model, there is an **Embedding Matrix**:

```
Embedding Matrix E  ∈  ℝ^(V × d_model)
  V      = vocab size (e.g., 50257 for GPT-2)
  d_model = embedding dimension (e.g., 768 for GPT-2 base)

Each row i  = the learned embedding vector for token ID i

E[0]     = embedding vector for token 0
E[15496] = embedding vector for "Hello"
E[716]   = embedding vector for "Ġam"
```

### How Token ID lookup works (embedding forward pass)

```
input_ids = [15496, 11, 314, 716, 4673, 48108, 68, 0]
              Hello  ,   ĠI  Ġam  Ġlea  ĠTrans  ers  !

                ↓ nn.Embedding(vocab_size, d_model) ↓

Output shape: [seq_len, d_model] = [8, 768]

Row 0 → E[15496]  = [0.12, -0.34, 0.89, ...]   ← vector for Hello
Row 1 → E[11]     = [0.55, -0.12, 0.01, ...]   ← vector for ,
Row 2 → E[314]    = [-0.07, 0.62, 0.44, ...]   ← vector for ĠI
...
```

```python
import torch
import torch.nn as nn

vocab_size = 50257
d_model    = 768

# This is exactly what lives inside GPT-2
embedding = nn.Embedding(vocab_size, d_model)

input_ids = torch.tensor([[15496, 11, 314, 716, 4673, 48108, 68, 0]])
# shape: [batch=1, seq_len=8]

embedded = embedding(input_ids)
# shape: [1, 8, 768] — each token ID replaced by its 768-dim vector
print(embedded.shape)   # torch.Size([1, 8, 768])
```

---

## 3. Encoding — Text → Tensor

Encoding is the full pipeline from raw string to model-ready tensors.

### The 4 stages

```
Stage 1: Normalization    →  clean the text (unicode, lowercase if needed)
Stage 2: Pre-tokenization →  rough split (by whitespace/punctuation)
Stage 3: Subword splitting →  BPE/Unigram/WordPiece merges/splits
Stage 4: ID mapping        →  token strings → integer IDs from vocab
```

### Full dry-run for GPT-2

```
Input: "Hello, I am learning Transformers!"

──────────────────────────────────────────────
STAGE 1 — Normalization
  GPT-2: NO lowercasing, NFC unicode normalization
  → "Hello, I am learning Transformers!"   (unchanged)

──────────────────────────────────────────────
STAGE 2 — Byte-level representation
  Every character mapped to its byte(s)
  'H' → 72,  'e' → 101,  'l' → 108 ...
  Space → byte 32 → represented as 'Ġ' in vocab

──────────────────────────────────────────────
STAGE 3 — BPE merges applied
  Start: individual bytes/chars
  Apply merge rules from merge table (learned):
    ('T','r') → 'Tr'
    ('Tr','a') → 'Tra'
    ...
    ('Ġ','T','r','a','n','s','f','o','r','m') → 'ĠTransform'
    ('e','r','s') → 'ers'
  
  Final pieces:
  ['Hello', ',', 'ĠI', 'Ġam', 'Ġlearning', 'ĠTransform', 'ers', '!']

──────────────────────────────────────────────
STAGE 4 — Vocab ID lookup
  'Hello'       → 15496
  ','           → 11
  'ĠI'          → 314
  'Ġam'         → 716
  'Ġlearning'   → 4673
  'ĠTransform'  → 48108
  'ers'         → 68
  '!'           → 0
  
  input_ids = [15496, 11, 314, 716, 4673, 48108, 68, 0]
```

### In HuggingFace — what each method does

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
text = "Hello, I am learning Transformers!"

# ── Method 1: tokenize() ──────────────────────────────────────────────
# Does ONLY Stage 1–3 (subword pieces, NO IDs, NO special tokens)
tokens = tok.tokenize(text)
# ['Hello', ',', 'ĠI', 'Ġam', 'Ġlearning', 'ĠTransform', 'ers', '!']

# ── Method 2: encode() ───────────────────────────────────────────────
# Does Stage 1–4 → returns a plain Python list of IDs
ids = tok.encode(text)
# [15496, 11, 314, 716, 4673, 48108, 68, 0]

# ── Method 3: __call__() ─────────────────────────────────────────────
# Full pipeline → returns dict of tensors (input_ids, attention_mask, etc.)
encoded = tok(text, return_tensors="pt")
# {
#   'input_ids'     : tensor([[15496, 11, 314, 716, 4673, 48108, 68, 0]]),
#   'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
# }

# ── Method 4: convert_tokens_to_ids() ────────────────────────────────
# Low-level: takes already-split tokens → IDs
ids2 = tok.convert_tokens_to_ids(tokens)
# [15496, 11, 314, 716, 4673, 48108, 68, 0]

# ── Method 5: convert_ids_to_tokens() ────────────────────────────────
# Reverse of above: IDs → token strings (no joining)
toks = tok.convert_ids_to_tokens(ids2)
# ['Hello', ',', 'ĠI', 'Ġam', 'Ġlearning', 'ĠTransform', 'ers', '!']
```

---

## 4. Attention Mask — What and Why

### The problem it solves: batching sequences of different lengths

When you process multiple sentences together (a batch), they must all be the **same length** (because tensors must be rectangular). Shorter sequences get **padded** with a dummy token ID (usually 0 or `pad_token_id`).

```
Sentence A: "Hello world"       → [15496, 995]              (len=2)
Sentence B: "I am learning"     → [314, 716, 4673]          (len=3)
Sentence C: "Hi"                → [17250]                   (len=1)

After padding to max_len=3:
Sentence A: [15496, 995,   0]   ← padded at position 2
Sentence B: [314,   716, 4673]  ← no padding needed
Sentence C: [17250,  0,    0]   ← padded at positions 1,2
```

**The model must not attend to padding tokens** — they are meaningless filler. The attention mask tells the model: "ignore these positions."

### What the attention mask is

```
attention_mask is a binary tensor, same shape as input_ids:
  1 → REAL token   → attend to this
  0 → PADDING      → IGNORE this

For our batch above:
input_ids:
  tensor([[15496,   995,     0],
          [  314,   716,  4673],
          [17250,     0,     0]])

attention_mask:
  tensor([[1, 1, 0],
          [1, 1, 1],
          [1, 0, 0]])
```

### How it's used inside the Transformer (self-attention)

The attention score between query Q and key K for each position pair is:

```
score(i, j) = (Q_i · K_j) / √d_k
```

Before the softmax, we apply the mask:

```
masked_score(i, j) = score(i, j)  + mask_bias(i, j)

where:
  mask_bias(i, j) = 0       if attention_mask[j] == 1  (real token)
  mask_bias(i, j) = -∞      if attention_mask[j] == 0  (padding)

After softmax:
  softmax(-∞) → 0.0   → padding positions get 0 weight → model ignores them
```

### From-scratch implementation in PyTorch

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, attention_mask=None):
    """
    Q, K, V shape : [batch, heads, seq_len, d_k]
    attention_mask: [batch, 1, 1, seq_len]  (broadcastable)
    """
    d_k = Q.size(-1)
    
    # Raw attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # shape: [batch, heads, seq_len, seq_len]
    
    if attention_mask is not None:
        # Where mask == 0 (padding), fill with a very large negative number
        # After softmax, e^(-1e9) ≈ 0 → padding is effectively ignored
        scores = scores.masked_fill(attention_mask == 0, -1e9)
    
    # Softmax over last dim (key positions)
    weights = F.softmax(scores, dim=-1)
    # shape: [batch, heads, seq_len, seq_len]
    
    output = torch.matmul(weights, V)
    # shape: [batch, heads, seq_len, d_k]
    
    return output, weights


# ── Concrete Example ─────────────────────────────────────────────────────────
batch, heads, seq_len, d_k = 1, 1, 3, 4

# Pretend Q=K=V are already projected (just random for illustration)
Q = torch.randn(batch, heads, seq_len, d_k)
K = torch.randn(batch, heads, seq_len, d_k)
V = torch.randn(batch, heads, seq_len, d_k)

# attention_mask for: ["Hello", "world", <PAD>]
raw_mask = torch.tensor([[1, 1, 0]])          # shape: [1, 3]
mask_4d  = raw_mask.unsqueeze(1).unsqueeze(2) # shape: [1, 1, 1, 3] — broadcastable

output, weights = scaled_dot_product_attention(Q, K, V, mask_4d)

print("Attention weights:\n", weights)
# The 3rd column (PAD position) will be ~0.0 after masking
# weights shape: [1, 1, 3, 3]
```

### Visualizing the attention weight matrix

```
Tokens: ["Hello"=pos0, "world"=pos1, "<PAD>"=pos2]
                          attends TO →
                    Hello   world   <PAD>
               ┌──────────────────────────┐
attends FROM   │
Hello    pos0  │  0.55    0.45    0.00  ← PAD gets 0 weight
world    pos1  │  0.30    0.70    0.00
<PAD>    pos2  │  0.50    0.50    0.00  ← (PAD token itself also "attends",
               └──────────────────────────┘  but its output is discarded)

Without masking, PAD would "steal" attention weight:
Hello    pos0  │  0.40    0.35    0.25  ← WRONG: 25% goes to meaningless PAD
```

> ⚠️ **Watch out:** Even with masking, the PAD token's own output vector is present in the batch. In classification tasks you should index only `output[:, 0, :]` (CLS token) or mean-pool over real tokens only, using the mask.

---

## 5. Token Type IDs — What and Why

### The problem it solves: two-segment inputs

Models like BERT are trained on **sentence pairs** — e.g., question + answer, hypothesis + premise. The model needs to know which tokens belong to Segment A and which to Segment B.

```
Task: Natural Language Inference
  Segment A (premise)   : "The cat sat on the mat."
  Segment B (hypothesis): "The cat is on a mat."

After tokenization with BERT:
  [CLS] The cat sat on the mat . [SEP] The cat is on a mat . [SEP]

Token type IDs:
  [  0   0   0   0  0   0   0  0   0    1   1   1  1   1  1  1   1  ]
   CLS  The cat sat on the mat .  SEP  The cat  is on  a  mat .  SEP
   ↑─────────── Segment A ──────────↑  ↑──────── Segment B ─────────↑
```

### What the values mean

```
token_type_ids[i] = 0   →  token belongs to Segment A (sentence 1)
token_type_ids[i] = 1   →  token belongs to Segment B (sentence 2)
```

### How it's used in the model

Inside BERT, the final input embedding is:

```
final_embedding = token_embedding      [from input_ids]
                + position_embedding   [from position IDs]
                + segment_embedding    [from token_type_ids]

segment_embedding comes from a learned matrix S ∈ ℝ^(2 × d_model):
  S[0] = the "I am in segment A" vector
  S[1] = the "I am in segment B" vector
```

```python
import torch
import torch.nn as nn

d_model = 768

# BERT's segment embedding table: 2 segments, d_model dims
segment_embedding = nn.Embedding(2, d_model)

# token_type_ids for the example above (17 tokens total)
token_type_ids = torch.tensor([[
    0, 0, 0, 0, 0, 0, 0, 0, 0,    # Segment A (incl CLS, SEP)
    1, 1, 1, 1, 1, 1, 1, 1         # Segment B (incl SEP)
]])
# shape: [1, 17]

seg_embeds = segment_embedding(token_type_ids)
# shape: [1, 17, 768]
# Each token now has an additional vector encoding its segment membership
```

### Which models use token_type_ids?

| Model | Uses token_type_ids? | Why |
|---|---|---|
| BERT | ✅ Yes | Trained on Next Sentence Prediction (NSP) |
| RoBERTa | ❌ No | Removed NSP; single segment training |
| GPT-2 | ❌ No | Decoder-only, no pairs |
| T5 | ❌ No | Encoder-decoder, no segment IDs |
| Llama-2 | ❌ No | Decoder-only |
| DistilBERT | ✅ Yes | Distilled from BERT |
| ALBERT | ✅ Yes | Sentence Order Prediction (SOP) |
| XLNet | ✅ Yes | Permutation LM, segment-aware |

> 💡 **Tip:** When you call `tokenizer(text_a, text_b)` in HuggingFace, token_type_ids are created automatically if the model uses them. You rarely need to build them manually.

### HuggingFace example — BERT sentence pair

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

sentence_a = "The cat sat on the mat."
sentence_b = "The cat is on a mat."

encoded = tok(sentence_a, sentence_b, return_tensors="pt")

print(encoded["input_ids"])
# tensor([[ 101, 1996, 4937, 2938,  ...  102, 1996, 4937, 2003, ... 102]])
#         [CLS]                      [SEP]                        [SEP]

print(encoded["token_type_ids"])
# tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
#         ↑────── Segment A ─────────↑ ↑──────── Segment B ────↑

print(encoded["attention_mask"])
# tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
# (no padding here — single example, no batch)
```

---

## 6. Decoding — Tensor → Text

Decoding is the reverse: integer IDs → human-readable string.

### The 3 stages

```
Stage 1: ID → token string    (reverse vocab lookup)
Stage 2: Join tokens          (recombine subwords)
Stage 3: Strip special tokens (remove [CLS], [SEP], <s>, </s> if requested)
```

### Dry-run for GPT-2

```
input_ids: [15496, 11, 314, 716, 4673, 48108, 68, 0]

Stage 1 — Reverse lookup (ID → token string):
  15496 → 'Hello'
  11    → ','
  314   → 'ĠI'
  716   → 'Ġam'
  4673  → 'Ġlearning'
  48108 → 'ĠTransform'
  68    → 'ers'
  0     → '!'

  token strings: ['Hello', ',', 'ĠI', 'Ġam', 'Ġlearning', 'ĠTransform', 'ers', '!']

Stage 2 — Join (convert Ġ → space, concatenate):
  'Hello' + ',' + ' I' + ' am' + ' learning' + ' Transform' + 'ers' + '!'
  = "Hello, I am learning Transformers!"

Stage 3 — Strip specials (none here for GPT-2):
  decoded = "Hello, I am learning Transformers!"
```

### decode() vs batch_decode()

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")

ids_single = [15496, 11, 314, 716, 4673, 48108, 68, 0]
ids_batch  = [[15496, 11], [314, 716, 4673]]

# Single sequence
decoded = tok.decode(ids_single)
# "Hello, I am learning Transformers!"

# Batch of sequences
decoded_batch = tok.batch_decode(ids_batch)
# ["Hello,", " I am learning"]

# skip_special_tokens=True strips [CLS], </s>, <s> etc.
tok.decode(ids_single, skip_special_tokens=True)

# clean_up_tokenization_spaces=True fixes spacing artefacts
tok.decode(ids_single, clean_up_tokenization_spaces=True)
```

### From scratch — pure Python decode

```python
# Build a simple decoder from vocabulary
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
inv_vocab = {v: k for k, v in tok.get_vocab().items()}

def decode_from_scratch(ids, inv_vocab, special_token_ids=None):
    """
    ids              : list of integer token IDs
    inv_vocab        : dict {id -> token_string}
    special_token_ids: set of IDs to skip (e.g. {50256})
    """
    if special_token_ids is None:
        special_token_ids = set()
    
    tokens = []
    for id_ in ids:
        if id_ in special_token_ids:
            continue
        tokens.append(inv_vocab.get(id_, "<unk>"))
    
    # Join tokens, converting Ġ → space
    text = "".join(tokens).replace("Ġ", " ").strip()
    return text

ids = [15496, 11, 314, 716, 4673, 48108, 68, 0]
print(decode_from_scratch(ids, inv_vocab))
# "Hello, I am learning Transformers!"
```

---

## 7. Special Tokens Deep-Dive

Special tokens are **non-natural-language** tokens that carry structural meaning. Each model family has its own set.

### Master table

| Token | Common ID | Purpose | Added when |
|---|---|---|---|
| `[CLS]` | 101 (BERT) | Classification representation; first token | Auto, BERT/DistilBERT |
| `[SEP]` | 102 (BERT) | Separates segments; marks sequence end | Auto, BERT |
| `[PAD]` | 0 (BERT), 1 (T5) | Padding to equal length | Explicit padding |
| `[MASK]` | 103 (BERT) | Masked token for MLM training | During MLM |
| `<s>` | 1 (Llama-2, RoBERTa) | Beginning of sequence (BOS) | Auto prepend |
| `</s>` | 2 (Llama-2), 1 (T5) | End of sequence (EOS) | Auto append (T5), manual (Llama) |
| `<unk>` | 0 (T5, Llama) | Unknown token fallback | When token not in vocab |
| `<pad>` | 0 (T5) | Padding | Explicit |
| `<extra_id_N>` | T5 only | Sentinel span corruption tokens | During T5 pre-training |
| `eos` | 50256 (GPT-2) | End of sequence | Manual |

### Model-by-model cheatsheet

```
GPT-2
  BOS: None (not added automatically)
  EOS: 50256  (must add manually)
  PAD: None   (must set = eos for batching)
  Format: raw text → [tokens] → [tokens, EOS]  (EOS optional)

BERT
  BOS/CLS: [CLS] (101)   auto-prepended
  SEP:     [SEP] (102)   auto-appended; between sentences
  PAD:     [PAD] (0)     explicit padding
  Format: [CLS] sent_A [SEP]         (single)
          [CLS] sent_A [SEP] sent_B [SEP]  (pair)

T5
  EOS: </s> (1)   auto-appended
  PAD: <pad> (0)  explicit
  BOS: None (decoder starts with PAD as decoder_start_token_id)
  Format: raw text → [tokens, </s>]

Llama-2
  BOS: <s> (1)    auto-prepended
  EOS: </s> (2)   NOT auto-appended — add manually
  PAD: None       set = eos manually
  Format: <s> [tokens]
  Chat:   <s>[INST] user_msg [/INST] model_resp </s>
```

### Checking and adding special tokens in HuggingFace

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")

# View all special tokens
print(tok.special_tokens_map)
# {'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>'}

print(tok.all_special_ids)   # [50256]
print(tok.all_special_tokens) # ['<|endoftext|>']

# Add a new special token (e.g., for fine-tuning)
tok.add_special_tokens({"pad_token": "[PAD]"})
# Now must resize model embeddings:
# model.resize_token_embeddings(len(tok))

# Encode WITH special tokens (BERT adds [CLS], [SEP])
tok_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
enc = tok_bert("Hello world", add_special_tokens=True)
# [101, 7592, 2088, 102]  ← [CLS] Hello world [SEP]

enc_no = tok_bert("Hello world", add_special_tokens=False)
# [7592, 2088]  ← no specials
```

---

## 8. From Scratch in Pure PyTorch

Building a minimal BPE tokenizer + Transformer input pipeline from scratch.

### 8.1 — Build a character-level vocabulary (baseline)

```python
import torch

text_corpus = "Hello, I am learning Transformers! Transformers are great."

# Build char-level vocab
chars = sorted(set(text_corpus))
vocab = {ch: i for i, ch in enumerate(chars)}
vocab["<PAD>"] = len(vocab)
vocab["<UNK>"] = len(vocab)

inv_vocab = {v: k for k, v in vocab.items()}

print(f"Vocab size: {len(vocab)}")
print(f"Vocab: {vocab}")
```

### 8.2 — Minimal BPE merge (conceptual)

```python
from collections import Counter, defaultdict

def get_vocab_from_corpus(corpus):
    """Convert corpus to word → (char-split, count) dict."""
    word_counts = Counter(corpus.split())
    vocab = {}
    for word, count in word_counts.items():
        # Represent each word as tuple of chars + end-of-word marker
        chars = tuple(word) + ("</w>",)
        vocab[chars] = count
    return vocab

def get_pair_stats(vocab):
    """Count frequency of adjacent symbol pairs."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = list(word)
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Merge the most frequent pair in all words."""
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        word_str = " ".join(word)
        new_word_str = word_str.replace(bigram, replacement)
        new_word = tuple(new_word_str.split())
        new_vocab[new_word] = freq
    return new_vocab

# Run BPE for N_MERGES iterations
corpus = "Hello world Hello Transformers world world"
vocab  = get_vocab_from_corpus(corpus)
N_MERGES = 5
merges = []

print("Initial vocab:", vocab)

for i in range(N_MERGES):
    pairs = get_pair_stats(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    merges.append(best_pair)
    vocab = merge_vocab(best_pair, vocab)
    print(f"Merge {i+1}: {best_pair} → {''.join(best_pair)}")
    print(f"  Vocab now: {vocab}\n")
```

### 8.3 — Full tokenizer pipeline from scratch

```python
import torch
import torch.nn as nn

# ── Step 1: Define a simple char-level tokenizer ─────────────────────────────
class SimpleTokenizer:
    def __init__(self, corpus, pad_token="<PAD>", unk_token="<UNK>"):
        chars = sorted(set(corpus))
        self.vocab = {ch: i+2 for i, ch in enumerate(chars)}  # start at 2
        self.vocab[pad_token] = 0
        self.vocab[unk_token] = 1
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_id = 0
        self.unk_id = 1

    def encode(self, text):
        return [self.vocab.get(ch, self.unk_id) for ch in text]

    def decode(self, ids, skip_specials=True):
        chars = []
        for id_ in ids:
            token = self.inv_vocab.get(id_, "<UNK>")
            if skip_specials and id_ in (self.pad_id,):
                continue
            chars.append(token)
        return "".join(chars)


# ── Step 2: Batch encoding with padding + attention mask ─────────────────────
def batch_encode(tokenizer, texts, max_length=None):
    """
    Returns:
      input_ids     : LongTensor [batch, seq_len]
      attention_mask: LongTensor [batch, seq_len]
    """
    encoded = [tokenizer.encode(t) for t in texts]
    
    if max_length is None:
        max_length = max(len(e) for e in encoded)
    
    input_ids      = []
    attention_masks = []
    
    for seq in encoded:
        seq = seq[:max_length]               # truncate
        pad_len = max_length - len(seq)
        
        mask = [1] * len(seq) + [0] * pad_len
        seq  = seq + [tokenizer.pad_id] * pad_len
        
        input_ids.append(seq)
        attention_masks.append(mask)
    
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_masks, dtype=torch.long)
    )


# ── Step 3: Embedding lookup + masking in a mini Transformer input layer ─────
class TransformerInputLayer(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.token_emb    = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_emb = nn.Embedding(max_seq_len, d_model)
        self.norm         = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # [1, T]
        
        tok_emb = self.token_emb(input_ids)        # [B, T, d_model]
        pos_emb = self.position_emb(positions)      # [1, T, d_model]
        
        x = self.norm(tok_emb + pos_emb)            # [B, T, d_model]
        
        # Zero out padding positions in the output (optional, good practice)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
        x = x * mask_expanded
        
        return x                                    # [B, T, d_model]


# ── Demo ─────────────────────────────────────────────────────────────────────
corpus = "Hello, I am learning Transformers! Great."
tokenizer = SimpleTokenizer(corpus)

texts = ["Hello,", "I am learning!"]
input_ids, attention_mask = batch_encode(tokenizer, texts)

print("input_ids:\n", input_ids)
print("\nattention_mask:\n", attention_mask)
# input_ids:
#  tensor([[H, e, l, l, o, ,, 0, 0],
#           [I,  , a, m,  , l, e, a, ...]])
# attention_mask:
#  tensor([[1, 1, 1, 1, 1, 1, 0, 0],
#           [1, 1, 1, 1, 1, 1, 1, 1]])

layer = TransformerInputLayer(
    vocab_size=len(tokenizer.vocab),
    d_model=16,
    max_seq_len=32
)

output = layer(input_ids, attention_mask)
print("\nOutput shape:", output.shape)  # [2, 8, 16]
```

### 8.4 — From-scratch attention with mask (matrix view)

```python
import torch
import torch.nn.functional as F
import math

def attention_with_mask_demo():
    """
    Demonstrate exactly how attention_mask becomes -inf before softmax.
    """
    seq_len = 4
    d_k     = 2

    # Pretend Q, K, V already projected (random)
    torch.manual_seed(42)
    Q = torch.randn(1, seq_len, d_k)
    K = torch.randn(1, seq_len, d_k)
    V = torch.randn(1, seq_len, d_k)

    # Tokens: [Hello, world, <PAD>, <PAD>]
    attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float)
    # shape: [1, 4]

    # Step 1: Raw scores
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    # shape: [1, 4, 4]
    print("Raw scores:\n", scores.squeeze(0))

    # Step 2: Expand mask to [1, 1, 4] and broadcast to [1, 4, 4]
    mask = attention_mask.unsqueeze(1)  # [1, 1, 4]
    scores = scores.masked_fill(mask == 0, float('-inf'))
    print("\nAfter masking (PAD cols = -inf):\n", scores.squeeze(0))

    # Step 3: Softmax — -inf → 0
    weights = F.softmax(scores, dim=-1)
    print("\nAttention weights (PAD cols = 0.0):\n", weights.squeeze(0))
    # Example output:
    # tensor([[0.43, 0.57, 0.00, 0.00],
    #         [0.61, 0.39, 0.00, 0.00],
    #         [0.48, 0.52, 0.00, 0.00],
    #         [0.35, 0.65, 0.00, 0.00]])

    # Step 4: Weighted sum of V
    output = torch.bmm(weights, V)
    print("\nOutput shape:", output.shape)  # [1, 4, 2]

attention_with_mask_demo()
```

---

## 9. HuggingFace Transformers — Full Pipeline

### 9.1 — GPT-2 complete worked example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 1. Load ────────────────────────────────────────────────────────────────
tok   = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tok.pad_token = tok.eos_token  # GPT-2 has no PAD, reuse EOS

# ── 2. Tokenize ────────────────────────────────────────────────────────────
text = "Hello, I am learning Transformers!"
enc  = tok(text, return_tensors="pt")

print("Keys:", enc.keys())
# dict_keys(['input_ids', 'attention_mask'])

print("input_ids shape:", enc["input_ids"].shape)         # [1, 8]
print("attention_mask shape:", enc["attention_mask"].shape)  # [1, 8]

# ── 3. Forward pass ────────────────────────────────────────────────────────
with torch.no_grad():
    out = model(**enc)

print("Logits shape:", out.logits.shape)
# [1, 8, 50257]  =  [batch, seq_len, vocab_size]
# → for each of the 8 token positions, we get a probability distribution
#   over all 50257 tokens (what comes next)

# ── 4. Decode most likely next token ──────────────────────────────────────
next_token_logits = out.logits[0, -1, :]     # last position
next_token_id     = next_token_logits.argmax()
next_token_str    = tok.decode([next_token_id])
print("Predicted next token:", next_token_str)

# ── 5. Full encode → decode round-trip ────────────────────────────────────
ids     = enc["input_ids"][0].tolist()
decoded = tok.decode(ids, skip_special_tokens=True)
print("Round-trip decoded:", decoded)
# "Hello, I am learning Transformers!"
```

### 9.2 — BERT complete worked example (with token_type_ids)

```python
from transformers import AutoTokenizer, AutoModel
import torch

tok   = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

sent_a = "The cat sat on the mat."
sent_b = "The cat is on a mat."

enc = tok(sent_a, sent_b, return_tensors="pt", padding=True)

print("input_ids:\n", enc["input_ids"])
# tensor([[101, 1996, 4937, 2938, 2006, 1996, 13523, 1012,
#          102, 1996, 4937, 2003, 2006, 1037, 13523, 1012, 102]])
#        [CLS  The   cat   sat   on   the   mat    .
#         SEP  The   cat   is    on    a    mat    .   SEP]

print("token_type_ids:\n", enc["token_type_ids"])
# tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
#          ← sent A (0) ────────────→ ← sent B (1) ──────────→

print("attention_mask:\n", enc["attention_mask"])
# tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

with torch.no_grad():
    out = model(**enc)

print("Last hidden state shape:", out.last_hidden_state.shape)
# [1, 17, 768]  =  [batch, seq_len, d_model]

# CLS token representation (used for classification)
cls_repr = out.last_hidden_state[:, 0, :]
print("CLS repr shape:", cls_repr.shape)  # [1, 768]
```

### 9.3 — offset_mapping: tracking token→character positions

```python
# Useful for NER, QA — know which character spans each token covers
tok = AutoTokenizer.from_pretrained("gpt2")
text = "Hello, I am learning!"

enc = tok(text, return_offsets_mapping=True)

for token_id, offset in zip(enc["input_ids"], enc["offset_mapping"]):
    token_str = tok.decode([token_id])
    char_span  = text[offset[0]:offset[1]]
    print(f"ID={token_id:6d}  token='{token_str}'  chars=[{offset[0]}:{offset[1]}] → '{char_span}'")

# Output:
# ID= 15496  token='Hello'  chars=[0:5] → 'Hello'
# ID=    11  token=','      chars=[5:6] → ','
# ID=   314  token=' I'     chars=[6:8] → ' I'
# ...
```

---

## 10. Batching — Padding & Truncation

### The full batching pipeline

```python
from transformers import AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

texts = [
    "Hello world",                              # short
    "I am learning Transformers in PyTorch",    # medium
    "Hi",                                       # very short
]

# ── padding="longest"  → pads to longest in batch ─────────────────────────
enc = tok(texts, padding="longest", return_tensors="pt")
print("input_ids:\n", enc["input_ids"])
print("\nattention_mask:\n", enc["attention_mask"])

# input_ids (padded to length of longest sequence):
# tensor([[ 15496,    995,  50256,  50256,  50256,  50256,  50256,  50256],
#         [   314,    716,   4673,   3602,  19637,  287,  9485, 15637],
#         [ 17250,  50256,  50256,  50256,  50256,  50256,  50256,  50256]])

# attention_mask:
# tensor([[1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0]])


# ── padding="max_length" + truncation → fixed length ──────────────────────
enc2 = tok(texts, padding="max_length", max_length=6, truncation=True, return_tensors="pt")
print("\nWith max_length=6:\n", enc2["input_ids"])

# ── padding_side matters! ─────────────────────────────────────────────────
tok.padding_side = "left"   # pad on the left (important for decoder-only models)
enc3 = tok(texts, padding="longest", return_tensors="pt")
print("\nLeft-padded attention_mask:\n", enc3["attention_mask"])
# tensor([[0, 0, 0, 0, 0, 0, 1, 1],   ← Hello world, padded on left
#         [1, 1, 1, 1, 1, 1, 1, 1],
#         [0, 0, 0, 0, 0, 0, 0, 1]])
```

### Why left-padding for decoder-only models (GPT-2, Llama)?

```
Decoder-only models generate tokens LEFT-TO-RIGHT.
Generation always starts from the LAST real token.

Right-padding:
  [Hello, world, <PAD>, <PAD>]
  Generation continues from <PAD> → WRONG — model is confused

Left-padding:
  [<PAD>, <PAD>, Hello, world]
  Generation continues from 'world' → CORRECT — last token is real
```

> ⚠️ **Watch out:** For encoder models (BERT), padding side doesn't matter because the encoder attends to all tokens simultaneously. For decoder models (GPT-2, Llama), always use `padding_side="left"`.

### Truncation strategies

```python
# truncation=True → truncate to model's max_length (model_max_length)
enc = tok(long_text, truncation=True)

# Truncate to specific length
enc = tok(long_text, max_length=128, truncation=True)

# For sentence pairs (BERT): truncate only the longest
enc = tok(sent_a, sent_b, truncation=True, max_length=128)
# strategy="only_first"  → truncate only sent_a
# strategy="only_second" → truncate only sent_b
# strategy="longest_first" (default) → truncate whichever is longer
```

---

## 11. Quick Reference

### Core tokenizer methods

| Method | Input | Output | Notes |
|---|---|---|---|
| `tokenize(text)` | str | List[str] | Subword pieces only, no IDs |
| `encode(text)` | str | List[int] | IDs only, no tensors |
| `__call__(text)` | str/List | Dict of tensors | Full pipeline, use this |
| `decode(ids)` | List[int] | str | IDs → string |
| `batch_decode(ids)` | List[List[int]] | List[str] | Batch decode |
| `convert_tokens_to_ids(toks)` | List[str] | List[int] | No normalization |
| `convert_ids_to_tokens(ids)` | List[int] | List[str] | No joining |
| `get_vocab()` | — | Dict[str,int] | Full vocab dict |

### tokenizer() call arguments

| Argument | Type | Purpose |
|---|---|---|
| `return_tensors` | `"pt"/"tf"/"np"` | Return format |
| `padding` | `True/"longest"/"max_length"` | Pad strategy |
| `truncation` | `True/False` | Enable truncation |
| `max_length` | int | Max sequence length |
| `add_special_tokens` | bool | Add BOS/EOS/CLS/SEP |
| `return_attention_mask` | bool | Include mask |
| `return_token_type_ids` | bool | Include type IDs |
| `return_offsets_mapping` | bool | Char span offsets |
| `stride` | int | Overlap for long docs |

### Concept summary

| Concept | What it is | Shape |
|---|---|---|
| **Token** | Subword unit of text | — |
| **Token ID** | Integer index into vocabulary | scalar |
| **input_ids** | Sequence of token IDs | `[B, T]` |
| **Embedding** | Float vector for each token ID | `[B, T, d_model]` |
| **attention_mask** | 1=real, 0=padding | `[B, T]` |
| **token_type_ids** | 0=segA, 1=segB | `[B, T]` |
| **position_ids** | 0,1,2,...,T-1 | `[1, T]` |
| **logits** | Raw unnormalized scores over vocab | `[B, T, V]` |

### Tokenizer algorithm comparison

| Algorithm | Model | Space marker | OOV handling | Vocab |
|---|---|---|---|---|
| BPE (byte-level) | GPT-2, RoBERTa | `Ġ` | Bytes → no OOV | 50K |
| SP-BPE | Llama-2, Mistral | `▁` | Bytes → no OOV | 32K |
| SP-Unigram | T5, ALBERT | `▁` | `<unk>` (rare) | 32K |
| WordPiece | BERT, DistilBERT | `##` suffix | `[UNK]` | 30K |

### WordPiece (BERT) vs BPE (GPT-2) space markers

```
BPE (GPT-2):  prefix on words AFTER a space
  "Hello world" → ['Hello', 'Ġworld']
  First word has NO marker; subsequent words get Ġ

WordPiece (BERT): suffix ## on subword continuations
  "Transformers" → ['Transform', '##ers']
  First piece: normal; continuations get ## prefix

SP (T5/Llama): prefix ▁ on ALL word starts
  "Hello world" → ['▁Hello', '▁world']
  Every word-start (including first) gets ▁
```

---

*Reference: HuggingFace Transformers `tokenizers` module — https://huggingface.co/docs/transformers/main_classes/tokenizer*