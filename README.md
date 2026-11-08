# Tiny GPT with Mini Corpus

A small-scale GPT-style language model built using PyTorch to explore Natural Language Processing (NLP), tokenization techniques, and text generation.

## About the Project

Tiny GPT with Mini Corpus is an experimental Generative AI project that implements a simplified GPT architecture using a small text corpus.

The project explores how different tokenization methods affect language modeling and text generation. It compares three SentencePiece tokenization approaches: Character, Byte Pair Encoding (BPE), and Unigram.

## Technologies & Tools

- Python
- PyTorch
- SentencePiece
- Natural Language Processing (NLP)
- Transformer Architecture
- Generative AI

## Model Architecture

The model uses a simplified Transformer-based architecture consisting of:

- Token Embedding
- Positional Embedding
- Transformer Blocks
- Layer Normalization
- Linear Output Layer
- Autoregressive Text Generation

## Tokenization Methods

This project compares three tokenization methods:

| Method | Description |
|---|---|
| Character | Tokenizes text into individual characters |
| BPE | Builds subword tokens by merging frequent character sequences |
| Unigram | Uses a probabilistic subword tokenization approach |

## Model Configuration

| Parameter | Value |
|---|---:|
| Embedding Dimension | 32 |
| Attention Heads | 2 |
| Transformer Layers | 2 |
| Block Size | 6 |
| Learning Rate | 0.001 |
| Training Epochs | 1,500 |
| Batch Size | 16 |
| Vocabulary Size Setting | 100 |

## Project Structure

```text
TinyGPT-MiniCorpus/
│
├── 23.11.5638 Mini Corpus.pdf
├── README.md
├── corpus.txt
├── tinygpt.py
├── transformer_blocks.py
│
├── tokenizer.model
├── tokenizer.vocab
│
├── tokenizer_char.model
├── tokenizer_char.vocab
│
├── tokenizer_bpe.model
├── tokenizer_bpe.vocab
│
├── tokenizer_unigram.model
└── tokenizer_unigram.vocab
```

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/tanzz15/TinyGPT-MiniCorpus.git
cd TinyGPT-MiniCorpus
```

### 2. Install Dependencies

```bash
pip install torch sentencepiece
```

### 3. Run the Project

```bash
python tinygpt.py
```

Make sure `corpus.txt` and `transformer_blocks.py` are available in the project directory.

## Training and Evaluation

The project records:

- Vocabulary size
- Total token count
- Training loss
- Generated text

The results are used to compare the behavior of the three tokenization methods during language model training.

## Learning Objectives

Through this project, I explored:

- Fundamental concepts of GPT architecture.
- Transformer-based language modeling.
- Text preprocessing and tokenization.
- Character and subword tokenization.
- Deep Learning using PyTorch.
- Basic Generative AI implementation.

## Developer

**GitHub:** [@tanzz15](https://github.com/tanzz15)

This project represents my exploration of Natural Language Processing, Transformer architecture, and Generative AI through a simplified GPT implementation.

---

*Developed as an experimental Generative AI project.*
