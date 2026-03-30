# Search Engine from Scratch

## Features

- SPIMI-based indexing pipeline (default), with optional BSBI compatibility mode.
- Multiple postings compression options:
  - Standard (raw integer array)
  - Variable-Byte Encoding (VBE)
  - Elias-Gamma Encoding
- Ranked retrieval methods:
  - TF-IDF (Term-at-a-Time)
  - BM25 with two modes:
    - TAAT (Term-at-a-Time)
    - WAND (faster top-k pruning)
- Evaluation module over 30 benchmark queries using:
  - RBP
  - DCG
  - NDCG
  - Average Precision (AP)

## Project Structure

- `bsbi.py`: Build inverted index with `spimi` (default) or `bsbi` mode, then merge into one main index.
- `search.py`: Run ranked retrieval for sample queries.
- `evaluation.py`: Evaluate retrieval quality using `queries.txt` and `qrels.txt`.
- `index.py`: Disk-based inverted index reader/writer.
- `compression.py`: Postings compression codecs.
- `util.py`: Utilities (ID mapping, postings merge helper).
- `collection/`: Document collection (grouped by blocks/folders).
- `index/`: Output index files (`*.index`, `*.dict`).

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run all commands from the project root folder.

### 1. Build the Index

```bash
python bsbi.py --compression elias-gamma --indexing-mode spimi
```

Available indexing modes:

- `spimi` (default)
- `bsbi`

SPIMI memory control (optional):

```bash
python bsbi.py --compression elias-gamma --indexing-mode spimi --spimi-max-terms 50000
```

Available compression values:

- `standard`
- `vbe`
- `elias-gamma`

Example with VBE:

```bash
python bsbi.py --compression vbe --indexing-mode spimi
```

### 2. Run Search

```bash
python search.py --compression elias-gamma --scoring bm25 --bm25 wand -k 10
```

Parameters:

- `--compression`: postings encoding used by the index
- `--scoring`: `tfidf` or `bm25`
- `--bm25`: `wand` or `taat` (used when `--scoring bm25`)
- `-k`: number of top documents to return

Important:

- The `--compression` value in search must match the one used when indexing.
- `search.py` currently runs three predefined sample queries. To test custom queries, edit the `queries` list in `search.py`.

### 3. Evaluate Retrieval Quality

```bash
python evaluation.py --compression elias-gamma --scoring all --bm25 wand -k 1000
```

Parameters:

- `--compression`: postings encoding used by the index
- `--scoring`: `all`, `tfidf`, or `bm25`
- `--bm25`: `wand` or `taat`
- `-k`: top-k depth per query
- `--k1`: BM25 saturation parameter (default: 1.2)
- `--b`: BM25 length-normalization parameter (default: 0.75)

Example BM25-only evaluation:

```bash
python evaluation.py --compression elias-gamma --scoring bm25 --bm25 taat -k 1000 --k1 1.2 --b 0.75
```
