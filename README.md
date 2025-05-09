# Semantic Code Search using Hypergraph Neural Networks

This project implements a novel approach to **semantic code search** using **Hypergraph Neural Networks (HGNNs)**, specifically the **AllSet Transformer architecture**. It models complex multi-way relationships between code functions and conceptual keywords as a **hypergraph**, enhancing contextual understanding during code retrieval.

---

## About the Project

Traditional graph-based methods fall short when modeling higher-order relationships. In this project:

- A **hypergraph** is constructed where **function nodes** and **conceptual keywords** are connected through **hyperedges**.
- We use **E5-small encodings** to initialize node embeddings.
- A **hyperpath-based contrastive loss** ensures semantic alignment between concept paths and function nodes.
- At inference, a **natural language query** activates relevant concept nodes, forming hyperpaths that guide function retrieval.

---

## Project Structure


```bash
scripts/
│
├── builder.py # Builds the hypergraph from CodeSearchNet (Python split)
├── config.py # Configuration and argument settings
├── data_loader.py # Loads and processes dataset
├── evaluation.py # Evaluation metric and logic
├── inference.py # Inference script for semantic search
├── model.py # Main AllSetTransformer model, loss function
├── modules.py # Utility modules
├── train.py # Training script using hyperpath contrastive loss
├── ui.py # Optional UI-based semantic search interface
```
---

## Installation

1. **Clone the repository**:
```
git clone https://github.com/Pranjaa/hgnn-code-search.git
cd hgnn-code-search
```

Set up the environment:

```
pip install -r requirements.txt
```

Make sure you have Python ≥ 3.8, PyTorch, and other listed dependencies.

## How to Run
### Step-by-Step Execution

Build the Hypergraph:

```
python scripts/builder.py
```

Load and Prepare Data:

```
python scripts/data_loader.py
```

Train the Model:

```
python scripts/train.py
```

Run Inference:

```
python scripts/inference.py 
```

UI interface (optional):

```
python scripts/ui.py
```

## Acknowledgements

- [AllSet](https://arxiv.org/abs/2106.13264): Chien et al., *"AllSet: Hypergraph Transformer for Set Representation Learning"*, ICML 2022.
- [Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](https://arxiv.org/abs/1904.06627): Wang et al., CVPR 2019.
- [CodeSearchNet Dataset](https://github.com/github/CodeSearchNet): Released by GitHub as part of the CodeSearchNet Challenge.
