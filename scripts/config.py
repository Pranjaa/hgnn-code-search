import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = r"checkpoints"
INTERMEDIATE_DIR = r"intermediate"
HYPERGRAPH_NAME = "semantic_hypergraph.json"
HYPERGRAPH_PATH = r"intermediate\semantic_hypergraph.json"
ANNOTATED_FILE = r"data\annotatedData.jsonl"
E5_MODEL = r"models\e5-small-v2"
ST_MODEL = r"models\all-MiniLM-L6-v2"
DATASET = r"code_search_net"

EPOCHS = 20
LR = 1e-4
FEATURE_DIM = 384
HIDDEN_DIM = 128
OUTPUT_DIM = 384
TYPE_EMBED_DIM = 32
NUM_LAYERS = 3
DROPOUT = 0.2
BATCH_SIZE = 256
GRAD_STEPS = 2

TOP_K = 5
TOP_K_CONCEPTS = 4