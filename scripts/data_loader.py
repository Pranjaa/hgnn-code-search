import json
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
import nltk

from config import (HYPERGRAPH_PATH, DEVICE)

class SemanticHypergraphDataset(Dataset):
    def __init__(self):
        print("Loading hypergraph from:", HYPERGRAPH_PATH)
        with open(HYPERGRAPH_PATH, 'r') as f:
            self.data = json.load(f)

        self.node_names = self.data['nodes']
        self.node_id_map = self.data['node_ids']
        self.node_types = self.data['node_types']
        self.all_embeddings = np.array(self.data["embeddings"])
        self.function_embeddings = np.array(self.data["function_embeddings"])
        self.concept_embeddings = np.array(self.data["concept_embeddings"])
        self.code_map = self.data['code_map']
        self.docstring_map = self.data['docstring_map']
        self.metadata = self.data['metadata']
        self.edges = self.data['edges']
        self.incidence_info = self.data['incidence_matrix']
        self.node_features = torch.tensor(self.all_embeddings, dtype=torch.float32)
        self.function_indices = self.data['function_ids']
        self.concept_indices = self.data['concept_ids']
        self.function_to_concepts = self.data['function_to_concepts']
        
        self.node_type_tensor = torch.tensor([
            0 if typ == "function" else 1 for typ in self.node_types
        ], dtype=torch.long)

        self.incidence_matrix = self._build_sparse_tensor()

    def _build_sparse_tensor(self):
        row = torch.tensor(self.incidence_info["row"], dtype=torch.long)
        col = torch.tensor(self.incidence_info["col"], dtype=torch.long)
        data = torch.tensor(self.incidence_info["data"], dtype=torch.float32)
        shape = tuple(self.incidence_info["shape"])
        coo = coo_matrix((data.numpy(), (row.numpy(), col.numpy())), shape=shape)
        print("Built sparse incidence matrix of shape", shape)
        return coo

    def __getitem__(self, idx):
        return {
            "node_id": idx,
            "node_name": self.node_names[idx],
            "embedding": self.all_embeddings[idx],
            "type": self.node_type_tensor[idx],
        }

def get_stopwords():
    nltk.download('stopwords', quiet=True)
    return set(nltk.corpus.stopwords.words("english"))

if __name__ == "__main__":
    print("Loading SemanticHypergraphDataset...")

    stopwords = get_stopwords()
    dataset = SemanticHypergraphDataset()
    
    torch.save(dataset.incidence_matrix, r"intermediate\H.pt")
    torch.save(dataset.node_features.to(DEVICE, dtype=torch.float32),  r"intermediate\X.pt")
    torch.save(dataset.node_types, r"intermediate\node_types.pt")
    torch.save(dataset.function_to_concepts, r"intermediate\function_to_concepts.pt")
    torch.save(dataset.concept_embeddings,  r"intermediate\concept_embs.pt")
    torch.save(dataset.all_embeddings,  r"intermediate\node_embs.pt")
    torch.save(dataset.concept_indices,  r"intermediate\concept_indices.pt")
    torch.save(dataset.function_indices,  r"intermediate\function_indices.pt")
    torch.save(dataset.node_names,  r"intermediate\nodes.pt")
    torch.save(dataset.node_type_tensor,  r"intermediate\node_type_map.pt")
    torch.save(dataset.code_map,  r"intermediate\code_map.pt")
    torch.save(stopwords,  r"intermediate\stopwords.pt")
    torch.save(dataset.node_names,  r"intermediate\nodes.pt")


    
    