import json
import os
import re
import torch
import nltk
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  
from nltk.stem import WordNetLemmatizer
from sentence_transformers import util
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize

from config import DEVICE, E5_MODEL, INTERMEDIATE_DIR, DATASET, HYPERGRAPH_NAME

class E5Encoder:
    def __init__(self, model_name=E5_MODEL, device=DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state[:, 0]
                embeddings.append(emb.cpu())

        embeddings = torch.cat(embeddings, dim=0).numpy()
        return embeddings if len(texts) > 1 else embeddings[0]
    
class SemanticHypergraphBuilder:
    def __init__(self, language="python", split="train", max_samples=10000):
        self._ensure_nltk()
        dataset = load_dataset(DATASET, language, split=split)
        self.dataset = dataset.select(range(min(len(dataset), max_samples)))

        self.encoder = E5Encoder()
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=5, max_features=10000)
        self.lemmatizer = WordNetLemmatizer()

        self.nodes = {} # Node text -> Node ID
        self.node_types = {} # Function / Concept
        self.node_embeddings = [] # E5 embeddings
        self.function_embeddings = []  # List of embeddings for function nodes
        self.concept_embeddings = []   # List of embeddings for concept nodes
        self.edges = [] # Nodes, type
        self.function_ids = [] # Function IDs
        self.concept_ids = [] # Concept IDs
        self.func_docstrings = []
        self.code_mapping = {} # Function ID -> Function code
        self.docstring_mapping = {} # Function ID -> Docstring
        self.func_metadata = {}  # Function ID -> Function name, Intent
        self.concept_to_func = defaultdict(set) # Concept -> Function IDs
        self.function_to_concepts = defaultdict(set)  # Function ID -> Concept

    def _ensure_nltk(self):
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

    def _get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN

    def _normalize(self, word):
        word = word.lower()
        word = re.sub(r'[^a-z]', '', word)

        if len(word) <= 2 or word.isnumeric():
            return ""

        tokens = word_tokenize(word)
        if not tokens:
            return ""

        pos = pos_tag(tokens)[0][1]
        wordnet_pos = self._get_wordnet_pos(pos)
        lemmatized = self.lemmatizer.lemmatize(tokens[0], pos=wordnet_pos)
        return lemmatized

    def _add_node(self, text, typ, embedding=None):
        if text in self.nodes:
            return self.nodes[text]

        node_id = len(self.nodes)
        self.nodes[text] = node_id
        self.node_types[node_id] = typ

        if embedding is None:
            embedding = self.encoder.encode(text)
        self.node_embeddings.append(embedding)

        if typ == "function":
            self.function_embeddings.append(embedding)
            self.function_ids.append(node_id)
        elif typ == "concept":
            self.concept_embeddings.append(embedding)
            self.concept_ids.append(node_id)

        return node_id

    def _extract_intents(self, docstring):
        if not docstring.strip():
            return []

        sentences = nltk.sent_tokenize(docstring)
        tagged = [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in sentences]
        grammar = "VP: {<VB.*><.*>*}"  
        chunk_parser = nltk.RegexpParser(grammar)

        intent_candidates = []
        for i, tagged_sent in enumerate(tagged):
            tree = chunk_parser.parse(tagged_sent)
            for subtree in tree.subtrees(filter=lambda t: t.label() == 'VP'):
                phrase = " ".join(word for word, tag in subtree.leaves())
                if len(phrase.split()) >= 3:
                    intent_candidates.append((i, phrase))

        if not intent_candidates:
            return sentences[:1]

        phrases = [x[1] for x in intent_candidates]
        phrase_embeddings = self.encoder.encode(phrases)
        query_embeddings = self.encoder.encode([ 
            "intent: what this function aims to do", 
            "intent: purpose of this method", 
            "intent: what happens when this runs"
        ])
        scores = util.cos_sim(query_embeddings, phrase_embeddings).mean(dim=0)
        top_indices = scores.topk(k=min(3, len(phrases))).indices.tolist()
        return [phrases[i] for i in top_indices]

    def extract_keywords(self, top_k=10):
        print("Extracting TF-IDF keywords...")
        docstrings = [self.docstring_mapping[func_id] for func_id in self.function_ids]
        tfidf_matrix = self.vectorizer.fit_transform(docstrings)
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        keywords_per_doc = defaultdict(list)
        for idx, row in enumerate(tfidf_matrix):
            sorted_indices = row.toarray().flatten().argsort()[::-1][:top_k]
            keywords = [self._normalize(feature_names[i]) for i in sorted_indices]
            keywords = list(filter(None, keywords))
            keywords_per_doc[self.function_ids[idx]].extend(keywords)

        del tfidf_matrix  
        return keywords_per_doc

    def build_function_nodes(self):
        texts, entries, docstrings = [], [], []

        for entry in tqdm(self.dataset, desc="Building function nodes"):
            doc = entry.get('func_documentation_string', '').strip()
            if not doc or len(doc.split()) < 3:
                continue
            doc = re.sub(r':param.*|:return.*', '', doc).strip()
            doc = doc.split('\n')[0]
            intents = self._extract_intents(doc)
            if not intents:
                continue
            intent_text = "; ".join(intents)
            texts.append(intent_text)
            entries.append(entry)
            docstrings.append(doc)

        embeddings = self.encoder.encode(texts)

        for intent, emb, entry, doc in zip(texts, embeddings, entries, docstrings):
            func_id = self._add_node(intent, "function", emb)
            self.function_ids.append(func_id)
            self.code_mapping[func_id] = entry.get("func_code_string", "")
            self.docstring_mapping[func_id] = doc
            self.func_metadata[func_id] = {
                "name": entry.get("func_name", ""),
                "intent": intent
            }

    def build_concept_edges(self, keywords_per_doc):
        concept_to_funcs = defaultdict(set)
        for func_id, keywords in keywords_per_doc.items():
            for kw in keywords:
                norm_kw = self._normalize(kw)
                if norm_kw:
                    concept_to_funcs[norm_kw].add(func_id)
                    self.function_to_concepts[func_id].add(norm_kw)

        concepts = list(concept_to_funcs.keys())
        embeddings = self.encoder.encode(concepts, batch_size=64)

        for concept, emb in zip(concepts, embeddings):
            concept_id = self._add_node(concept, "concept", emb)
            self.edges.append({
                "nodes": [concept_id] + list(concept_to_funcs[concept]),
                "type": "semantic-keyword"
            })

        del embeddings

    def _build_sparse_incidence_matrix(self):
        from scipy.sparse import coo_matrix
        rows, cols = [], []
        for edge_id, edge in enumerate(self.edges):
            for node_id in edge["nodes"]:
                rows.append(node_id)
                cols.append(edge_id)
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)
        data = np.ones(len(rows), dtype=np.int8)
        coo = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_edges))
        return {
            "row": coo.row.tolist(),
            "col": coo.col.tolist(),
            "data": coo.data.tolist(),
            "shape": coo.shape
        }

    def save_node_texts(self):
        os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
        function_texts, concept_texts = [], []

        for text, node_id in tqdm(self.nodes.items(), desc="Saving node texts"):
            node_type = self.node_types[node_id]
            if node_type == "function":
                function_texts.append(text)
            elif node_type == "concept":
                concept_texts.append(text)

        with open(os.path.join(INTERMEDIATE_DIR, "function_nodes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(function_texts))

        with open(os.path.join(INTERMEDIATE_DIR, "concept_nodes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(concept_texts))

    def save(self):
        os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
        sparse_matrix = self._build_sparse_incidence_matrix()

        with open(os.path.join(INTERMEDIATE_DIR, HYPERGRAPH_NAME), 'w') as f:
            json.dump({
                "nodes": self.nodes,
                "node_ids": list(self.nodes.keys()),
                "node_types": [self.node_types[i] for i in range(len(self.nodes))],
                "edges": self.edges,
                "embeddings": [x.tolist() for x in self.node_embeddings],
                "function_embeddings": [x.tolist() for x in self.function_embeddings],
                "concept_embeddings": [x.tolist() for x in self.concept_embeddings],
                "function_ids": self.function_ids,
                "concept_ids": self.concept_ids,
                "code_map": self.code_mapping,
                "docstring_map": self.docstring_mapping,
                "metadata": self.func_metadata,
                "incidence_matrix": sparse_matrix,
                "concept_to_func": {
                    concept: list(function_ids)
                    for concept, function_ids in self.concept_to_func.items()
                },
                "function_to_concepts": {
                    str(func_id): list(concepts)
                    for func_id, concepts in self.function_to_concepts.items()
                }
            }, f, indent=2)

    def build(self):
        print("Constructing hypergraph...")
        self.build_function_nodes()
        keywords = self.extract_keywords()
        self.build_concept_edges(keywords)
        self.save()
        self.save_node_texts()
        print("Hypergraph constructed.")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    builder = SemanticHypergraphBuilder(max_samples=30000)
    builder.build()
