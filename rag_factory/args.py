from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class DatasetConfig:
    dataset_name: str
    n_samples: int = 0
    chunk_size: int = 1024
    chunk_overlap: int = 20

@dataclass
class LLMConfig:
    type: str = "OpenAILike"
    base_url: str = ""
    api_key: str = ""
    model: str = ""

@dataclass
class EmbeddingConfig:
    type: str = "OpenAILikeEmbedding"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    dimension: int = 0

@dataclass
class StorageConfig:
    type: str = "vector_store"  # "vector_store" or "graph_store"
    url: str = ""
    username: str = ""
    password: str = ""

@dataclass
class RAGConfig:
    solution: str = "naive_rag"  # "naive_rag", "graph_rag" or "multi_modal_rag"
    mode: str = "local"  # "local" or "global"
    num_workers: int = 4
    similarity_top_k: int = 10
    stages: List[str] = field(default_factory=lambda: ["create", "inference", "evaluation"])
    max_paths_per_chunk: int = 10
    max_cluster_size: int = 50

@dataclass
class Query:
    question: str = field()
    answer: str = field()
    evidence: List[Tuple[str, int]] = field()