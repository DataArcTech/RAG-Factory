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
    context_window: int = 32000

@dataclass
class EmbeddingConfig:
    type: str = "OpenAILikeEmbedding"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    dimension: int = 0

@dataclass
class RerankerConfig:
    type: str = "XinferenceRerank"
    model: str = "bge-reranker-base"
    base_url: str = "http://localhost:9997/v1"
    top_n: int = 5

@dataclass
class StorageConfig:
    type: str = "vector_store"  # "vector_store" or "graph_store"
    url: str = ""
    username: str = ""
    password: str = ""
    refresh_schema: bool = False  # Only for graph store

@dataclass
class RAGConfig:
    solution: str = "naive_rag"  # "naive_rag", "graph_rag" or "multi_modal_rag"
    mode: str = "local"  # "local" or "global"
    use_ReAct: bool = False
    use_rerank: bool = False
    use_multi_step: bool = False
    insert_community_nodes: bool = False
    num_workers: int = 16
    similarity_top_k: int = 10
    stages: List[str] = field(default_factory=lambda: ["create", "inference", "evaluation"])
    max_paths_per_chunk: int = 10
    max_cluster_size: int = 50

@dataclass
class Query:
    question: str = field()
    answer: str = field()
    evidence: List[Tuple[str, int]] = field()