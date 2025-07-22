import argparse
import json
import os
import sys
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf
from rag_factory.args import (
    DatasetConfig,
    LLMConfig,
    EmbeddingConfig,
    StorageConfig,
    RAGConfig,
    Query
)

import numpy as np
import xxhash
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from llama_index.core import Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import PropertyGraphIndex
from llama_index.core.llms import ChatMessage
from rag_factory.llms import OpenAICompatible
from rag_factory.embeddings import OpenAICompatibleEmbedding
from rag_factory.caches import init_db

from rag_factory.documents import kg_triples_parse_fn
from rag_factory.prompts import KG_TRIPLET_EXTRACT_TMPL

from rag_factory.graph_constructor import GraphRAGConstructor
from rag_factory.retrivers.graphrag_query_engine import GraphRAGQueryEngine


def read_args(config_path: Union[str, Path]) -> Tuple[DatasetConfig, LLMConfig, EmbeddingConfig, StorageConfig, RAGConfig]:
    r"""Get arguments from the command line or a config file."""
    config_path = Path(config_path)
    if config_path.suffix in (".yaml", ".yml", ".json"):
        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.load(config_path)
        config_dict = OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
        
        return (
            DatasetConfig(**config_dict.get("dataset", {})),
            LLMConfig(**config_dict.get("llm", {})),
            EmbeddingConfig(**config_dict.get("embedding", {})),
            StorageConfig(**config_dict.get("storage", {})),
            RAGConfig(**config_dict.get("rag", {}))
        )

def initialize_components(
    dataset_config: DatasetConfig,
    llm_config: LLMConfig,
    embedding_config: EmbeddingConfig,
    storage_config: StorageConfig,
    rag_config: RAGConfig
):
    # 初始化LLM
    llm = OpenAICompatible(
        api_base=llm_config.base_url,
        api_key=llm_config.api_key,
        model=llm_config.model
    )
    Settings.llm = llm
    
    # 初始化Embedding模型
    embedding = OpenAICompatibleEmbedding(
        api_base=embedding_config.base_url,
        api_key=embedding_config.api_key,
        model_name=embedding_config.model
    )
    Settings.embed_model = embedding
    
    if storage_config.type == "vector_store":
        # 初始化向量存储
        import qdrant_client
        from rag_factory.storages.vector_storages import QdrantVectorStore
        client = qdrant_client.QdrantClient(
            url=storage_config.url,
        )
        store = QdrantVectorStore(client=client, collection_name=dataset_config.dataset_name)
    elif storage_config.type == "graph_store":
        from rag_factory.storages.graph_storages import GraphRAGStore
        # 初始化图存储
        store = GraphRAGStore(
            llm=llm,
            max_cluster_size=rag_config.max_cluster_size,
            url=storage_config.url,
            username=storage_config.username,
            password=storage_config.password,
        )
    
    return llm, embedding, store

def load_dataset(dataset_name: str, subset: int = 0) -> Any:
    """加载数据集"""
    with open(f"./data/{dataset_name}/samples.json", "r") as f:
        dataset = json.load(f)
    return dataset[:subset] if subset else dataset

def get_corpus(dataset: Any, dataset_name: str) -> Dict[int, Tuple[str, str]]:
    """获取语料库"""
    passages = {}
    for datapoint in dataset:
        for title, text in datapoint["context"]:
            text = "\n".join(text)
            hash_t = xxhash.xxh3_64_intdigest(text)
            passages[hash_t] = (title, text)
    return passages

def get_queries(dataset: Any) -> List[Query]:
    """获取查询"""
    return [
        Query(
            question=datapoint["question"],
            answer=datapoint["answer"],
            evidence=list(datapoint["supporting_facts"])
        )
        for datapoint in dataset
    ]

def _query_task(retriever, query_engine, query: Query, solution="naive_rag") -> Dict[str, Any]:
        question = query.question
        retrived_docs = [node.text for node in retriever.retrieve(question)]
        query_engine_response = query_engine.query(question)
        # retrived_docs = [node.text for node in query_engine_response.source_nodes]
        answer = query_engine_response.response


        return {
            "question": query.question,
            "answer": answer,
            "evidence": retrived_docs,
            "ground_truth": [e[0] for e in query.evidence],
            "ground_truth_answer": query.answer,
        }

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="RAG-Factory CLI")
    parser.add_argument("-c", "--config", default="examples/graphrag/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    dataset_config, llm_config, embedding_config, storage_config, rag_config = read_args(args.config)
    print("Loading config file:", args.config)
    # 加载基础组件
    llm, embedding, store = initialize_components(
        dataset_config,
        llm_config,
        embedding_config,
        storage_config,
        rag_config
    )


    print("Loading dataset...")
    dataset_name = dataset_config.dataset_name
    n_samples = dataset_config.n_samples
    dataset = load_dataset(dataset_name, n_samples)
    corpus = get_corpus(dataset, dataset_name)
    documents = [Document(text=f"{title}: {text}") for _, (title, text) in corpus.items()]

    # 初始化数据库
    cache_folder = os.path.join(".cache", dataset_name)
    # convert to Path object
    cache_folder = Path(cache_folder)
    print(f"Initializing database at {cache_folder}")
    init_db(cache_folder, remove_exists=False)

    
    splitter = SentenceSplitter(
        chunk_size=dataset_config.chunk_size,
        chunk_overlap=dataset_config.chunk_overlap
    )
    nodes = splitter.get_nodes_from_documents(documents)

    args.create = "create" in rag_config.stages
    args.inference = "inference" in rag_config.stages
    args.evaluation = "evaluation" in rag_config.stages

    if args.create:
        print("Create Index...")
        if rag_config.solution == "naive_rag":
            from llama_index.core import StorageContext
            from llama_index.core import VectorStoreIndex

            storage_context = StorageContext.from_defaults(vector_store=store)
            # if collection exists, no need to create index again
            if store._collection_exists(collection_name=dataset_name):
                print(f"Collection {dataset_name} already exists, skipping index creation.")
                index = VectorStoreIndex.from_vector_store(
                    store,
                    storage_context=storage_context,
                    embed_model=Settings.embed_model
                )
            else:
                print(f"Creating collection {dataset_name}...")
                # 创建向量索引
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                )
        elif rag_config.solution == "graph_rag":
            # 创建知识提取器
            kg_extractor = GraphRAGConstructor(
                llm=llm,
                extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
                max_paths_per_chunk=rag_config.max_paths_per_chunk,
                parse_fn=kg_triples_parse_fn,
                num_workers=rag_config.num_workers
            )

            # 构建索引
            index = PropertyGraphIndex(
                nodes=nodes,
                kg_extractors=[kg_extractor],
                property_graph_store=store,
                show_progress=True
            )
            
            # 构建社区
            index.property_graph_store.build_communities()
            print("Knowledge graph construction completed.")

    if args.inference:
        print("Running benchmark...")
        if index is None:
            if rag_config.solution == "naive_rag":
                index = VectorStoreIndex.from_vector_store(
                    store,
                    # Embedding model should match the original embedding model
                    # embed_model=Settings.embed_model
                )
            elif rag_config.solution == "graph_rag":
                index = PropertyGraphIndex.from_existing(
                    property_graph_store=store,
                    embed_kg_nodes=True
                )
                # 加载社区信息
                if not index.property_graph_store.community_summary or not index.property_graph_store.community_info or not index.property_graph_store.entity_info:
                    print(f"loading entity info, community info and summaries from cache")
                    index.property_graph_store.load_entity_info()
                    index.property_graph_store.load_community_info()
                    index.property_graph_store.load_community_summaries()
        
        queries = get_queries(dataset)
        results = []

        # retriver
        retriever = index.as_retriever(
            similarity_top_k=rag_config.similarity_top_k,
        )
        # query engine
        if rag_config.solution == "naive_rag":
            query_engine = index.as_query_engine()
        elif rag_config.solution == "graph_rag":
            if rag_config.mode == "local":
                query_engine = index.as_query_engine()
            elif rag_config.mode == "global":
                query_engine = GraphRAGQueryEngine(
                    graph_store=index.property_graph_store,
                    llm=llm,
                    index=index,
                    similarity_top_k = rag_config.similarity_top_k,
                )
        elif rag_config.solution == "multi_modal_rag":
            # TODO: Implement Multi-modal RAG solution
            raise NotImplementedError("Multi-modal RAG solution is not implemented yet.")
        else:
            raise ValueError(f"Unsupported RAG solution: {rag_config.solution}")

        for query in tqdm(queries, desc="Processing queries"):
            response = _query_task(retriever, query_engine, query, solution=rag_config.solution)
            results.append(response)

        # Save results
        os.makedirs(f"./results/{dataset_name}/{rag_config.solution}", exist_ok=True)
        result_file = f"./results/{dataset_name}/{rag_config.solution}/{dataset_name}_{n_samples}.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
    
    if args.evaluation:
        print("Evaluating results...")
        # Compute evaluation metrics
        if not results:
            with open(result_file, "r") as f:
                results = json.load(f)

        answer_scores: List[float] = []
        for result in results:
            ground_truth_answer = result["ground_truth_answer"]
            predicted_answer = result["answer"]

            p_answer = 1 if ground_truth_answer in predicted_answer else 0
            answer_scores.append(p_answer)
        
        print(f"answer EM score:{np.mean(answer_scores)}")