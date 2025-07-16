import argparse
import json
import os
from pathlib import Path
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import xxhash
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from llama_index.core import Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import PropertyGraphIndex
from rag_factory.llms import OpenAICompatible
from rag_factory.embeddings import OpenAICompatibleEmbedding
from rag_factory.caches import init_db

from rag_factory.documents import kg_triples_parse_fn
from rag_factory.prompts import KG_TRIPLET_EXTRACT_TMPL

from rag_factory.graph_constructor import GraphRAGConstructor
from rag_factory.storages.graph_storages import GraphRAGStore

@dataclass
class Query:
    question: str = field()
    answer: str = field()
    evidence: List[Tuple[str, int]] = field()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def initialize_components(config):
    # 初始化LLM
    llm = OpenAICompatible(
        api_base=config['llm']['base_url'],
        api_key=config['llm']['api_key'],
        model=config['llm']['model']
    )
    Settings.llm = llm
    
    # 初始化Embedding模型
    embedding = OpenAICompatibleEmbedding(
        api_base=config['embedding']['base_url'],
        api_key=config['embedding']['api_key'],
        model_name=config['embedding']['model']
    )
    Settings.embed_model = embedding
    
    # 初始化图存储
    graph_store = GraphRAGStore(
        llm=llm,
        url=config['graph_store']['url'],
        username=config['graph_store']['username'],
        password=config['graph_store']['password']
    )
    
    return llm, embedding, graph_store

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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="RAG-Factory CLI")
    parser.add_argument("-c", "--config", default="examples/graphrag/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    print("加载配置文件:", args.config)
    # 加载基础组件
    llm, embedding, graph_store = initialize_components(config)


    print("加载数据集...")
    dataset_name = config['dataset']['dataset_name']
    n_samples = config['dataset'].get('n_samples', 0)
    dataset = load_dataset(dataset_name, n_samples)
    corpus = get_corpus(dataset, dataset_name)
    documents = [Document(text=f"{title}: {text}") for _, (title, text) in corpus.items()]

    # 初始化数据库
    cache_folder = os.path.join(".cache", config['dataset']['dataset_name'])
    # convert to Path object
    cache_folder = Path(cache_folder)
    print(f"初始化缓存数据库: {cache_folder}")
    init_db(cache_folder, remove_exists=False)

    
    splitter = SentenceSplitter(
        chunk_size=config['rag']['chunk_size'],
        chunk_overlap=config['rag']['chunk_overlap']
    )
    nodes = splitter.get_nodes_from_documents(documents)

    args.create = "create" in config['rag']['stages']
    args.inference = "inference" in config['rag']['stages']
    args.evaluation = "evaluation" in config['rag']['stages']

    if args.create:
        print("创建知识图谱...")

        # 创建知识提取器
        kg_extractor = GraphRAGConstructor(
            llm=llm,
            extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
            max_paths_per_chunk=config['rag']['max_paths_per_chunk'],
            parse_fn=kg_triples_parse_fn,
            num_workers=config['rag']['num_workers']
        )

        # 构建索引
        index = PropertyGraphIndex(
            nodes=nodes,
            kg_extractors=[kg_extractor],
            property_graph_store=graph_store,
            show_progress=True
        )
        
        # 构建社区
        index.property_graph_store.build_communities()
        print("知识图谱创建完成")

    if args.inference:
        print("运行基准测试...")
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            embed_kg_nodes=True
        )
        
        queries = get_queries(dataset)
        results = []
        
        for query in tqdm(queries, desc="处理查询"):
            response = index.as_query_engine().query(query.question)
            results.append({
                "question": query.question,
                "answer": response.response,
                "ground_truth": query.answer
            })
        
        # 保存结果
        os.makedirs(f"./results/{dataset_name}", exist_ok=True)
        result_file = f"./results/{dataset_name}/{dataset_name}_{n_samples}.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
    
    if args.evaluation:
        print("评估结果...")
        # 计算评估指标
        from llama_index.core.evaluation import (
            CorrectnessEvaluator,
            FaithfulnessEvaluator,
            RelevancyEvaluator
        )
        
        evaluators = [
            CorrectnessEvaluator(llm=llm),
            FaithfulnessEvaluator(llm=llm),
            RelevancyEvaluator(llm=llm)
        ]
        
        metrics = {}
        for evaluator in evaluators:
            eval_name = evaluator.__class__.__name__
            metrics[eval_name] = []
            for result in results:
                eval_result = evaluator.evaluate(
                    query=result["question"],
                    response=result["answer"],
                    reference=result["ground_truth"]
                )
                metrics[eval_name].append(eval_result.score)
        
        # 计算平均分
        avg_metrics = {
            name: sum(scores)/len(scores)
            for name, scores in metrics.items()
        }
        
        # 保存评估结果
        with open(f"./results/{args.dataset}_{args.n}_metrics.json", "w") as f:
            json.dump({
                "detailed_metrics": metrics,
                "average_metrics": avg_metrics
            }, f, indent=4)
        
        print(f"基准测试完成，结果已保存到 {result_file}")
        print("评估指标:")
        for name, score in avg_metrics.items():
            print(f"- {name}: {score:.2f}")