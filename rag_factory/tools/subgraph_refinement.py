# 获取实体，找到对应的chunk，然后给问题和chunk来回答问题，chunk太长就再抽一遍子图（问题+chunk，带着问题抽子图）
import argparse
import json
import os
import sys
import asyncio
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf
import sys
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from rag_factory.args import (
    DatasetConfig,
    LLMConfig,
    EmbeddingConfig,
    RerankerConfig,
    StorageConfig,
    RAGConfig,
    Query
)

import numpy as np
import xxhash
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image
from io import BytesIO

from llama_index.core import Settings, Document 
from llama_index.core.schema import ImageDocument
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, PropertyGraphIndex
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

# For ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.agent import ReActChatFormatter
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import ToolCallResult, AgentStream

from rag_factory.llms import OpenAICompatible
from llama_index.llms.openai import OpenAI
from llama_index.llms.openrouter import OpenRouter
from rag_factory.embeddings import OpenAICompatibleEmbedding
from rag_factory.caches import init_db
from rag_factory.indexer.graph_indexer import CachedPropertyGraphIndex

from rag_factory.documents import kg_triples_parse_fn
from rag_factory.prompts import KG_TRIPLET_EXTRACT_TMPL, MULTIMODAL_QA_TMPL

from rag_factory.graph_constructor import GraphRAGConstructor
from rag_factory.retrivers.graphrag_query_engine import GraphRAGQueryEngine

from rag_factory.rerankers import XinferenceRerank

def read_args(config_path: Union[str, Path]) -> Tuple[DatasetConfig, LLMConfig, EmbeddingConfig, RerankerConfig, StorageConfig, RAGConfig]:
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
            RerankerConfig(**config_dict.get("reranker", {})),
            StorageConfig(**config_dict.get("storage", {})),
            RAGConfig(**config_dict.get("rag", {}))
        )

def initialize_components(
    dataset_config: DatasetConfig,
    llm_config: LLMConfig,
    embedding_config: EmbeddingConfig,
    reranker_config: RerankerConfig,
    storage_config: StorageConfig,
    rag_config: RAGConfig
):  
    r"""Initialize the components required for RAG."""

    # 初始化LLM
    if rag_config.solution == "mm_rag":
        from rag_factory.multi_modal_llms import OpenAICompatibleMultiModal
        llm = OpenAICompatibleMultiModal(
                api_base=llm_config.base_url,
                api_key=llm_config.api_key,
                model=llm_config.model,
            )
    else:
        llm = OpenAICompatible(
            api_base=llm_config.base_url,
            api_key=llm_config.api_key,
            model=llm_config.model,
            context_window=llm_config.context_window,
            # max_tokens=512
        )
        # model = "gpt-3.5-turbo"
        # model = "gpt-oss-20b"
        # model = "gpt-5-nano"
        # model = "gpt-4o-mini"
        
        # llm = OpenAI(temperature=0, model=model, api_base=os.getenv("OPENAI_API_URL"), api_key=os.getenv("OPENAI_API_KEY"))
        # model = "qwen/qwen3-30b-a3b-instruct-2507"
        # model = "google/gemini-2.5-pro-exp-03-25"
        # model = "google/gemini-2.0-flash-exp:free"
        # llm = OpenRouter(temperature=0, model=model, api_key=os.getenv("OPENROUTER_API_KEY"))


    Settings.llm = llm
    
    # 初始化Embedding模型
    embedding = OpenAICompatibleEmbedding(
        api_base=embedding_config.base_url,
        api_key=embedding_config.api_key,
        model_name=embedding_config.model
    )
    Settings.embed_model = embedding

    reranker = XinferenceRerank(
        base_url=reranker_config.base_url,
        model=reranker_config.model,
        top_n=reranker_config.top_n,
    )


    text_store, graph_store, image_store = None, None, None
    
    if storage_config.type == "vector_store":
        # 初始化向量存储
        import qdrant_client
        from rag_factory.storages.vector_storages import QdrantVectorStore
        client = qdrant_client.QdrantClient(
            url=storage_config.url,
        )
        text_store = QdrantVectorStore(client=client, collection_name=dataset_config.dataset_name)
    elif storage_config.type == "graph_store":
        from rag_factory.storages.graph_storages import GraphRAGStore
        # 初始化图存储
        graph_store = GraphRAGStore(
            llm=llm,
            max_cluster_size=rag_config.max_cluster_size,
            url=storage_config.url,
            username=storage_config.username,
            password=storage_config.password,
            refresh_schema=storage_config.refresh_schema
        )
    elif storage_config.type == "mm_store":
        # import qdrant_client
        # from rag_factory.storages.vector_storages import QdrantVectorStore
        # client = qdrant_client.QdrantClient(
        #     url=storage_config.url,
        # )
        # text_store = QdrantVectorStore(client=client, collection_name=dataset_config.dataset_name+"_text_collection")
        # image_store = QdrantVectorStore(client=client, collection_name=dataset_config.dataset_name+"_image_collection")
        from rag_factory.storages.multimodal_storages import Neo4jVectorStore
        text_store = Neo4jVectorStore(
            url=storage_config.url,
            username=storage_config.username,
            password=storage_config.password,
            index_name=f"{dataset_config.dataset_name}_text_collection",
            node_label="Chunk",
            embedding_dimension=embedding_config.dimension
        )
        image_store = Neo4jVectorStore(
            url=storage_config.url,
            username=storage_config.username,
            password=storage_config.password,
            index_name=f"{dataset_config.dataset_name}_image_collection",
            node_label="Image",
            embedding_dimension=512

        )

    else:
        raise ValueError(f"Unsupported storage type: {storage_config.type}")
    
    stores = {
        "text_store": text_store,
        "graph_store": graph_store,
        "image_store": image_store,
    }

    return llm, embedding, stores, reranker


# community summary also put into neo4j graph store

from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    Triplet,
    LabelledNode,
    Relation,
    EntityNode,
    ChunkNode,
)

from rag_factory.storages.graph_storages.graphrag_store import CommunityNode
import json
from tqdm import tqdm
def insert_community_nodes(index, embedding_model):
    """
    community_summary: {社区ID: 摘要文本}
    community_info: {社区ID: [实体关系三元组列表]}
    entity_info: {实体名: [关联社区ID列表]}
    """
    # 创建CommunityNode列表
            
    community_nodes = []
    for comm_id, summary in index.property_graph_store.community_summary.items():
        info = index.property_graph_store.community_info.get(comm_id, [])
        community_nodes.append(CommunityNode(
            id_=f"summary_{comm_id}",
            text=summary,
            name=f"CommunitySummary_{comm_id}",
            properties={
                "community_id": comm_id,
                "info": json.dumps(info)
            }
        ))
    

    # 建立community的embedding
    node_texts = [node.text for node in community_nodes]
    embeddings = embedding_model.get_text_embedding_batch(
            node_texts, show_progress=True
        )

    for node, embedding in zip(community_nodes, embeddings):
        node.embedding = embedding

    # 存入图数据库
    index.property_graph_store.upsert_nodes(community_nodes)

    # 建立与实体的关系
    # 首先构建社区到实体的反向映射
    community_to_entities = {}
    for entity_name, community_ids in index.property_graph_store.entity_info.items():
        for comm_id in community_ids:
            if comm_id not in community_to_entities:
                community_to_entities[comm_id] = []
            community_to_entities[comm_id].append(entity_name)

    # 建立关系
    # 为每个社区建立与其实体的关系
    for comm_id, entities in tqdm(community_to_entities.items()):
        # 确保社区节点存在
        if comm_id not in index.property_graph_store.community_summary:
            continue
        # 建立关系
        index.property_graph_store.structured_query(
            """
            MATCH (s:__Community__ {community_id: $comm_id})
            MATCH (e:__Entity__)
            WHERE e.name IN $entities
            MERGE (s)-[:SUMMARY_FOR]->(e)
            RETURN count(*)
            """,
            param_map={
                "comm_id": comm_id,
                "entities": entities
            }
        )

def insert_kg_triplets(triplet,index, embedding_model):
    entity_nodes = []
    head, relation, tail = triplet['head'], triplet['relation'], triplet['tail']
    head_node = EntityNode(
        # id_=f"entity_{head}",
        name=head,
        # text=head,
        properties={"name": head}
    )
    tail_node = EntityNode(
        # id_=f"entity_{tail}",
        name=tail,
        # text=tail,
        properties={"name": tail}
    )
    entity_nodes.append(head_node)
    entity_nodes.append(tail_node)

    # 建立embedding
    node_texts = [node.name for node in entity_nodes]
    embeddings = embedding_model.get_text_embedding_batch(
        node_texts, show_progress=True
    )

    for node, embedding in zip(entity_nodes, embeddings):
        node.embedding = embedding

    index.property_graph_store.upsert_nodes(entity_nodes)

    # 建立关系
    index.property_graph_store.upsert_relations([
        Relation(
            source_id=head_node.id,
            target_id=tail_node.id,
            label=relation,
            properties={}
        )
    ])

    return entity_nodes, relation



def subgraph_refine(chunk, question, index, llm, embedding_model):
    # 连接到Neo4j数据库
    # prompt设计: 根据问题和chunk来抽子图triplet
    # prompt = f"""根据问题和chunk来抽子图triplet，问题:{question}，chunk:{chunk}.输出json格式的triplet，"""
    prompt = f"Please extract all the knowledge triplets from the following context that are relevant to the question. question: {question}, context: {chunk}.\n"
    # prompt += """格式如下：[{"head": "head", "relation": "relation", "tail": "tail"}]"""
    prompt += """Format: [{"head": "head", "relation": "relation", "tail": "tail"}], and make sure the output is valid JSON and language is English."""
    # 提取实体
    response = llm.complete(
        prompt
    )
    triplets = response.text
    print("提取的triplets:", triplets)

    # # 解析triplets并存入Neo4j
    try:    
        triplet_list = json.loads(triplets.replace("'", "\""))
        for triplet in triplet_list:
            head = triplet['head']
            relation = triplet['relation']
            tail = triplet['tail']
            print(f"Storing triplet - Head: {head}, Relation: {relation}, Tail: {tail}")
            insert_kg_triplets(triplet, index, embedding_model)
    #         # 创建节点和关系
    #         graph.run("""
    #             MERGE (a:Entity {name: $head})
    #             MERGE (b:Entity {name: $tail})
    #             MERGE (a)-[r:RELATION {type: $relation}]->(b)
    #             """, head=head, tail=tail, relation=relation)
    #     print("子图已存入Neo4j")
    except json.JSONDecodeError:
        print("无法解析triplets:", triplets)
    # except Exception as e:
    #     print("存入Neo4j时出错:", e)
    # return triplets

# 测试
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG-Factory CLI")
    parser.add_argument("-c", "--config", default="examples/graphrag/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # 从.env文件中加载环境变量
    load_dotenv()

    # 加载配置
    dataset_config, llm_config, embedding_config, reranker_config, storage_config, rag_config = read_args(args.config)
    print("Loading config file:", args.config)

    # 加载基础组件
    llm, embedding, stores, reranker = initialize_components(
        dataset_config,
        llm_config,
        embedding_config,
        reranker_config,
        storage_config,
        rag_config
    )

    text_store, graph_store, image_store = stores["text_store"], stores["graph_store"], stores["image_store"]

    # 创建知识提取器
    kg_extractor = GraphRAGConstructor(
        llm=llm,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=rag_config.max_paths_per_chunk,
        parse_fn=kg_triples_parse_fn,
        num_workers=rag_config.num_workers
    )

    # 构建索引
    graph_store = stores["graph_store"]
    index = None
    # index = CachedPropertyGraphIndex(
    #     nodes=nodes,
    #     kg_extractors=[kg_extractor],
    #     property_graph_store=graph_store,
    #     show_progress=True,
    #     embed_model_name=embedding_config.model
    # )

    index = CachedPropertyGraphIndex.from_existing(
                    property_graph_store=graph_store,
                    embed_kg_nodes=True,
                    embed_model_name=embedding_config.model
                )
    # 加载社区信息
    # if not index.property_graph_store.community_summary or not index.property_graph_store.community_info or not index.property_graph_store.entity_info:
    #     print(f"loading entity info, community info and summaries from cache")
    #     index.property_graph_store.load_entity_info()
    #     index.property_graph_store.load_community_info()
    #     index.property_graph_store.load_community_summaries()


    chunk = """Lothair II (835 \u2013) was the king of Lotharingia from 855 until his death.\nHe was the second son of Emperor Lothair I and Ermengarde of Tours.\nHe was married to Teutberga (died 875), daughter of Boso the Elder."""
    question = "Lothair II的母亲是谁？"
    subgraph_refine(chunk, question, index, llm, embedding)
    