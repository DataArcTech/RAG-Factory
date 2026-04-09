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


Refinable_KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document and question, identify all entities and their entity types from the text and all relationships among the identified entities that are relevant to the question. 
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

3. Output Formatting:
- Return the result in valid JSON format with two keys: 'entities' (list of entity objects) and 'relationships' (list of relationship objects).
- Exclude any text outside the JSON structure (e.g., no explanations or comments).
- If no entities or relationships are identified, return empty lists: { "entities": [], "relationships": [] }.

-An Output Example-
{
  "entities": [
    {
      "entity_name": "Albert Einstein",
      "entity_type": "Person",
      "entity_description": "Albert Einstein was a theoretical physicist who developed the theory of relativity and made significant contributions to physics."
    },
    {
      "entity_name": "Theory of Relativity",
      "entity_type": "Scientific Theory",
      "entity_description": "A scientific theory developed by Albert Einstein, describing the laws of physics in relation to observers in different frames of reference."
    },
    {
      "entity_name": "Nobel Prize in Physics",
      "entity_type": "Award",
      "entity_description": "A prestigious international award in the field of physics, awarded annually by the Royal Swedish Academy of Sciences."
    }
  ],
  "relationships": [
    {
      "source_entity": "Albert Einstein",
      "target_entity": "Theory of Relativity",
      "relation": "developed",
      "relationship_description": "Albert Einstein is the developer of the theory of relativity."
    },
    {
      "source_entity": "Albert Einstein",
      "target_entity": "Nobel Prize in Physics",
      "relation": "won",
      "relationship_description": "Albert Einstein won the Nobel Prize in Physics in 1921."
    }
  ]
}

-Real Data-
######################
text: {text}\n
question: {question}
######################
output:"""


from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    EntityNode,
    Relation,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
)
from llama_index.core.schema import BaseNode, TransformComponent
class RefinableGraphRAGConstructor(GraphRAGConstructor):
    def __init__(self, llm, extract_prompt, max_paths_per_chunk, parse_fn, num_workers):
        super().__init__(llm=llm, extract_prompt=extract_prompt, max_paths_per_chunk=max_paths_per_chunk, parse_fn=parse_fn, num_workers=num_workers)

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")
        assert hasattr(node, "metadata")

         # 获取文本内容和问题

        text = node.get_content(metadata_mode="llm")
        question = node.metadata.get("question", "")
        # print(f"text: {text}")
        # print(f"question: {question}")
        try:
            # llm_response = self.llm_response_cache.get(text)
            # if not llm_response or question:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
                question=question
            )
                # self.llm_response_cache.set(text, llm_response)

            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )

            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node
    
def subgraph_refine(chunk, question, index, llm, embedding_model):
    pass

# 测试
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG-Factory CLI")
    parser.add_argument("-c", "--config", default="/finance_ML/wuxiaojun/RAG/RAG-Factory/examples/graphrag/config.yaml", help="配置文件路径")
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

    # 初始化数据库
    print("Loading dataset...")
    dataset_name = dataset_config.dataset_name
    cache_folder = os.path.join(".cache", dataset_name)
    # convert to Path object
    cache_folder = Path(cache_folder)
    print(f"Initializing database at {cache_folder}")
    init_db(cache_folder, remove_exists=False)

    # 创建知识提取器
    kg_extractor = RefinableGraphRAGConstructor(
        llm=llm,
        extract_prompt=Refinable_KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=rag_config.max_paths_per_chunk,
        parse_fn=kg_triples_parse_fn,
        num_workers=rag_config.num_workers
    )

    # 构建索引
    graph_store = stores["graph_store"]
    index = None
    documents = [Document(text="""Lothair II (835 \u2013) was the king of Lotharingia from 855 until his death.\nHe was the second son of Emperor Lothair I and Ermengarde of Tours.\nHe was married to Teutberga (died 875), daughter of Boso the Elder.""",extra_info={"question": "who is Lothair II's mother?"})]

     # 文本切分
    splitter = SentenceSplitter(
        chunk_size=dataset_config.chunk_size,
        chunk_overlap=dataset_config.chunk_overlap
    )
    nodes = splitter.get_nodes_from_documents(documents)

    index = CachedPropertyGraphIndex(
        nodes=nodes,
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        show_progress=True,
        embed_model_name=embedding_config.model
    )

    # index = CachedPropertyGraphIndex.from_existing(
    #                 property_graph_store=graph_store,
    #                 embed_kg_nodes=True,
    #                 embed_model_name=embedding_config.model
    #             )
    # 加载社区信息
    # if not index.property_graph_store.community_summary or not index.property_graph_store.community_info or not index.property_graph_store.entity_info:
    #     print(f"loading entity info, community info and summaries from cache")
    #     index.property_graph_store.load_entity_info()
    #     index.property_graph_store.load_community_info()
    #     index.property_graph_store.load_community_summaries()


    # chunk = """Lothair II (835 \u2013) was the king of Lotharingia from 855 until his death.\nHe was the second son of Emperor Lothair I and Ermengarde of Tours.\nHe was married to Teutberga (died 875), daughter of Boso the Elder."""
    # question = "Lothair II的母亲是谁？"
    # subgraph_refine(chunk, question, index, llm, embedding)
    