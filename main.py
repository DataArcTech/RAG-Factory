import argparse
import json
import os
import sys
import asyncio
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf
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
        # llm = OpenAICompatible(
        #     api_base=llm_config.base_url,
        #     api_key=llm_config.api_key,
        #     model=llm_config.model,
        #     # max_tokens=512
        # )
        model = "gpt-3.5-turbo"
        # model = "gpt-oss-20b"
        # model = "gpt-5-nano"
        # model = "gpt-4o-mini"
        
        llm = OpenAI(temperature=0, model=model, api_base=os.getenv("OPENAI_API_URL"), api_key=os.getenv("OPENAI_API_KEY"))
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

def get_images(dataset: Any, dataset_name: str) -> Dict[int, Tuple[str, str]]:
    """获取图片库"""
    all_images = []

    images_matadata_path = Path(f"./data/{dataset_name}/images_metadata.json")
    images_path = Path(f"./data/{dataset_name}/images")
    # load metadata from json file

    with open(images_matadata_path, "r") as f:
        images_matadata = json.load(f)

    for image in images_matadata:
        image_path = images_path / Path(image["file_name"]+".png")
        if image_path.exists():
            # img_content = Image.open(BytesIO(image_path.read_bytes()))
            # all_images.append(img_content)
            all_images.append({"text": image["caption"], "path": image_path})
    return all_images


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

async def get_handler(agent, question: str, ctx: Context) -> Any:
    """获取处理器"""
    # handler = agent.run(question)
    # for ev in handler.stream_events():
    #     # if isinstance(ev, ToolCallResult): # response.tool_calls
    #     #     print(f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}")
    #     if isinstance(ev, AgentStream):
    #         print(f"{ev.delta}", end="", flush=True)
    question = question.query_str if isinstance(question, QueryBundle) else question
    response = await agent.run(question)
    for tool_call in response.tool_calls:
        # tool_calls.tool_output.raw_output.source_nodes
        print(f"\nCall {tool_call.tool_name} with {tool_call.tool_kwargs}\nReturned: {tool_call.tool_output}")
    return response.response


def _query_task(retriever, query_engine, query: Query, solution="naive_rag", use_ReAct=False) -> Dict[str, Any]:
        
        use_short_answer = False
        if dataset_config.dataset_name in ["hotpotqa", "2wikimultihopqa", "musique"] and use_short_answer:
            question = QueryBundle(
                query_str=query.question
                + " Give a short answer (as few words as possible). Don't explain or note anything, just answer, the answer is less than 10 words.",
                custom_embedding_strs=[query.question],
            )
        else:
            question = query.question
        # retrived_docs = [node.text for node in retriever.retrieve(question)]

        if use_ReAct:
            # tool1 = FunctionTool.from_defaults(fn=execute_sql)
            query_engine_tools = [
                QueryEngineTool.from_defaults(
                    query_engine=query_engine,
                    name="rag_tool",
                    description=(
                        "Provides information about retrieval augmented generation task. "
                        "Use a detailed plain text question as input to the tool."

                    ),
                )
            ]
            output_parser = ReActOutputParser()
            agent = ReActAgent(
                tools=query_engine_tools,
                # llm=OpenAI(model="gpt-4o-mini"),
                # system_prompt="if can not answer the question, use the tool to retrieve information or subgraph refinement.",
            )
            # context to hold this session/state
            ctx = Context(agent)
            query_engine_response = asyncio.run(get_handler(agent, question, ctx))
            # query_engine_response = query_engine.chat(question)
            answer = query_engine_response.content


        else:
            query_engine_response = query_engine.query(question)
            if rag_config.use_multi_step:
                sub_qa = query_engine_response.metadata["sub_qa"]
            answer = query_engine_response.response

        if not use_ReAct:
            retrived_docs = [node.text for node in query_engine_response.source_nodes]
        else:
            retrived_docs = [node.text for node in retriever.retrieve(question)]
        # print(f"length of retrived docs: {len(retrived_docs)}")
        ## Print the response retrived source nodes and their scores
        # print(query_engine_response.source_nodes[0].text, query_engine_response.source_nodes[0].score)
        # display_query_and_multimodal_response(question, query_engine_response)
        if solution == "mm_rag":
            image_nodes = query_engine_response.metadata["image_nodes"] or []
            # text_nodes = query_engine_response.metadata["text_nodes"] or []
            # retrived_docs.extend([node.text for node in text_nodes])
            retrived_images = [scored_img_node.node.image_path for scored_img_node in image_nodes]
            retrived_docs.extend(retrived_images)

       


        return {
            "question": query.question,
            "answer": answer.lower(),
            "evidence": retrived_docs,
            "ground_truth": [e[0] for e in query.evidence],
            "ground_truth_answer": query.answer.lower(),
        }


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

    print(f"length of dataset: {len(dataset)}")
    print(f"length of documents: {len(documents)}")
    print(f"length of nodes: {len(nodes)}")

    if rag_config.solution == "mm_rag":
        # 获取图片数据
        all_images = get_images(dataset, dataset_name)

        # 将图片转换为ImageDocument对象
        image_documents = [ImageDocument(text=img["text"], image_path=img["path"]) for img in all_images]
        # 添加图片节点到nodes
        nodes.extend(image_documents)

    args.create = "create" in rag_config.stages
    args.inference = "inference" in rag_config.stages
    args.evaluation = "evaluation" in rag_config.stages

    index = None
    results = None
    if args.create:
        print("Create Index...")
        if rag_config.solution == "naive_rag":

            text_store = stores["text_store"]
            storage_context = StorageContext.from_defaults(vector_store=text_store)
            # if collection exists, no need to create index again
            if text_store._collection_exists(collection_name=dataset_name):
                print(f"Collection {dataset_name} already exists, skipping index creation.")
                index = VectorStoreIndex.from_vector_store(
                    text_store,
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
            graph_store = stores["graph_store"]
            index = CachedPropertyGraphIndex(
                nodes=nodes,
                kg_extractors=[kg_extractor],
                property_graph_store=graph_store,
                show_progress=True,
                embed_model_name=embedding_config.model,
            )
            
            # 构建社区
            index.property_graph_store.build_communities()
            print("Knowledge graph construction completed.")
        elif rag_config.solution == "mm_rag":
            text_store, image_store = stores["text_store"], stores["image_store"]
            storage_context = StorageContext.from_defaults(
                vector_store=text_store, image_store=image_store
            )

            # if collection exists, no need to create index again
            if text_store.retrieve_existing_index() and image_store.retrieve_existing_index():
                print(f"Collection {dataset_name} already exists, skipping index creation.")
                index = MultiModalVectorStoreIndex.from_vector_store(
                    vector_store=text_store,
                    image_vector_store=image_store
                )
            else:
                print(f"Creating collection {dataset_name}...")

                # Create the MultiModal index
                index = MultiModalVectorStoreIndex.from_documents(
                    nodes,
                    image_embed_model="clip:ViT-B/32",
                    storage_context=storage_context,
                    show_progress=True,
                    # is_image_to_text=True # when ImageNodess that have populated text fields, we can choose to use this text to build embeddings on that will be used for retrieval 
                )

    if args.inference:
        print("Running benchmark...")
        if index is None:
            if rag_config.solution == "naive_rag":
                index = VectorStoreIndex.from_vector_store(
                    text_store,
                    # Embedding model should match the original embedding model
                    # embed_model=Settings.embed_model
                )
            elif rag_config.solution == "graph_rag":
                index = CachedPropertyGraphIndex.from_existing(
                    property_graph_store=graph_store,
                    embed_kg_nodes=True
                )
                # 加载社区信息
                if not index.property_graph_store.community_summary or not index.property_graph_store.community_info or not index.property_graph_store.entity_info:
                    print(f"loading entity info, community info and summaries from cache")
                    index.property_graph_store.load_entity_info()
                    index.property_graph_store.load_community_info()
                    index.property_graph_store.load_community_summaries()

            elif rag_config.solution == "mm_rag":
                index = MultiModalVectorStoreIndex.from_vector_store(
                    vector_store=text_store,
                    image_vector_store=image_store
                )
            
            else:   
                raise ValueError(f"Unsupported RAG solution: {rag_config.solution}")
        
        queries = get_queries(dataset)
        results = []

        # retriver
        if rag_config.solution == "mm_rag":
            retriever = index.as_retriever(
                similarity_top_k=rag_config.similarity_top_k,
                image_similarity_top_k=rag_config.similarity_top_k,
            )
        else:
            # text retriever
            retriever = index.as_retriever(
                similarity_top_k=rag_config.similarity_top_k,
            )
        # query engine
        if rag_config.solution == "naive_rag":
            query_engine = index.as_query_engine(
                similarity_top_k=rag_config.similarity_top_k,
            )
        elif rag_config.solution == "graph_rag":
            if rag_config.mode == "local":
                if rag_config.use_rerank:
                    query_engine = index.as_query_engine(
                        similarity_top_k=rag_config.similarity_top_k,
                        node_postprocessors=[reranker]
                    )
                else:
                    query_engine = index.as_query_engine(
                        similarity_top_k=rag_config.similarity_top_k,
                    )
            if rag_config.use_multi_step:
                from llama_index.core.indices.query.query_transform.base import (
                    StepDecomposeQueryTransform,
                )
                from llama_index.core.query_engine import MultiStepQueryEngine
                index_summary = f"Used to answer questions about the {dataset_name} dataset."
                step_decompose_transform = StepDecomposeQueryTransform(llm=Settings.llm, verbose=True)
                query_engine = MultiStepQueryEngine(
                    query_engine=query_engine,
                    query_transform=step_decompose_transform,
                    index_summary=index_summary,
                )
                
                # if rag_config.use_ReAct:
                #     query_engine = index.as_chat_engine(chat_mode="react", verbose=True)
            elif rag_config.mode == "global":
                query_engine = GraphRAGQueryEngine(
                    graph_store=index.property_graph_store,
                    llm=llm,
                    index=index,
                    similarity_top_k = rag_config.similarity_top_k,
                )
        elif rag_config.solution == "mm_rag":
            query_engine = index.as_query_engine(
                text_qa_template=MULTIMODAL_QA_TMPL,
                # similarity_top_k=rag_config.similarity_top_k, # TODO: check limit_mm_per_prompt='{"image":3}' in VLM inference service
                # image_similarity_top_k=rag_config.similarity_top_k,
                # limit_mm_per_prompt=rag_config.similarity_top_k,
            )
        
        else:
            raise ValueError(f"Unsupported RAG solution: {rag_config.solution}")

        for query in tqdm(queries, desc="Processing queries"):
            response = _query_task(retriever, query_engine, query, solution=rag_config.solution, use_ReAct=rag_config.use_ReAct)
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
            result_file = f"./results/{dataset_name}/{rag_config.solution}/{dataset_name}_{n_samples}.json"
            with open(result_file, "r") as f:
                results = json.load(f)

        em_scores: List[float] = []
        f1_scores: List[Tuple[float, float, float]] = []
        for result in results:
            ground_truth_answer = result["ground_truth_answer"]
            predicted_answer = result["answer"]

            # p_answer = 1 if ground_truth_answer in predicted_answer else 0
            # answer_scores.append(p_answer)

            from rag_factory.evaluations.exact_match import exact_match, f1_score
            em_score = exact_match(ground_truth_answer, predicted_answer)
            f1_score = f1_score(ground_truth_answer, predicted_answer)
            em_scores.append(em_score)
            f1_scores.append(f1_score)

        # Save evaluation results
        save_metrics = {
            "exact_match": np.mean(em_scores),
            "f1_score": np.mean([f1_score[0] for f1_score in f1_scores]),
            "precision": np.mean([f1_score[1] for f1_score in f1_scores]),
            "recall": np.mean([f1_score[2] for f1_score in f1_scores]),
        }

        print("Evaluation metrics:")
        print(f"Exact Match: {save_metrics['exact_match']:.4f}")
        print(f"F1 Score: {save_metrics['f1_score']:.4f}")
        print(f"Precision: {save_metrics['precision']:.4f}")            
        print(f"Recall: {save_metrics['recall']:.4f}")

        save_metrics_file = f"./results/{dataset_name}/{rag_config.solution}/{dataset_name}_{n_samples}_metrics.json"
        os.makedirs(os.path.dirname(save_metrics_file), exist_ok=True)

        with open(save_metrics_file, "w") as f:
            json.dump(save_metrics, f, indent=4)
        print(f"Evaluation metrics saved to {save_metrics_file}")
        # print(f"answer EM score:{np.mean(answer_scores)}")