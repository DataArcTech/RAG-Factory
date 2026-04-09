from llama_index.core import PropertyGraphIndex

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Type, TYPE_CHECKING
from llama_index.core.base.base_retriever import BaseRetriever
import tqdm
if TYPE_CHECKING:
    from llama_index.core.indices.property_graph.sub_retrievers.base import (
        BasePGRetriever,
    )
from llama_index.core.ingestion.pipeline import (
    run_transformations,
    arun_transformations,
)
from llama_index.core.graph_stores.types import (
    LabelledNode,
    Relation,
    PropertyGraphStore,
    TRIPLET_SOURCE_KEY,
)

from llama_index.core.schema import BaseNode, MetadataMode, TextNode, TransformComponent

from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    VECTOR_SOURCE_KEY,
)

from rag_factory.caches.cache import EmbeddingCache


class CachedPropertyGraphIndex(PropertyGraphIndex):
    """
    A class for indexing Tog3 data.
    """


    # def __init__(self, data: dict):
    #     """
    #     Initialize the Tog3Index with the given data.

    #     Args:
    #         data (dict): The data to index.
    #     """
    #     super().__init__(data)
    def __init__(self, embed_model_name, **kwargs):
        
        self._embed_model_name = embed_model_name
        self._embed_cache = EmbeddingCache(
            embed_model_name=embed_model_name,
            # embed_model_params=getattr(self._embed_model, 'model_kwargs', {})
        )

        super().__init__(**kwargs)

    # def search(self, query: str):
    #     """

    def _insert_nodes(self, nodes: Sequence[BaseNode]) -> Sequence[BaseNode]:
        """Insert nodes to the index struct."""
        if len(nodes) == 0:
            return nodes

        # run transformations on nodes to extract triplets
        if self._use_async:
            nodes = asyncio.run(
                arun_transformations(
                    nodes, self._kg_extractors, show_progress=self._show_progress
                )
            )
        else:
            nodes = run_transformations(
                nodes, self._kg_extractors, show_progress=self._show_progress
            )

        # ensure all nodes have nodes and/or relations in metadata
        assert all(
            node.metadata.get(KG_NODES_KEY) is not None
            or node.metadata.get(KG_RELATIONS_KEY) is not None
            for node in nodes
        )

        kg_nodes_to_insert: List[LabelledNode] = []
        kg_rels_to_insert: List[Relation] = []
        for node in nodes:
            # remove nodes and relations from metadata
            kg_nodes = node.metadata.pop(KG_NODES_KEY, [])
            kg_rels = node.metadata.pop(KG_RELATIONS_KEY, [])

            # add source id to properties
            for kg_node in kg_nodes:
                kg_node.properties[TRIPLET_SOURCE_KEY] = node.id_
            for kg_rel in kg_rels:
                kg_rel.properties[TRIPLET_SOURCE_KEY] = node.id_

            # add nodes and relations to insert lists
            kg_nodes_to_insert.extend(kg_nodes)
            kg_rels_to_insert.extend(kg_rels)

        # filter out duplicate kg nodes
        kg_node_ids = {node.id for node in kg_nodes_to_insert}
        existing_kg_nodes = self.property_graph_store.get(ids=list(kg_node_ids))
        existing_kg_node_ids = {node.id for node in existing_kg_nodes}
        kg_nodes_to_insert = [
            node for node in kg_nodes_to_insert if node.id not in existing_kg_node_ids
        ]

        # filter out duplicate llama nodes
        existing_nodes = self.property_graph_store.get_llama_nodes(
            [node.id_ for node in nodes]
        )
        existing_node_hashes = {node.hash for node in existing_nodes}
        nodes = [node for node in nodes if node.hash not in existing_node_hashes]

        # embed nodes (if needed)
        if self._embed_kg_nodes:
            # embed llama-index nodes
            node_texts = [
                node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes
            ]

            # 
            embeddings = self._embed_with_cache(node_texts, "_llama_nodes")  

            print(f"Embedding {len(nodes)} llama nodes...")
            for node, embedding in tqdm.tqdm(zip(nodes, embeddings)):
                node.embedding = embedding

            # embed kg nodes
            kg_node_texts = [str(kg_node) for kg_node in kg_nodes_to_insert]
            self._use_async = False
            # if self._use_async:
            #     kg_embeddings = asyncio.run(
            #         self._embed_model.aget_text_embedding_batch(
            #             kg_node_texts, show_progress=self._show_progress
            #         )
            #     )
            # else:
            #     kg_embeddings = self._embed_model.get_text_embedding_batch(
            #         kg_node_texts,
            #         show_progress=self._show_progress,
            #     )
            kg_embeddings = self._embed_with_cache(kg_node_texts, "_kg_nodes")  

            print(f"Embedding {len(kg_nodes_to_insert)} kg nodes...")
            for kg_node, embedding in tqdm.tqdm(zip(kg_nodes_to_insert, kg_embeddings)):
                kg_node.embedding = embedding

        # if graph store doesn't support vectors, or the vector index was provided, use it
        if self.vector_store is not None and len(kg_nodes_to_insert) > 0:
            self._insert_nodes_to_vector_index(kg_nodes_to_insert)

        if len(nodes) > 0:
            self.property_graph_store.upsert_llama_nodes(nodes)

        if len(kg_nodes_to_insert) > 0:
            self.property_graph_store.upsert_nodes(kg_nodes_to_insert)

        # important: upsert relations after nodes
        if len(kg_rels_to_insert) > 0:
            self.property_graph_store.upsert_relations(kg_rels_to_insert)

        # refresh schema if needed
        if self.property_graph_store.supports_structured_queries:
            self.property_graph_store.get_schema(refresh=False)

        return nodes
    
    def _embed_with_cache(self, texts: List[str], cache_key_suffix: str = "") -> List[List[float]]:  
        """使用缓存的 embedding 生成函数"""  
        # 初始化缓存  
        # embed_model_name = f"{self._embed_model.__class__.__name__}{cache_key_suffix}"  
        # cache = EmbeddingCache(  
        #     embed_model_name=embed_model_name,  
        #     embed_model_params=getattr(self._embed_model, 'model_kwargs', {})  
        # )  
        
        # 检查缓存  
        cached_embeddings = {}  
        uncached_texts = []  
        uncached_indices = []  
        
        for i, text in enumerate(texts):  
            # breakpoint()
            cached_embedding = self._embed_cache.get(text)  
            if cached_embedding:  
                cached_embeddings[i] = cached_embedding  
            else:  
                uncached_texts.append(text)  
                uncached_indices.append(i)  
        
        # 生成未缓存的 embeddings  
        if uncached_texts:  
            if self._use_async:  
                new_embeddings = asyncio.run(  
                    self._embed_model.aget_text_embedding_batch(  
                        uncached_texts, show_progress=self._show_progress  
                    )  
                )  
            else:  
                new_embeddings = self._embed_model.get_text_embedding_batch(  
                    uncached_texts, show_progress=self._show_progress  
                )  
            
            # 缓存新生成的 embeddings  
            for text, embedding, idx in zip(uncached_texts, new_embeddings, uncached_indices):  
                self._embed_cache.set(text, embedding)  
                cached_embeddings[idx] = embedding  
        
        # 按原始顺序返回所有 embeddings  
        return [cached_embeddings[i] for i in range(len(texts))]
    
    # def as_retriever(
    #     self,
    #     sub_retrievers: Optional[List["BasePGRetriever"]] = None,
    #     include_text: bool = True,
    #     **kwargs: Any,
    # ) -> BaseRetriever:
    #     """
    #     Return a retriever for the index.

    #     Args:
    #         sub_retrievers (Optional[List[BasePGRetriever]]):
    #             A list of sub-retrievers to use. If not provided, a default list will be used:
    #             `[LLMSynonymRetriever, VectorContextRetriever]` if the graph store supports vector queries.
    #         include_text (bool):
    #             Whether to include source-text in the retriever results.
    #         **kwargs:
    #             Additional kwargs to pass to the retriever.

    #     """
    #     from llama_index.core.indices.property_graph.retriever import (
    #         PGRetriever,
    #     )
    #     from llama_index.core.indices.property_graph.sub_retrievers.vector import (
    #         VectorContextRetriever,
    #     )
    #     from llama_index.core.indices.property_graph.sub_retrievers.llm_synonym import (
    #         LLMSynonymRetriever,
    #     )

    #     if sub_retrievers is None:
    #         sub_retrievers = [
    #             LLMSynonymRetriever(
    #                 graph_store=self.property_graph_store,
    #                 include_text=include_text,
    #                 llm=self._llm,
    #                 **kwargs,
    #             ),
    #         ]

    #         if self._embed_model and (
    #             self.property_graph_store.supports_vector_queries or self.vector_store
    #         ):
    #             sub_retrievers.append(
    #                 VectorContextRetriever(
    #                     graph_store=self.property_graph_store,
    #                     vector_store=self.vector_store,
    #                     include_text=include_text,
    #                     embed_model=self._embed_model,
    #                     **kwargs,
    #                 )
    #             )

    #     return PGRetriever(sub_retrievers, use_async=self._use_async, **kwargs)

    