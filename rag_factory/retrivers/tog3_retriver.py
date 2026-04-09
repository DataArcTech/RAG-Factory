from llama_index.core.graph_stores import PropertyGraphStore
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.llm_rerank import LLMRerank


from typing import Optional, Any, Union

from llama_index.core.retrievers import (
    CustomPGRetriever,
    VectorContextRetriever,
    TextToCypherRetriever,
)


class MyCustomRetriever(CustomPGRetriever):
    """Custom retriever with cohere reranking."""

    def init(
        self,
        ## vector context retriever params
        vector_retriever: Optional[VectorContextRetriever] = None,
        property_graph_retriver: Optional[PropertyGraphStore] = None,
        embed_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[VectorStore] = None,
        similarity_top_k: int = 4,
        path_depth: int = 1,
        ## text-to-cypher params
        llm: Optional[LLM] = None,
        text_to_cypher_template: Optional[Union[PromptTemplate, str]] = None,
        ## cohere reranker params
        cohere_api_key: Optional[str] = None,
        cohere_top_n: int = 3,
        **kwargs: Any,
    ) -> None:
        """Uses any kwargs passed in from class constructor."""

        # self.vector_retriever = VectorContextRetriever(
        #     self.graph_store,
        #     include_text=self.include_text,
        #     embed_model=embed_model,
        #     vector_store=vector_store,
        #     similarity_top_k=similarity_top_k,
        #     path_depth=path_depth,
        # )
        self.vector_retriever = vector_retriever

        # self.cypher_retriever = TextToCypherRetriever(
        #     self.graph_store,
        #     llm=llm,
        #     text_to_cypher_template=text_to_cypher_template
        #     ## NOTE: you can attach other parameters here if you'd like
        # )
        self.cypher_retriever = property_graph_retriver

        self.reranker = LLMRerank(
            top_n=cohere_top_n
        )

    def custom_retrieve(self, query_str: str) -> str:
        """Define custom retriever with reranking.

        Could return `str`, `TextNode`, `NodeWithScore`, or a list of those.
        """
        nodes_1 = self.vector_retriever.retrieve(query_str)
        nodes_2 = self.cypher_retriever.retrieve(query_str)
        reranked_nodes = self.reranker.postprocess_nodes(
            nodes_1 + nodes_2, query_str=query_str
        )

        ## TMP: please change
        final_text = "\n\n".join(
            [n.get_content(metadata_mode="llm") for n in reranked_nodes]
        )

        return final_text