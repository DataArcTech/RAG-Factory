from abc import abstractmethod
from typing import Any, Dict, List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.graph_stores.types import PropertyGraphStore, Triplet
from llama_index.core.indices.property_graph.base import (
    TRIPLET_SOURCE_KEY,
)
from llama_index.core.schema import (
    BaseNode,
    NodeWithScore,
    NodeRelationship,
    RelatedNodeInfo,
    QueryBundle,
    TextNode,
)
import re


DEFAULT_PREAMBLE = "Here are some facts extracted from the provided text:\n\n"

# 编写正则表达式
# summary_  : 匹配固定的前缀
# (\d+)     : \d+匹配一个或多个数字，括号()将其创建为一个捕获组，以便我们提取出来
community_pattern = r'summary_(\d+)'

class BasePGRetriever(BaseRetriever):
    """
    The base class for property graph retrievers.

    By default, will retrieve nodes from the graph store and add source text to the nodes if needed.

    Args:
        graph_store (PropertyGraphStore):
            The graph store to retrieve data from.
        include_text (bool, optional):
            Whether to include source text in the retrieved nodes. Defaults to True.
        include_text_preamble (Optional[str], optional):
            The preamble to include before the source text. Defaults to DEFAULT_PREAMBLE.

    """

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        include_text: bool = True,
        include_text_preamble: Optional[str] = DEFAULT_PREAMBLE,
        include_properties: bool = False,
        **kwargs: Any,
    ) -> None:
        self._graph_store = graph_store
        self.include_text = include_text
        self._include_text_preamble = include_text_preamble
        self.include_properties = include_properties
        super().__init__(callback_manager=kwargs.get("callback_manager"))

    def _get_nodes_with_score(
        self, triplets: List[Triplet], scores: Optional[List[float]] = None
    ) -> List[NodeWithScore]:
        results = []
        for i, triplet in enumerate(triplets):
            source_id = triplet[0].properties.get(TRIPLET_SOURCE_KEY, None)
            relationships = {}
            if source_id is not None:
                relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=source_id
                )

            if self.include_properties:
                text = f"{triplet[0]!s} -> {triplet[1]!s} -> {triplet[2]!s}"
            else:
                text = f"{triplet[0].id} -> {triplet[1].id} -> {triplet[2].id}"
            results.append(
                NodeWithScore(
                    node=TextNode(
                        text=text,
                        relationships=relationships,
                    ),
                    score=1.0 if scores is None else scores[i],
                )
            )

        return results

    def _add_source_text(
        self, retrieved_nodes: List[NodeWithScore], og_node_map: Dict[str, BaseNode]
    ) -> List[NodeWithScore]:
        """Combine retrieved nodes/triplets with their source text, using provided preamble."""
        # map of ref doc id to triplets/retrieved labelled nodes
        graph_node_map: Dict[str, List[str]] = {}
        for node in retrieved_nodes:
            ref_doc_id = node.node.ref_doc_id or ""
            if ref_doc_id not in graph_node_map:
                graph_node_map[ref_doc_id] = []

            graph_node_map[ref_doc_id].append(node.node.get_content())

        result_nodes: List[NodeWithScore] = []
        for node_with_score in retrieved_nodes:
            match = re.search(community_pattern, node_with_score.node.get_content())
            mapped_node = og_node_map.get(node_with_score.node.ref_doc_id or "", None)

            if mapped_node:
                graph_content = graph_node_map.get(mapped_node.node_id, [])
                if len(graph_content) > 0:
                    graph_content_str = "\n".join(graph_content)
                    cur_content = mapped_node.get_content()
                    preamble_text = (
                        self._include_text_preamble
                        if self._include_text_preamble
                        else ""
                    )
                    new_content = (
                        preamble_text + graph_content_str + "\n\n" + cur_content
                    )
                    mapped_node = TextNode(**mapped_node.dict())
                    mapped_node.text = new_content
                result_nodes.append(
                    NodeWithScore(
                        node=mapped_node,
                        score=node_with_score.score,
                    )
                )
            # TODO: hard code method in source code
            elif match:
                community_content = community_summary = node_with_score.node.get_content()
                extracted_id = int(match.group(1))
                if extracted_id in self._graph_store.community_summary:
                    community_summary = self._graph_store.community_summary.get(extracted_id)
                elif str(extracted_id) in self._graph_store.community_summary:
                    community_summary = self._graph_store.community_summary.get(str(extracted_id))
                else:
                    community_summary = ""
                node_with_score.node.text = f"{community_content}. Community_{str(extracted_id)} Summary: {community_summary}"
                result_nodes.append(node_with_score)
            else:
                result_nodes.append(node_with_score)

        return result_nodes

    def add_source_text(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Combine retrieved nodes/triplets with their source text."""
        og_nodes = self._graph_store.get_llama_nodes(
            [x.node.ref_doc_id for x in nodes if x.node.ref_doc_id is not None]
        )
        node_map = {node.node_id: node for node in og_nodes}

        return self._add_source_text(nodes, node_map)

    async def async_add_source_text(
        self, nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Combine retrieved nodes/triplets with their source text."""
        og_nodes = await self._graph_store.aget_llama_nodes(
            [x.node.ref_doc_id for x in nodes if x.node.ref_doc_id is not None]
        )
        og_node_map = {node.node_id: node for node in og_nodes}

        return self._add_source_text(nodes, og_node_map)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self.retrieve_from_graph(query_bundle)
        if self.include_text and nodes:
            nodes = self.add_source_text(nodes)
        return nodes

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self.aretrieve_from_graph(query_bundle)
        if self.include_text and nodes:
            nodes = await self.async_add_source_text(nodes)
        return nodes

    @abstractmethod
    def retrieve_from_graph(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes from the labelled property graph."""
        ...

    @abstractmethod
    async def aretrieve_from_graph(
        self, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """Retrieve nodes from the labelled property graph."""
        ...
