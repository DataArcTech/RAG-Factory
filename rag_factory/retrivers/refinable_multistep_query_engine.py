from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    get_response_synthesizer,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag_factory.tools.subgraph_refinement_v2 import subgraph_refine
from rag_factory.tools.subgraph_refinement_v2 import RefinableGraphRAGConstructor, Refinable_KG_TRIPLET_EXTRACT_TMPL
from rag_factory.indexer.graph_indexer import CachedPropertyGraphIndex

from llama_index.core import Settings, Document 
from rag_factory.documents import kg_triples_parse_fn
from llama_index.core.node_parser import SentenceSplitter





def default_stop_fn(stop_dict: Dict) -> bool:
    """Stop function for multi-step query combiner."""
    query_bundle = cast(QueryBundle, stop_dict.get("query_bundle"))
    if query_bundle is None:
        raise ValueError("Response must be provided to stop function.")

    return "none" in query_bundle.query_str.lower()


class RefinableMultiStepQueryEngine(BaseQueryEngine):
    """
    Multi-step query engine.

    This query engine can operate over an existing base query engine,
    along with the multi-step query transform.

    Args:
        query_engine (BaseQueryEngine): A BaseQueryEngine object.
        query_transform (StepDecomposeQueryTransform): A StepDecomposeQueryTransform
            object.
        response_synthesizer (Optional[BaseSynthesizer]): A BaseSynthesizer
            object.
        num_steps (Optional[int]): Number of steps to run the multi-step query.
        early_stopping (bool): Whether to stop early if the stop function returns True.
        index_summary (str): A string summary of the index.
        stop_fn (Optional[Callable[[Dict], bool]]): A stop function that takes in a
            dictionary of information and returns a boolean.

    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        query_transform: StepDecomposeQueryTransform,
        index: CachedPropertyGraphIndex,
        splitter: SentenceSplitter = SentenceSplitter(),
        response_synthesizer: Optional[BaseSynthesizer] = None,
        num_steps: Optional[int] = 3,
        early_stopping: bool = True,
        index_summary: str = "None",
        stop_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> None:
        self._query_engine = query_engine
        self._query_transform = query_transform
        self._response_synthesizer = response_synthesizer or get_response_synthesizer(
            callback_manager=self._query_engine.callback_manager
        )
        self.index=index
        self.splitter=splitter

        self._index_summary = index_summary
        self._num_steps = num_steps
        self._early_stopping = early_stopping
        # TODO: make interface to stop function better
        self._stop_fn = stop_fn or default_stop_fn
        # num_steps must be provided if early_stopping is False
        if not self._early_stopping and self._num_steps is None:
            raise ValueError("Must specify num_steps if early_stopping is False.")
        
        # 
        # 创建知识提取器
        self.refinable_kg_extractor = RefinableGraphRAGConstructor(
            llm=Settings.llm,
            extract_prompt=Refinable_KG_TRIPLET_EXTRACT_TMPL,
            max_paths_per_chunk=2,
            parse_fn=kg_triples_parse_fn,
            num_workers=4
        )

        callback_manager = self._query_engine.callback_manager
        super().__init__(callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {
            "response_synthesizer": self._response_synthesizer,
            "query_transform": self._query_transform,
        }

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes, source_nodes, metadata = self._query_multistep(query_bundle)

            final_response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
                additional_source_nodes=source_nodes,
            )
            final_response.metadata = metadata

            query_event.on_end(payload={EventPayload.RESPONSE: final_response})

        return final_response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes, source_nodes, metadata = self._query_multistep(query_bundle)

            final_response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
                additional_source_nodes=source_nodes,
            )
            final_response.metadata = metadata

            query_event.on_end(payload={EventPayload.RESPONSE: final_response})

        return final_response

    def _combine_queries(
        self, query_bundle: QueryBundle, prev_reasoning: str
    ) -> QueryBundle:
        """Combine queries."""
        transform_metadata = {
            "prev_reasoning": prev_reasoning,
            "index_summary": self._index_summary,
        }
        return self._query_transform(query_bundle, metadata=transform_metadata)

    def _query_multistep(
        self, query_bundle: QueryBundle
    ) -> Tuple[List[NodeWithScore], List[NodeWithScore], Dict[str, Any]]:
        """Run query combiner."""
        prev_reasoning = ""
        cur_response = None
        should_stop = False
        cur_steps = 0

        # use response
        final_response_metadata: Dict[str, Any] = {"sub_qa": []}

        text_chunks = []
        source_nodes = []
        while not should_stop:
            if self._num_steps is not None and cur_steps >= self._num_steps:
                should_stop = True
                break
            elif should_stop:
                break
            refinable_chunks = []
            updated_query_bundle = self._combine_queries(query_bundle, prev_reasoning)

            cur_response = self._query_engine.query(updated_query_bundle)

            # append to response builder
            cur_qa_text = (
                f"\nQuestion: {updated_query_bundle.query_str}\n"
                f"Answer: {cur_response!s}"
            )
            text_chunks.append(cur_qa_text)
            for source_node in cur_response.source_nodes:
                source_nodes.append(source_node)

                # new_content = (
                #        preamble_text + graph_content_str + "\n\n" + cur_content
                #    ) in llama_index/core.indices/property_graph/sub_retrievers/base.py line 114
                if "SUMMARY_FOR" in source_node.text:
                    # If the source node is a summary, use it as is
                    # tmp_text_ls = source_node.text.split(".")[1:]
                    # refinable_chunks.append(".".join(tmp_text_ls))
                    continue
                else:
                    # Otherwise, split and take the second part
                    refinable_chunks.append(source_node.text.split("\n\n")[2])

            # refine subgraph
            refinable_documents = [Document(text="\n".join(refinable_chunks),extra_info={"question": query_bundle.query_str})]
            refinable_nodes = self.splitter.get_nodes_from_documents(refinable_documents)
            self.refinable_index = CachedPropertyGraphIndex(
                nodes=refinable_nodes,
                kg_extractors=[self.refinable_kg_extractor],
                property_graph_store=self.index.property_graph_store,
                show_progress=True,
                embed_model_name=Settings.embed_model.model_name
            )
            # update metadata
            final_response_metadata["sub_qa"].append(
                (updated_query_bundle.query_str, cur_response)
            )

            prev_reasoning += (
                f"- {updated_query_bundle.query_str}\n- {cur_response!s}\n"
            )
            cur_steps += 1

            # TODO: make stop logic better
            stop_dict = {"query_bundle": updated_query_bundle}
            if self._stop_fn(stop_dict):
                should_stop = True
                break

        nodes = [
            NodeWithScore(node=TextNode(text=text_chunk)) for text_chunk in text_chunks
        ]
        return nodes, source_nodes, final_response_metadata
