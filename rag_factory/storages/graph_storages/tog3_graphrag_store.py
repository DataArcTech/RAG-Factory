import re
import json
import networkx as nx
from graspologic.partition import hierarchical_leiden
from collections import defaultdict
from tqdm import tqdm

from llama_index.core.llms import ChatMessage
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from rag_factory.caches.cache import (
    LlmResponseCache,
    EntityInfoCache,
    CommunityInfoCache,
    CommunitySummaryCache,
)


from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    Triplet,
    LabelledNode,
    Relation,
    EntityNode,
    ChunkNode,
)
from typing import Any, List, Dict, Optional, Tuple, Type
from llama_index.core.bridge.pydantic import BaseModel, Field, SerializeAsAny

CHUNK_SIZE = 1000
BASE_ENTITY_LABEL = "__Entity__"
BASE_NODE_LABEL = "__Node__"
BASE_COMMUNITY_LABEL = "__Community__"


class CommunityNode(LabelledNode):
    """A community in a graph."""

    text: str = Field(description="The text content of the community.")
    id_: Optional[str] = Field(
        default=None, description="The id of the node. Defaults to a hash of the text."
    )
    name: Optional[str] = Field(
        default=None,
        description="The name of the community node"
    )
    label: str = Field(default="community", description="The label of the node.")
    properties: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        """Return the string representation of the node."""
        return self.text

    @property
    def id(self) -> str:
        """Get the node id."""
        return str(hash(self.text)) if self.id_ is None else self.id_


class TOG3GraphRAGStore(Neo4jPropertyGraphStore):
    community_summary = {}
    entity_info = None
    community_info = {}
    max_cluster_size = 5

    def __init__(self, llm, max_cluster_size=5, **kwargs):
        super().__init__(**kwargs)
        self.max_cluster_size = max_cluster_size
        self.llm = llm
        self.llm_response_cache = LlmResponseCache(llm_name=llm.model)
        self.entity_info_cache = EntityInfoCache()
        self.community_info_cache = CommunityInfoCache()
        self.community_summary_cache = CommunitySummaryCache()

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        # Lists to hold separated types
        entity_dicts: List[dict] = []
        chunk_dicts: List[dict] = []
        community_dicts: List[dict] = []

        # Sort by type
        for item in nodes:
            if isinstance(item, EntityNode):
                entity_dicts.append({**item.dict(), "id": item.id})
            elif isinstance(item, ChunkNode):
                chunk_dicts.append({**item.dict(), "id": item.id})
            elif isinstance(item, CommunityNode):
                community_dicts.append({**item.dict(), "id": item.id})
            else:
                # Log that we do not support these types of nodes
                # Or raise an error?
                pass

        if chunk_dicts:
            for index in range(0, len(chunk_dicts), CHUNK_SIZE):
                chunked_params = chunk_dicts[index : index + CHUNK_SIZE]
                self.structured_query(
                    f"""
                    UNWIND $data AS row
                    MERGE (c:{BASE_NODE_LABEL} {{id: row.id}})
                    SET c.text = row.text, c:Chunk
                    WITH c, row
                    SET c += row.properties
                    WITH c, row.embedding AS embedding
                    WHERE embedding IS NOT NULL
                    CALL db.create.setNodeVectorProperty(c, 'embedding', embedding)
                    RETURN count(*)
                    """,
                    param_map={"data": chunked_params},
                )

        if community_dicts:
            for index in range(0, len(community_dicts), CHUNK_SIZE):
                chunked_params = community_dicts[index : index + CHUNK_SIZE]
                self.structured_query(
                    f"""
                    UNWIND $data AS row
                    MERGE (c:{BASE_NODE_LABEL} {{id: row.id}})
                    SET c.name = row.name, c.text = row.text, c:{BASE_COMMUNITY_LABEL}
                    SET c += row.properties
                    WITH c, row.embedding AS embedding
                    WHERE embedding IS NOT NULL
                    CALL db.create.setNodeVectorProperty(c, 'embedding', embedding)
                    RETURN count(*)
                    """,
                    param_map={"data": chunked_params},
                )


        if entity_dicts:
            for index in range(0, len(entity_dicts), CHUNK_SIZE):
                chunked_params = entity_dicts[index : index + CHUNK_SIZE]
                self.structured_query(
                    f"""
                    UNWIND $data AS row
                    MERGE (e:{BASE_NODE_LABEL} {{id: row.id}})
                    SET e += apoc.map.clean(row.properties, [], [])
                    SET e.name = row.name, e:`{BASE_ENTITY_LABEL}`
                    WITH e, row
                    CALL apoc.create.addLabels(e, [row.label])
                    YIELD node
                    WITH e, row
                    CALL (e, row) {{
                        WITH e, row
                        WHERE row.embedding IS NOT NULL
                        CALL db.create.setNodeVectorProperty(e, 'embedding', row.embedding)
                        RETURN count(*) AS count
                    }}
                    WITH e, row WHERE row.properties.triplet_source_id IS NOT NULL
                    MERGE (c:{BASE_NODE_LABEL} {{id: row.properties.triplet_source_id}})
                    MERGE (e)<-[:MENTIONS]-(c)
                    """,
                    param_map={"data": chunked_params},
                )

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        params = [r.dict() for r in relations]
        for index in range(0, len(params), CHUNK_SIZE):
            chunked_params = params[index : index + CHUNK_SIZE]

            self.structured_query(
                f"""
                UNWIND $data AS row
                MERGE (source: {BASE_NODE_LABEL} {{id: row.source_id}})
                ON CREATE SET source:Chunk
                MERGE (target: {BASE_NODE_LABEL} {{id: row.target_id}})
                ON CREATE SET target:Chunk
                WITH source, target, row
                CALL apoc.merge.relationship(source, row.label, {{}}, row.properties, target) YIELD rel
                RETURN count(*)
                """,
                param_map={"data": chunked_params},
            )



    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM."""
        cached_summary = self.llm_response_cache.get(text)
        if cached_summary:
            return cached_summary

        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        try:
            clean_response = self.llm_response_cache.get(text)
            if not clean_response:
                response = self.llm.chat(messages)
                clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
                self.llm_response_cache.set(text, clean_response)
        except Exception as e:
            print(f"Error generating community summary: {e}")

        return clean_response

    def build_communities(self):
        """Builds communities from the graph and summarizes them."""

        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        self.entity_info, self.community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )

        # Store entity info in cache
        if self.entity_info:
            for entity, info in self.entity_info.items():
                self.entity_info_cache.set(entity, json.dumps(info))
        
        # Store community info in cache
        if self.community_info:
            for community_id, details in self.community_info.items():
                self.community_info_cache.set(community_id, json.dumps(details))

        self._summarize_communities(self.community_info)
        # Store summaries in cache
        if self.community_summary:
            for community_id, summary in self.community_summary.items():
                self.community_summary_cache.set(community_id, summary)


    def _create_nx_graph(self):
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        for entity1, relation, entity2 in triplets:
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        """
        Collect information for each node based on their community,
        allowing entities to belong to multiple clusters.
        """
        entity_info = defaultdict(set)
        community_info = defaultdict(list)
        for item in clusters:
            node = item.node
            cluster_id = item.cluster
            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    community_info[cluster_id].append(detail)

        # Convert sets to lists for easier serialization if needed
        entity_info = {k: list(v) for k, v in entity_info.items()}

        return dict(entity_info), dict(community_info)

    def _summarize_communities(self, community_info):
        """Generate and store summaries for each community."""
        print(f"Generating summaries for {len(community_info)} communities...")
        for community_id, details in tqdm(community_info.items()):
            details_text = "\n".join(details) + "."
            self.community_summary[
                community_id
            ] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary:
            self.build_communities()
        return self.community_summary
    

    def load_entity_info(self):
        """Load entity information from the cache."""
        if not self.entity_info:
            self.entity_info = self.entity_info_cache.get_all()
        return self.entity_info

    def load_community_info(self):
        """Load community information from the cache."""
        if not self.community_info:
            self.community_info = self.community_info_cache.get_all()
        return self.community_info

    def load_community_summaries(self):
        """Load community summaries from the cache."""
        if not self.community_summary:
            self.community_summary = self.community_summary_cache.get_all()
        return self.community_summary