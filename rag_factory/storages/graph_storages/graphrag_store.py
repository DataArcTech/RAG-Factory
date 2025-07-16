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


class GraphRAGStore(Neo4jPropertyGraphStore):
    community_summary = {}
    entity_info = None
    community_info = {}
    max_cluster_size = 5

    def __init__(self, llm, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.llm_response_cache = LlmResponseCache(llm_name=llm.model)
        self.entity_info_cache = EntityInfoCache()
        self.community_info_cache = CommunityInfoCache()
        self.community_summary_cache = CommunitySummaryCache()

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