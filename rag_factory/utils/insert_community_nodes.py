# community summary also put into neo4j graph store

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