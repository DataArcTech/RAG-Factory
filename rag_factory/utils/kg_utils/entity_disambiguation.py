


# 实体消歧
def named_entity_disambiguation(query: str, kg: dict) -> str:
    """
    实体消歧函数
    :param query: 用户查询
    :param kg: 知识图谱，格式为 {实体名称: 实体ID}
    :return: 消歧后的实体ID
    """
    # 简单的字符串匹配
    for entity, entity_id in kg.items():
        if entity.lower() in query.lower():
            return entity_id
    return None  # 如果没有找到匹配的实体ID