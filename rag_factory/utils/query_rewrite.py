import json
import re

# query改写
def query_rewrite(query: str, query_rewrite_rules: dict) -> str:
    """
    Rewrite the query based on the provided rules.

    Args:
        query (str): The original query string.
        query_rewrite_rules (dict): A dictionary containing rewrite rules.

    Returns:
        str: The rewritten query.
    """
    for pattern, replacement in query_rewrite_rules.items():
        if re.search(pattern, query):
            return re.sub(pattern, replacement, query)
    return query

# query分解
def query_decompose(query: str, decomposition_rules: dict) -> list:
    """
    Decompose the query into sub-queries based on the provided rules.

    Args:
        query (str): The original query string.
        decomposition_rules (dict): A dictionary containing decomposition rules.

    Returns:
        list: A list of sub-queries.
    """
    sub_queries = []
    for pattern, replacement in decomposition_rules.items():
        if re.search(pattern, query):
            sub_queries.append(re.sub(pattern, replacement, query))
    return sub_queries if sub_queries else [query]