import re
import json
from typing import Any, List, Tuple


def kg_triples_parse_fn(response_str: str) -> Any:
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, response_str, re.DOTALL)
    entities = []
    relationships = []
    if not match:
        return entities, relationships
    try:
        data = json.loads(match.group(0))
        entities = [
            (entity["entity_name"], entity["entity_type"], entity["entity_description"])
            for entity in data.get("entities", [])
        ]
        relationships = [
            (rel["source_entity"], rel["target_entity"], rel["relation"], rel["relationship_description"])
            for rel in data.get("relationships", [])
        ]
        return entities, relationships
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing JSON: {e}")
        return entities, relationships