import orjson
from typing import Any

def to_json_string(data: Any) -> str:
    """
    Serialize data to a JSON string using orjson for better performance.
    
    Args:
        data: Data to serialize (e.g., dict, list)
    """
    return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode('utf-8')