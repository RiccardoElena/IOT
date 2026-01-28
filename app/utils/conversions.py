import orjson
from typing import Any
from io import BytesIO
from PIL import Image

def to_json_string(data: Any) -> str:
    """
    Serialize data to a JSON string using orjson for better performance.
    
    Args:
        data: Data to serialize (e.g., dict, list)
    """
    return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode('utf-8')

def fig_to_png(fig:Any) -> Image.Image:
    img_bytes = fig.to_image(
        format="png",
        width = 1200,
        height= 800,
        scale= 2.0
    )
    img = Image.open(BytesIO(img_bytes))
    return img
