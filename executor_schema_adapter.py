"""
executor_schema_adapter.py
Helpers for QueryExecutor to resolve schema by fingerprint and apply light validation/casting.
"""

from typing import Any, Dict, Tuple
import dateutil.parser
from auto_schema import get_schema_by_fp

def resolve_schema_for_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    If payload contains 'schema_fp' -> fetch schema dict from LMDB via auto_schema.get_schema_by_fp
    Returns stored schema dict or None.
    """
    fp = payload.get("schema_fp")
    if not fp:
        return None
    return get_schema_by_fp(fp)

def cast_value_to_type(value: Any, target_type: str):
    """
    Cast a raw value (from query literal) to the inferred target_type from schema.
    Very small safe subset:
      - integer, number -> cast via int/float
      - string -> str
      - null -> None
      - date-like strings remain strings but try to parse on comparisons
    If casting fails, return original value (no crash).
    """
    if value is None:
        return None
    try:
        if target_type == "integer":
            return int(value)
        if target_type == "number":
            return float(value)
        if target_type == "boolean":
            if isinstance(value, bool):
                return value
            if str(value).lower() in ("true", "1", "t"):
                return True
            if str(value).lower() in ("false", "0", "f"):
                return False
            return bool(value)
        # leave date handling to the executor; here we attempt to parse into datetime for comparisons
        if target_type == "string":
            return str(value)
        # arrays/objects: no casting attempt
        return value
    except Exception:
        return value


def get_field_type_from_schema(schema_stored: Dict[str, Any], field_path: str) -> str:
    """
    field_path e.g. "meta.published_at" or "chunks.text" (top-level only best-effort).
    This is naive: supports only top-level keys and chunks.* lookup.
    """
    if not schema_stored:
        return None
    schema = schema_stored.get("schema") or schema_stored
    props = schema.get("properties", {})
    if field_path.startswith("chunks."):
        # look into chunks.items.properties
        _, sub = field_path.split(".", 1)
        ch = props.get("chunks", {}).get("items", {}).get("properties", {})
        return ch.get(sub, {}).get("type")
    return props.get(field_path, {}).get("type")
