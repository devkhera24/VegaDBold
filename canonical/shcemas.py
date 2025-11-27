from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query_str: str = Field(..., description="Query string to execute")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Optional named parameters")

class HealthResponse(BaseModel):
    status: str = Field(..., description="health status, e.g. \"ok\"")