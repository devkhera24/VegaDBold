from typing import Any, Dict, Optional
import asyncio
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# canonical schemas
from canonical.schemas import QueryRequest

# executor (we provide a stub in core/query/executor.py)
from core.query.executor import execute_query

app = FastAPI(title="API / Orchestrator - Member A")

logger = logging.getLogger("api.main")
logging.basicConfig(level=logging.INFO)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query")
async def query_endpoint(payload: QueryRequest):
    query_str = payload.query_str
    params = payload.params or {}

    try:
        result = execute_query(query_str, params)
        if asyncio.iscoroutine(result):
            result = await result

        result_jsonable = jsonable_encoder(result)
        return JSONResponse(content=result_jsonable)

    except Exception as e:
        logger.exception("Error executing query")
        raise HTTPException(status_code=500, detail=str(e))