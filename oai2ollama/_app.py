import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse
from maping import Recorder
from maping.asgi import MapingMiddleware

from .config import env

recorder = Recorder(service="oai2ollama")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await recorder.start()
    yield
    await recorder.shutdown()


_app = FastAPI(lifespan=_lifespan)


@Depends
async def _new_client():
    from httpx import AsyncClient

    async with AsyncClient(base_url=str(env.base_url), headers={"Authorization": f"Bearer {env.api_key}"}, timeout=60, http2=True, follow_redirects=True) as client:
        yield client


@_app.get("/api/tags")
async def models(client=_new_client):
    res = await client.get("/models")
    res.raise_for_status()
    try:
        data = res.json()["data"]
    except (KeyError, TypeError):
        data = []
    models_map = {i["id"]: {"name": i["id"], "model": i["id"]} for i in data} | {i: {"name": i, "model": i} for i in env.extra_models}
    return {"models": list(models_map.values())}


@_app.post("/api/show")
async def show_model():
    return {
        "model_info": {"general.architecture": "CausalLM"},
        "capabilities": ["completion", *env.capabilities],
    }


@_app.get("/v1/models")
async def list_models(client=_new_client):
    res = await client.get("/models")
    res.raise_for_status()
    return res.json()


@_app.post("/v1/chat/completions")
async def chat_completions(request: Request, client=_new_client):
    data = await request.json()

    if data.get("stream", False):

        async def stream():
            async with client.stream("POST", "/chat/completions", json=data) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

        return StreamingResponse(stream(), media_type="text/event-stream")

    else:
        res = await client.post("/chat/completions", json=data)
        res.raise_for_status()
        return res.json()


@_app.get("/api/version")
async def ollama_version():
    return {"version": "0.12.10"}


app = MapingMiddleware(_app, recorder=recorder) if os.environ.get("MAPING_KEY") else _app
