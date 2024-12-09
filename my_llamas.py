import os

import modal
from settings import AppSettings

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "lshw")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("fastapi[standard]==0.115.4", "httpx")
    .env(
        {
            "OLLAMA_HOST": "127.0.0.1:11434",
            "OLLAMA_MODELS": "/models/.ollama",
        }
    )
)

try:
    volume = modal.Volume.lookup(
        AppSettings.MODELS_FOLDER_NAME, create_if_missing=False
    )
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_llama.py")


app = modal.App(
    f"MyLlamas-GPU-{str(AppSettings.gpu).replace(':', '-')}",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("llama-food"),
    ],
)


@app.cls(
    image=image,
    gpu=AppSettings.gpu,
    container_idle_timeout=5 * AppSettings.MINUTES,
    timeout=5 * AppSettings.MINUTES,
    allow_concurrent_inputs=1,
    volumes={AppSettings.MODELS_DIR: volume},
)
class MyLlamas:

    @modal.enter()
    def enter(self):
        import subprocess
        import time

        subprocess.Popen(["ollama", "serve"], close_fds=True)
        time.sleep(2)

    @modal.asgi_app()
    def serve(self):
        import os
        import time
        from contextlib import asynccontextmanager, contextmanager

        import fastapi
        import httpx
        import starlette
        from fastapi.middleware.cors import CORSMiddleware

        volume.reload()

        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            async with httpx.AsyncClient(base_url="http://127.0.0.1:11434/") as client:
                yield {"client": client}

        web_app = fastapi.FastAPI(
            title=f"OpenAI-compatible {AppSettings.MODELS_FOLDER_NAME} server",
            description="Run an OpenAI-compatible LLM server on modal.com 🚀",
            version="0.0.1",
            docs_url="/docs",
            lifespan=lifespan,
        )

        http_bearer = fastapi.security.HTTPBearer()
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        timeout = httpx.Timeout(30.0, connect=120.0)

        async def is_authenticated(
            api_key: fastapi.security.HTTPAuthorizationCredentials = fastapi.Security(
                http_bearer
            ),
        ):
            if api_key.credentials != os.environ["LLAMA_FOOD"]:
                raise fastapi.HTTPException(
                    status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                )
            return {"username": "authenticated_user"}

        async def reverse_proxy(
            request: fastapi.Request,
            path: str,
            background_tasks: fastapi.BackgroundTasks,
        ):
            client = request.state.client
            url = path
            headers = {
                k.decode(): v.decode() for k, v in request.headers.raw if k != b"host"
            }
            method = request.method
            content = await request.body()

            request = client.build_request(
                headers=headers,
                method=method,
                url=url,
                content=content,
                timeout=timeout,
            )
            response = await client.send(request, stream=True)
            background_tasks.add_task(response.aclose)
            return fastapi.responses.StreamingResponse(
                response.aiter_raw(),
                headers=response.headers,
            )

        router = fastapi.APIRouter()

        router.add_api_route(
            "/{path:path}",
            reverse_proxy,
            methods=["GET", "POST", "HEAD"],
            dependencies=[fastapi.Depends(is_authenticated)],
        )

        web_app.include_router(router)
        return web_app
