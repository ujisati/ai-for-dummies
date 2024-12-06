<h1><div align="center">Serverless AI for Dummies</div></h1>

Tools for Modal. Intended for personal use and experiments.

## Tools

### My Llamas

My Llamas is a Modal app that can download models to a Modal volume. You can specify Ollama model names, or specify Huggingface paths to files, including multipart files.

The app has a scale-to-zero Ollama inference server with token authentication through FastAPI. The FastAPI app just proxies requests to the Ollama REST server running in the container.

Connect from a chat client of your choice like open-webui. You can also use `client.py` for quick testing, which I pulled from [modal examples](https://github.com/modal-labs/modal-examples/tree/main).

You get $30 of free credits per month from Modal.

#### Config

All of the models you want to store, whether through Ollama or Huggingface, are stored in `args.py`. Create it and fill it with the following example.

The beauty here is you can create as many configs as you like and all the models will be downloaded and built to the volume.

```python
from typing import Literal

DOWNLOAD = {
    "qwen": {
        "hf_path": "Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-fp16.gguf",
        "pet_name": "Qwen",
        "modelfile": "qwen-test",
        "gpu": "t4:1",
    },
}
DOWNLOAD_DEFAULT = "luminum"

PULL = {"qwq": {"gpu": "l4:1"}}
PULL_DEFAULT = "qwq"

CHOSEN_SOURCE: Literal["download"] | Literal["pull"] = "download"
```

##### Ollama source

```bash
modal run tame_llama::pull
```

This will use `PULL_DEFAULT` config.

##### Huggingface source

```bash
modal run --detach tame_llama::download
modal run --detach tame_llama::compile
```

This will use `DOWNLOAD_DEFAULT` config.

##### Testing

Test with qwen to make sure everything is working.

#### Setup

```bash
  # first, create a python virtual environment please
  # activate it

  # See Config section to populate this file
  touch args.py

  pip install modal
  modal secret create huggingface-secret HF_TOKEN=<secret>
  modal secret create llama-food LLAMA_FOOD=<secret> # Bearer auth for fastapi
  modal setup
```

```bash
modal run --detach tame_llama::pull
```

### Commands

See the modal CLI for app, shell, deploy, secret, volume commands, etc.

#### Ollama logs on server

```bash
journalctl -u ollama --no-pager
```

#### Handy alias for testing

```bash
alias chat='python client.py \
  --app-name=myllamas-gpu-l4-1-myllamas \
  --function-name=serve \
  --model=qwq \
  --max-tokens 1000 \
  --api-key $(echo $LLAMA_FOOD) \
  --temperature 0.9 \
  --frequency-penalty 1.03 \
  --chat'
```
