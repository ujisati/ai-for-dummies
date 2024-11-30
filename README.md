<h1><div align="center">☁️ Serverless AI for Dummies🦙</div></h1>

> Inference.
> 
> *Fast*, *free*, and *from anywhere*.

Tools for Modal. Intended for personal use and experiments.

## Tools

### My Llamas

My Llamas is a Modal app that can download individual or multipart files from Huggingface to a Modal volume. The app has a scale-to-zero Ollama GPU inference server with Token authentication through FastAPI. Look at the docs for the Ollama REST API. The FastAPI server just proxies them to the Ollama server running in the container. Connect from a chat client of your choice like open-webui. You can also use `client.py`. See helpers.

#### Run

```bash
  # first, create a python virtual environment please
  # activate it

  pip install modal
  modal secret create huggingface-secret HF_TOKEN=<secret>
  modal secret create llama-food LLAMA_FOOD=<secret> # Bearer auth for fastapi
  modal setup
  modal run tame_llama --config-id <id> # Download/create model. See Config section.
  modal deploy my_llamas
  source helpers.sh; chat # inference is that easy now???
  modal app stop MyLlamas
```

#### Config

Arguments to functions are validated by Pydantic models.

`config-id` is passed to Modal function calls like so

```bash
modal run tame_llama --config-id "luminum"
```

The beauty here is you can create as many configs as you like and all the Llamas will be downloaded and Modelfiles built to sane places in the Volume, when you run `modal run tame_llama`

```python
config_data = {
    "luminum": {
        "path": "mradermacher/Luminum-v0.1-123B-i1-GGUF/Luminum-v0.1-123B.i1-IQ3_XS.gguf",
        "multipart": True,
        "name": "Luminum",
        "modelfile": "luminum",
    },
    "qwen": {
        "path": "Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-fp16.gguf",
        "name": "qwen-test",
        "modelfile": "qwen-test",
    },
}
```

Test with qwen to make sure everything is working.

### Commands

See the modal CLI for app, shell, deploy, secret, volume, etc commands.

#### Ollama logs on server

```bash
journalctl -u ollama --no-pager
```

#### Sanity check if client.py isn't working

```bash
curl https://ujisati--myllamas-myllamas-serve.modal.run/api/chat -d '{
  "model": "Luminum",
  "messages": [
    {
      "role": "user",
      "content": "Why is the sky yellow?"
    }
  ]
}'
```

#### Helpers

```bash
source helpers.sh
chat # useful for testing your OpenAI API parameters
```
