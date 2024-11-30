#!/usr/bin/bash
restart_llamas ()
{
  modal app stop MyLlamas
  modal deploy my_llamas
}

alias chat='python client.py \
  --app-name=myllamas-myllamas \
  --function-name=serve \
  --model=Luminum \
  --max-tokens 1000 \
  --api-key $(echo $LLAMA_FOOD) \
  --system-prompt "$(cat system.txt)" \
  --temperature 0.9 \
  --frequency-penalty 1.03 \
  --chat'
