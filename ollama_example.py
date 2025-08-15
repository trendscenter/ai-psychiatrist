import json
import requests

OLLAMA_NODE = "arctrddgxa003" # TODO: Change this variable to the node where Ollama is running
BASE_URL = f"http://{OLLAMA_NODE}:11434/api/chat"

model = "gemma3-optimized:27b" # TODO: Change this variable to the model you want to use
message = "What is the capital of France?" # TODO: Change this variable to the message you want to ask the model

response = requests.post(
  BASE_URL,
  json = {
    "model": model,
    "messages": [{"role": "user", "content": message}],
    "stream": False
  }
)

print(json.dumps(response.json(), indent=2))