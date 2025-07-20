try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover - handled via mock in tests
    class _Dummy:
        class ChatCompletion:
            @staticmethod
            def create(*args, **kwargs):  # pragma: no cover - replaced in tests
                raise RuntimeError("openai package is required")

    openai = _Dummy()

import requests


class Agent:
    """Simple wrapper around an OpenAI chat model."""

    def __init__(self, name: str, model: str = "gemma-3n-e4b-it-mlx") -> None:
        self.name = name
        self.model = model

    def chat(self, messages):
        """Envoie les messages à l'API locale et retourne la réponse de l'assistant."""
        url = "http://localhost:1234/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class MultiAgentSystem:
    """Coordinate a list of agents in a simple round-robin fashion."""

    def __init__(self, agents):
        self.agents = agents

    def run(self, prompt: str, turns: int = 1):
        messages = [{"role": "user", "content": prompt}]
        for _ in range(turns):
            for agent in self.agents:
                reply = agent.chat(messages)
                messages.append({"role": "assistant", "content": reply})
        return messages
