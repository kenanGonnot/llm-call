try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover - handled via mock in tests
    class _Dummy:
        class ChatCompletion:
            @staticmethod
            def create(*args, **kwargs):  # pragma: no cover - replaced in tests
                raise RuntimeError("openai package is required")

    openai = _Dummy()


class Agent:
    """Simple wrapper around an OpenAI chat model."""

    def __init__(self, name: str, model: str = "gpt-3.5-turbo") -> None:
        self.name = name
        self.model = model

    def chat(self, messages):
        """Send messages to the OpenAI API and return the assistant's reply."""
        response = openai.ChatCompletion.create(model=self.model, messages=messages)
        return response.choices[0].message["content"]


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
