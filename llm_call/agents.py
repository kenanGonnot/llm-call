try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover - handled via mock in tests
    class _Dummy:
        class ChatCompletion:
            @staticmethod
            def create(*args, **kwargs):  # pragma: no cover - replaced in tests
                raise RuntimeError("openai package is required")

    openai = _Dummy()

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - handled via mock in tests
    class _RequestsDummy:
        @staticmethod
        def post(*args, **kwargs):
            raise RuntimeError("requests package is required")

    requests = _RequestsDummy()

class Agent:
    """Simple wrapper around an OpenAI chat model with a responsibility."""

    def __init__(self, name: str, responsibility: str, model: str = "gemma-3n-e4b-it-mlx") -> None:
        self.name = name
        self.model = model
        self.responsibility = responsibility
        self.registry: dict[str, str] | None = None

    def update_registry(self, registry: dict[str, str]) -> None:
        """Store the system registry excluding this agent."""
        self.registry = registry

    def chat(self, messages):
        """Send a raw list of messages to the chat API and return the reply."""
        url = "http://localhost:1234/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def reply(self, message: str) -> str:
        """Reply to a single user message."""
        return self.chat([{"role": "user", "content": message}])


class CommandAgent(Agent):
    """Agent executing shell commands instead of using an LLM."""

    def __init__(self, name: str, responsibility: str, root: str) -> None:
        super().__init__(name, responsibility)
        self.root = root

    def reply(self, message: str) -> str:  # type: ignore[override]
        import subprocess

        result = subprocess.run(
            message,
            shell=True,
            capture_output=True,
            text=True,
            cwd=self.root,
        )
        output = result.stdout + result.stderr
        return output.strip()


class MultiAgentSystem:
    """Coordinate a list of agents in a simple round-robin fashion."""

    def __init__(self, agents):
        self.agents = agents
        self.started = False

    def start(self) -> None:
        """Distribute the registry of all agents to each participant."""
        if self.started:
            return
        registry = {agent.name: agent.responsibility for agent in self.agents}
        for agent in self.agents:
            # each agent should not receive itself in the registry
            other = {name: resp for name, resp in registry.items() if name != agent.name}
            agent.update_registry(other)
        self.started = True

    def _get_agent(self, name: str) -> Agent:
        for agent in self.agents:
            if agent.name == name:
                return agent
        raise ValueError(f"Unknown agent: {name}")

    def send_message(self, agent_name: str, body: str) -> str:
        """Send ``body`` to the specified agent and return its reply."""
        self.start()
        agent = self._get_agent(agent_name)
        return agent.reply(body)

    def run(self, prompt: str, turns: int = 1):
        self.start()
        messages = [{"role": "user", "content": prompt}]
        for _ in range(turns):
            for agent in self.agents:
                reply = agent.chat(messages)
                messages.append(
                    {"role": "assistant", "name": agent.name, "content": reply}
                )
        return messages
