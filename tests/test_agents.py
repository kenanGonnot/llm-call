from unittest.mock import patch

from llm_call.agents import Agent, MultiAgentSystem


class DummyResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def fake_post(url, json, headers):
    return DummyResponse(f"echo:{json['messages'][-1]['content']}")


def test_system_run():
    agent = Agent(name="tester")
    system = MultiAgentSystem([agent])

    with patch("llm_call.agents.requests.post", side_effect=fake_post):
        messages = system.run("hello", turns=1)
    assert messages[-1]["content"] == "echo:hello"
    assert messages[-1]["name"] == "tester"
