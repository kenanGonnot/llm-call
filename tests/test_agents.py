from unittest.mock import patch

from llm_call.agents import Agent, MultiAgentSystem


class DummyResponse:
    def __init__(self, content):
        self.choices = [type("obj", (), {"message": {"content": content}})]


def fake_create(model, messages):
    return DummyResponse(f"echo:{messages[-1]['content']}")


def test_system_run():
    agent = Agent(name="tester")
    system = MultiAgentSystem([agent])

    with patch("llm_call.agents.openai.ChatCompletion.create", side_effect=fake_create):
        messages = system.run("hello", turns=1)

    assert messages[-1]["content"] == "echo:hello"
