from unittest.mock import patch

from llm_call.agents import Agent, CommandAgent, MultiAgentSystem


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
    agent = Agent(name="tester", responsibility="echo")
    system = MultiAgentSystem([agent])

    with patch("llm_call.agents.requests.post", side_effect=fake_post):
        messages = system.run("hello", turns=1)
    assert messages[-1]["content"] == "echo:hello"
    assert messages[-1]["name"] == "tester"


def test_registry_distribution():
    agent1 = Agent(name="agent1", responsibility="resp1")
    agent2 = Agent(name="agent2", responsibility="resp2")
    system = MultiAgentSystem([agent1, agent2])

    with patch("llm_call.agents.requests.post", side_effect=fake_post):
        system.run("hi", turns=0)

    assert agent1.registry == {"agent2": "resp2"}
    assert agent2.registry == {"agent1": "resp1"}


def test_agent_reply():
    agent = Agent(name="tester", responsibility="echo")
    with patch("llm_call.agents.requests.post", side_effect=fake_post):
        response = agent.reply("ping")
    assert response == "echo:ping"


def test_send_message():
    agent = Agent(name="tester", responsibility="echo")
    system = MultiAgentSystem([agent])
    with patch("llm_call.agents.requests.post", side_effect=fake_post):
        reply = system.send_message("tester", "hello")
    assert reply == "echo:hello"


def test_command_agent(tmp_path):
    agent = CommandAgent(name="runner", responsibility="cmd", root=str(tmp_path))
    output = agent.reply("echo hi")
    assert output == "hi"
