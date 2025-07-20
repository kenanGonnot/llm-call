from .agents import Agent, MultiAgentSystem


def main():
    """Run a very small multi-agent demo."""
    agents = [Agent("Alice", "Developer"), Agent("Bob", "Can execute unix commands line")]
    system = MultiAgentSystem(agents)
    try:
        messages = system.run("Fais moi un projet minimalist Python", turns=2)
        for msg in messages:
            speaker = msg.get("name", msg["role"])
            print(f"{speaker}: {msg['content']}")
    except RuntimeError as exc:
        print(exc)
        print("Please install the openai package and configure API credentials.")


if __name__ == "__main__":
    main()
