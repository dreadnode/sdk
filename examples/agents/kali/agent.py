from dreadnode.agent.agent import Agent
from dreadnode.agent.tools.kali.tool import KaliTool

agent = Agent(
    name="kali-agent",
    description="An agent that uses the Kali toolset to perform penetration testing tasks.",
    model="gpt-4",
    tools=[KaliTool()],
)


def main() -> None:
    agent.run("Perform a network scan on the target IP")


if __name__ == "__main__":
    main()
