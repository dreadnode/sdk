from dreadnode.agent.agent import Agent
from dreadnode.agent.tools.bbot.tool import BBotTool

agent = Agent(
    name="bbot-agent",
    description="An agent that uses BBOT to perform various tasks.",
    model="gpt-4",
    tools=[BBotTool()],
)
