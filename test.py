# import asyncio

# import dreadnode

# # Initialize with default settings
# dreadnode.configure(
#     server="http://localhost:8000",  # Replace with your Dreadnode server URL
#     token="vvJTvN4KWsTNSAbIrwonxmsN9r9d3nlU",
#     # Or omit to use the DREADNODE_API_KEY environment variable
# )

# NAMES = ["Nick", "Will", "Brad", "Brian"]


# # Create a new task
# @dreadnode.task()
# async def say_hello(name: str) -> str:
#     return f"Hello, {name}!"


# async def main():
#     # Start a new run
#     with dreadnode.run("first-run"):
#         # Log parameters
#         dreadnode.log_params(
#             name_count=len(NAMES),
#         )

#         # Log inputs
#         dreadnode.log_input("names", NAMES)

#         # Run your tasks
#         greetings = [await say_hello(name) for name in NAMES]

#         # Save outputs
#         dreadnode.log_output("greetings", greetings)

#         # Track metrics
#         dreadnode.log_metric("accuracy", 0.65, step=0)
#         dreadnode.log_metric("accuracy", 0.85, step=1)

#         # Save the current script
#         dreadnode.log_artifact(__file__)


# asyncio.run(main())

import asyncio

import rigging as rg


async def main():
    async with rg.mcp("sse", url="https://mcp.deepwiki.com/sse") as mcp:
        pipeline = rg.get_generator("gemini-2.5-flash-preview-05-20").chat().using(mcp.tools)

        await rg.interact(pipeline)


asyncio.run(main())
