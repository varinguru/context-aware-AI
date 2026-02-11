import asyncio
import os
from dotenv import load_dotenv

from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent, AgentOutput, ToolCallResult

# Load environment variables
load_dotenv()

# Configure Gemini as the LLM
llm = GoogleGenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="models/gemini-2.5-flash-lite",  # Try gemini-3-flash or gemini-2.5-flash-lite
)
Settings.llm = llm

# Initialize LlamaCloud Index
index = LlamaCloudIndex(
    name="chase_bank",
    api_key=os.getenv("LLAMACLOUD_API_KEY")
)

index2 = LlamaCloudIndex(
    name="chase2",
    api_key=os.getenv("LLAMACLOUD_API_KEY")
)

# Create global query engines
engine = index.as_query_engine(llm=llm)
engine2 = index2.as_query_engine(llm=llm)

# Create combined Chase query tool that queries both indexes in parallel
async def query_chase_combined(query: str) -> str:
    """Query both Chase indexes in parallel and combine results."""
    import asyncio

    # Query both engines in parallel
    results = await asyncio.gather(
        engine.aquery(query),
        engine2.aquery(query)
    )

    # Combine results
    combined = f"=== Index 1 Results ===\n{str(results[0])}\n\n=== Index 2 Results ===\n{str(results[1])}"
    return combined

chase_combined = FunctionTool.from_defaults(fn=query_chase_combined)

# Create add tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add_numbers)

# Create agent workflow
workflow = FunctionAgent(
    tools=[add_tool, chase_combined],
    llm=llm,
    system_prompt="You are an expert in Chase banking fees and procedures"
)


async def main():
    handler = workflow.run("""You have a Chase Total Checking account with $25 in your balance on
Monday morning. Throughout Monday, you make a $15 grocery purchase (debit card),
write a $20 check that gets cashed, and have a $25 automatic utility bill payment (ACH)
that processes. On Tuesday, you request a rush replacement for your lost debit card and
place a stop payment on another check over the phone with a banker. What is the total
amount in fees you would be charged, and when would each fee be assessed?""")

    # Stream the agent's reasoning process
    async for event in handler.stream_events():
        if isinstance(event, AgentOutput):
            for tool_call in event.tool_calls:
                print("-" * 20)
                print("Tool called: " + tool_call.tool_name)
                print("Tool arguments:")
                for key, value in tool_call.tool_kwargs.items():
                    print(f"  {key}: {value}")
                print("-" * 10)
        elif isinstance(event, ToolCallResult):
            print("Tool output: ", event.tool_output)

    result = await handler
    print(str(result))


if __name__ == "__main__":
    asyncio.run(main())
