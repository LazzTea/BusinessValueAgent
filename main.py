from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

from financialModel import train_ai_model
data, results = train_ai_model("AAPL")

if(data is False):
    print("data is false")
    exit(1)

print(f'data list {data}\n')
print(f'Results {results}\n')


load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Train the AI to evaluate companies

# Use llm to query the user
llm = ChatOpenAI(model="gpt-4o")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

begin_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Return the nasdaq symbol for the company being talked about\n{format_instructions}
            """,
        ),
        ("placeholder","{chat_history}"),
        ("human","{query}"),
        ("placeholder","{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

begin_agent = create_tool_calling_agent(
    llm = llm,
    prompt = begin_prompt,
    tools = []
)

# agent_executor = AgentExecutor(agent=begin_agent, tools=[], verbose=True)
# raw_response = agent_executor.invoke({"query": "Is Apple under or overvalued?"})
# print(raw_response)


final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Given info about the given company's value summarize it for the user.\n{format_instructions}
            """,
        ),
        ("placeholder","{chat_history}"),
        ("human","{query}"),
        ("placeholder","{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

final_agent = create_tool_calling_agent(
    llm = llm,
    prompt = final_prompt,
    tools = []
)

agent_executor = AgentExecutor(agent=final_agent, tools=[], verbose=True)
raw_response = agent_executor.invoke({"query": str(data) + " " + str(results)})
