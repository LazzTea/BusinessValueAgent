from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

from financialModel import train_ai_model

# Load env file
load_dotenv()

# Get question from user
query = input("What company do you want financial information about today?")

# Set up Research Response format
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# Use llm to break down the user's question
llm = ChatOpenAI(model="gpt-4o")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

begin_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a financial analysis assistant. Your job is to extract key information about a company a user is asking about.

            You must return a JSON object matching this schema:
            - topic: The **NASDAQ ticker symbol** of the company (e.g., AAPL for Apple) as a string.
            - summary: A one-paragraph summary of the company’s value position.
            - sources: A list of sources you used or referenced.
            - tools_used: List the tools you would use to gather this information (e.g., Yahoo Finance, SEC Filings).\n{format_instructions}
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

agent_executor = AgentExecutor(agent=begin_agent, tools=[], verbose=True)
raw_response = agent_executor.invoke({"query": "What company do you want financial information about today? " + query})
print(raw_response)

parsed = parser.parse(raw_response["output"])
symbol = parsed.topic

data, results = train_ai_model(symbol)

if(data is False):
    print("The company you are looking for information on does not have enough data to analyze or does not exist")
    exit(1)

print(f'data list {data}\n')
print(f'Results {results}\n')



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

parsed = parser.parse(raw_response["output"])
topic = parsed.topic
summary = parsed.summary

print("\n Here is what the user's output would be \n")
print(topic + ": " + summary + "\n")
