import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ------------------------------------------
# Test LLM
# ------------------------------------------
# Initialize the OpenAI model
llm = OpenAI(model="gpt-4", temperature=0.7)

# Test the LLM
prompt = "Write a creative tagline for an AI-powered application."
response = llm(prompt)
print(response)

# ------------------------------------------
# Use Prompt Templates
# ------------------------------------------
# Define a template with placeholders
template = PromptTemplate(
    input_variables=["product"],
    template="What is a catchy slogan for {product}?"
)

# Use the template
prompt = template.format(product="an AI chatbot")
llm = OpenAI(model="gpt-4", temperature=0.7)
response = llm(prompt)
print(response)

# ------------------------------------------
# Build a Chain
# ------------------------------------------
# Define prompt and LLM
template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)
llm = OpenAI(model="gpt-4", temperature=0.7)

# Create a chain
chain = LLMChain(llm=llm, prompt=template)

# Run the chain
response = chain.run("quantum computing")
print(response)


# ------------------------------------------
# Add Memory
# ------------------------------------------
# Set up memory and LLM
memory = ConversationBufferMemory()
llm = OpenAI(model="gpt-4", temperature=0.7)

# Create a conversational chain
conversation = ConversationChain(llm=llm, memory=memory)

# Interact with the chain
print(conversation.run("Hello!"))
print(conversation.run("Can you remember what I just said?"))

# ------------------------------------------
# Use External Tools, Combining Tools and LLMs
# ------------------------------------------
from langchain.tools import Tool
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms import OpenAI

# Define a tool that processes input
def simple_tool(input: str) -> str:
    return f"Processed input: {input}"

tool = Tool(
    name="Simple Tool",
    func=simple_tool,
    description="This tool processes the input and provides a result."
)

# Initialize the OpenAI LLM
llm = OpenAI(model="gpt-4", temperature=0.7)

# Initialize an agent that can use the tool
agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Ask the agent to use the tool
response = agent.run("Use the tool to process the text 'LangChain is awesome.'")
print(response)

# ------------------------------------------
# Create a Conversational Agent
# ------------------------------------------
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# Define the tools
def calculator_tool(input):
    try:
        return str(eval(input))
    except Exception as e:
        return f"Error: {e}"

tools = [
    Tool(name="Calculator", func=calculator_tool, description="Perform calculations.")
]

# Set up memory and LLM
memory = ConversationBufferMemory()
llm = OpenAI(model="gpt-4", temperature=0.7)

# Initialize an agent
agent = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory)

# Interact with the agent
print(agent.run("What is 25 + 78?"))
print(agent.run("What is the result multiplied by 2?"))


# ------------------------------------------
# Build a Chain
# ------------------------------------------