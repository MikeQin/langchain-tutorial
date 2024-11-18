# LangChain Tutorial

### Step 1: Install LangChain and Dependencies
LangChain requires Python and some associated libraries. You’ll also need an OpenAI account.

1. Install LangChain and the OpenAI SDK:

```bash
pip install langchain openai
```

2. Install additional dependencies:

```bash
pip install python-dotenv
```

3. Set up your OpenAI API key:

  - Create a .env file in your project directory and add:
```makefile
OPENAI_API_KEY=your_openai_api_key_here
```
  - Load the API key in your code:
```python
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
```

### Step 2: Import LangChain and Set Up OpenAI LLM
Create a script to test basic LangChain functionality.

**Code:**
```python
from langchain.llms import OpenAI

# Initialize the OpenAI model
llm = OpenAI(model="gpt-4", temperature=0.7)

# Test the LLM
prompt = "Write a creative tagline for an AI-powered application."
response = llm(prompt)
print(response)
```

### Step 3: Use Prompt Templates

LangChain provides PromptTemplate to dynamically format prompts.

**Code:**
```python
from langchain import PromptTemplate
from langchain.llms import OpenAI

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
```

### Step 4: Build a Chain
LangChain’s LLMChain combines prompts and LLMs into reusable chains.

**Code:**
```python
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain import PromptTemplate

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
```

### Step 5: Add Memory

LangChain supports memory to maintain the context of conversations.

**Code:**
```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

# Set up memory and LLM
memory = ConversationBufferMemory()
llm = OpenAI(model="gpt-4", temperature=0.7)

# Create a conversational chain
conversation = ConversationChain(llm=llm, memory=memory)

# Interact with the chain
print(conversation.run("Hello!"))
print(conversation.run("Can you remember what I just said?"))
```

### Step 6: Use External Tools

LangChain can integrate external tools, such as databases or APIs.

**Example: Use Search API with LangChain**
```python
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
```

**Example Output:**
```bash
> Entering new AgentExecutor chain...
I will use the Simple Tool to process the input.

Processed input: LangChain is awesome.
> Finished chain.
```

### Step 7: Create a Conversational Agent

Combine chains, memory, and tools to build a basic chatbot.

**Code:**
```python
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
```

### Step 8: Explore Advanced Topics

1. Custom Chains: Build unique workflows by chaining multiple LLM calls.
2. Fine-Tuning: Train custom models with OpenAI and integrate them into LangChain.
3. Deployment: Host your AI app on platforms like Heroku, AWS, or Vercel.

### Next Steps

Would you like to:

1. Work on a specific project (e.g., a chatbot)?
2. Explore more advanced LangChain features?
3. Get additional resources for learning?

#### More Complex Chained Examples

[Complex Chained Examples](./ComplexChainedExamples.md)