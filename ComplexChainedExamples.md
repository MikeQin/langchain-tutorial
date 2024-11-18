# Complex Chained Example

## Example 1

Here's a more complex chained example combining multiple steps and tools with LangChain. We'll build a pipeline where:

1. The user provides a topic.
2. The LLM generates a question about the topic.
3. A tool fetches additional information based on the question.
4. The LLM combines everything into a coherent summary.

### Example: Multi-Step Pipeline

```python
from langchain.tools import Tool
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Step 1: Initialize the LLM
llm = OpenAI(model="gpt-4", temperature=0.7)

# Step 2: Define the Prompt Templates
# 2.1 Generate a question about a topic
question_template = PromptTemplate(
    input_variables=["topic"],
    template="Generate an interesting question about the topic: {topic}"
)

# 2.2 Summarize the topic and fetched information
summary_template = PromptTemplate(
    input_variables=["topic", "info"],
    template=(
        "Summarize the topic '{topic}' using the following information: {info}. "
        "Ensure the summary is detailed and informative."
    )
)

# Step 3: Define Chains
# Chain to generate a question
question_chain = LLMChain(llm=llm, prompt=question_template)

# Mock tool for fetching additional information
def info_tool(question: str) -> str:
    # Simulating a fetch operation; in a real scenario, this could query an API or database
    return f"This is the fetched information about: '{question}'"

info_tool_wrapper = Tool(
    name="InfoTool",
    func=info_tool,
    description="Fetches information based on a question."
)

# Chain to summarize the topic and information
summary_chain = LLMChain(llm=llm, prompt=summary_template)

# Combine chains into a sequential workflow
pipeline = SimpleSequentialChain(chains=[question_chain, summary_chain])

# Step 4: Execute the Workflow
topic = "Artificial Intelligence in Healthcare"
question = question_chain.run(topic)  # Step 1: Generate question
fetched_info = info_tool_wrapper.run(question)  # Step 2: Fetch information
summary = summary_chain.run({"topic": topic, "info": fetched_info})  # Step 3: Summarize

print("Generated Question:", question)
print("Fetched Information:", fetched_info)
print("Summary:", summary)

```

### Explanation of the Workflow

1. Generate a Question:
  - The first LLMChain takes a topic and generates a question.
  - For example, for the topic "Artificial Intelligence in Healthcare", the question might be:

```t
"How is AI being used to improve diagnostic accuracy in healthcare?"
```

2. Fetch Additional Information:

  * The generated question is passed to a tool (here simulated as info_tool).
  * In a real-world application, this could call an external API like Wikipedia or a knowledge base.

3. Summarize the Information:

  * A second LLMChain takes the original topic and fetched information to create a detailed summary.

### Output Example

```bash
Generated Question: What are the current challenges in using AI for diagnosing diseases in healthcare?

Fetched Information: This is the fetched information about: 'What are the current challenges in using AI for diagnosing diseases in healthcare?'

Summary: Artificial Intelligence in Healthcare faces several challenges, especially in diagnosing diseases. Current challenges include data privacy, ensuring the reliability of AI models, and addressing biases in medical datasets. Using the fetched information, it's clear that while AI holds promise, these barriers must be overcome for widespread adoption.
```

### Use Case Highlights:

* Chained logic: Sequential tasks depend on prior outputs.
* Tool integration: Leverage APIs or databases to augment LLM capabilities.
* Scalable workflows: Add more chains for advanced pipelines.

## Example 2

Let’s enhance the example by adding live API calls using web search and building a more complex workflow with LangChain.

### Part 1: Adding Live API Calls Using Web Search
We’ll integrate a web search tool to fetch live information.

**Setup**

1. Install the required library for web search:

```bash
pip install google-search-results
```

2. Sign up for the SerpAPI and get an API key for Google search.

3. Add your SERPAPI_API_KEY to the `.env` file:

```makefile
SERPAPI_API_KEY=your_serpapi_api_key
```

**Code for Web Search Integration:**
```python
from langchain.tools import Tool
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# Initialize Google Search tool
search = GoogleSearchAPIWrapper(serpapi_api_key=serpapi_api_key)

# Wrap the search function as a tool
search_tool = Tool(
    name="WebSearch",
    func=search.run,
    description="Use this tool to perform web searches for up-to-date information."
)

# Initialize OpenAI LLM
llm = OpenAI(model="gpt-4", temperature=0.7)

# Define an agent with the search tool
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Use the agent to perform a search and summarize
query = "What are the latest advancements in AI for healthcare?"
response = agent.run(query)
print("Agent Response:", response)
```

#### Output Example

The agent will use the web search tool, fetch results, and summarize:

```rust
> Entering new AgentExecutor chain...
Using the WebSearch tool for the query.

Found the following information: [Latest advancements in AI for healthcare include the use of machine learning for drug discovery, improving diagnostic tools, and enhancing personalized treatment plans.]

> Finished chain.
```

### Part 2: Complex Workflow with Live API Calls and Summarization
Now, we’ll chain multiple steps:

1. Generate a specific search query for a topic.
2. Perform a web search using the generated query.
3. Summarize the fetched results.

**Code for Complex Workflow:**

```python
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# Step 1: Initialize OpenAI LLM
llm = OpenAI(model="gpt-4", temperature=0.7)

# Step 2: Define prompt templates
query_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate a detailed search query for the topic: {topic}"
)

summary_prompt = PromptTemplate(
    input_variables=["topic", "results"],
    template="Summarize the topic '{topic}' using these search results: {results}"
)

# Step 3: Define chains
query_chain = LLMChain(llm=llm, prompt=query_prompt)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Step 4: Initialize search tool
search = GoogleSearchAPIWrapper(serpapi_api_key=serpapi_api_key)
search_tool = Tool(
    name="WebSearch",
    func=search.run,
    description="Perform live web searches."
)

# Step 5: Complex Workflow
def complex_workflow(topic: str):
    # Step 1: Generate search query
    query = query_chain.run(topic)
    print(f"Generated Query: {query}")

    # Step 2: Perform web search
    search_results = search_tool.run(query)
    print(f"Search Results: {search_results}")

    # Step 3: Summarize search results
    summary = summary_chain.run({"topic": topic, "results": search_results})
    return summary

# Run the workflow
topic = "AI in cancer treatment"
final_summary = complex_workflow(topic)
print("Final Summary:", final_summary)
```

#### Workflow Breakdown

1. Generate Search Query:
  * The LLM creates a specific search query based on a topic.
  * Example Output: "Latest AI techniques in cancer treatment and their impact on diagnostics."

2. Perform Web Search:
  * The search query is passed to the web search tool, which fetches live information.

3. Summarize Results:
  * The LLM summarizes the fetched information into a coherent paragraph.

#### Example Output

```rust
Generated Query: Latest AI techniques in cancer treatment and their impact on diagnostics.
Search Results: Recent studies show AI is being used to identify cancerous tissues, predict treatment outcomes, and analyze medical imaging.
Final Summary: Artificial Intelligence is revolutionizing cancer treatment by improving the accuracy of diagnosis, analyzing complex medical imaging, and predicting treatment outcomes with advanced algorithms.
```

## Next Steps

1. Add More Tools:
  * Use APIs like Wikipedia, Wolfram Alpha, or custom APIs for more specialized information.

2. Fine-Tune Workflow:
  * Introduce conditional logic: Based on the fetched results, decide the next steps.

3. Deploy:
  * Package the workflow as a chatbot or API for public use.