# AI Topic Explorer

Let’s extend the example by adding Wikipedia as a second tool, wrapping everything into a Streamlit web application, and providing a solid structure with a catchy name. The application will allow users to explore topics by combining Google Search and Wikipedia summaries, all neatly packaged with Streamlit.

## Web Application Name

**"AI Topic Explorer"**
A powerful assistant that combines live web search and Wikipedia to give users a deep understanding of any topic.

## Code for AI Topic Explorer

### Install Required Libraries

Make sure to install the necessary dependencies:

```bash
pip install streamlit langchain openai python-dotenv google-search-results wikipedia-api
```

### Complete Application Code

```python
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
import wikipediaapi
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# Initialize tools
# Google Search tool
google_search = GoogleSearchAPIWrapper(serpapi_api_key=serpapi_api_key)

# Wikipedia tool
def fetch_wikipedia_summary(query: str) -> str:
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page = wiki_wiki.page(query)
    if page.exists():
        return page.summary[:1000]  # Return the first 1000 characters
    else:
        return "No Wikipedia page found for this topic."

wikipedia_tool = Tool(
    name="Wikipedia",
    func=fetch_wikipedia_summary,
    description="Fetches a summary of the topic from Wikipedia."
)

# Initialize LLM
llm = OpenAI(model="gpt-4", temperature=0.7)

# Define prompts
search_query_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate a detailed search query for the topic: {topic}"
)

summary_prompt = PromptTemplate(
    input_variables=["topic", "google_results", "wikipedia_results"],
    template=(
        "Summarize the topic '{topic}' using these search results from Google: {google_results}, "
        "and this summary from Wikipedia: {wikipedia_results}. "
        "Ensure the summary is detailed and provides a clear understanding."
    )
)

# Define chains
search_query_chain = LLMChain(llm=llm, prompt=search_query_prompt)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Define the workflow
def explore_topic(topic: str) -> str:
    # Generate a search query
    query = search_query_chain.run(topic)
    
    # Fetch information from Google
    google_results = google_search.run(query)
    
    # Fetch information from Wikipedia
    wikipedia_results = fetch_wikipedia_summary(topic)
    
    # Summarize all the information
    final_summary = summary_chain.run({
        "topic": topic,
        "google_results": google_results,
        "wikipedia_results": wikipedia_results
    })
    
    return final_summary

# Streamlit app
st.title("AI Topic Explorer")
st.subheader("Deep dive into any topic with AI-powered summaries!")

# User input
topic = st.text_input("Enter a topic to explore:", "")

if topic:
    with st.spinner("Exploring the topic..."):
        try:
            summary = explore_topic(topic)
            st.success("Exploration Complete!")
            st.header("Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")

```

### How the Application Works

1. **User Input:** Enter a topic in the Streamlit interface.

2. **Workflow:**
  * Generate a search query using OpenAI’s LLM.
  * Use Google Search to fetch live information.
  * Fetch additional insights from Wikipedia.
  * Summarize everything into a coherent paragraph using OpenAI’s LLM.

3. **Output:** A detailed, AI-generated summary of the topic.

### To Run the Application

1. Save the code to a file, e.g., ai_topic_explorer.py.
2. Run the Streamlit server:
```bash
streamlit run ai_topic_explorer.py
```
3. Open your browser at the URL provided (default is http://localhost:8501).

### Example Interaction

* Input: "Artificial Intelligence in Medicine"
* Output Summary:
```rust
Artificial Intelligence in Medicine is transforming healthcare through innovations like predictive analytics, diagnostic tools, and personalized treatment plans. Recent advancements, as found in Google search, include AI's role in analyzing medical imaging and drug discovery. Additionally, Wikipedia highlights the use of machine learning in predicting disease outbreaks and enhancing patient outcomes. Together, these insights reveal how AI is driving a healthcare revolution.
```

This app is ready to be your personalized research assistant! 😊

## Next Step

Here are some ways we can enhance the "AI Topic Explorer" app for even more powerful capabilities:

### 1. Multi-Modal Outputs

* Feature: Include images, graphs, or videos related to the topic.
* How:
  * Use APIs like `Unsplash` for images.
  * Include links or embed content from `YouTube` using the YouTube Data API.
* Enhancement:
  * Display visual aids alongside the textual summary for better engagement.

### 2. Interactive Exploration

* Feature: Allow the user to ask follow-up questions on the same topic.
* How:
  * Save context (Google/Wikipedia results) and enable follow-up queries using the LangChain Memory module.
  * Example: "Tell me more about its applications in developing countries."
* Enhancement:
  * Create an interactive chat-like experience for in-depth exploration.

### 3. Sentiment Analysis and Opinion Trends

* Feature: Analyze public sentiment about the topic based on news or social media data.
* How:
  * Use tools like Twitter API or Google News API.
  * Apply sentiment analysis using TextBlob or OpenAI.
* Enhancement:
  * Provide insights into public opinion on the topic.

### 4. Export and Share

* Feature: Allow users to download the summary or share it on social media.
* How:
  * Use libraries like reportlab for PDF generation.
  * Integrate sharing options with APIs like Twitter or LinkedIn.
* Enhancement:
  * Encourage sharing and keep users engaged with your app.

### 5. Advanced Data Sources
* Feature: Add structured data from advanced APIs like Wolfram Alpha, PubMed, or academic databases.
* How:
  * Integrate the Wolfram Alpha API for mathematical or scientific queries.
  * Use PubMed API for medical research summaries.
* Enhancement:
  * Deliver highly authoritative and domain-specific insights.

### 6. Multi-Language Support
* Feature: Allow the user to choose the language for summaries.
* How:
  * Use OpenAI’s multilingual capabilities or tools like Google Translate API.
* Enhancement:
  * Expand your audience by supporting global languages.

### 7. Topic Recommendations

* Feature: Suggest related topics based on the user’s input.
* How:
  * Use OpenAI to generate related topic ideas.
  * Display clickable recommendations for exploration.
* Enhancement:
  * Keep users engaged and encourage deeper research.

### 8. Personalization

* Feature: Save user preferences for summary length, detail level, or specific tools.
* How:
  * Use Streamlit’s SessionState or a small backend database (e.g., Firebase or SQLite).
* Enhancement:
  * Make the experience tailored to individual needs.

### 9. Deployment to the Cloud

* Feature: Deploy the app for global access.
* How:
  * Use platforms like Streamlit Cloud, Heroku, AWS, or Azure App Service.
* Enhancement:
  * Make the app accessible anywhere, anytime.

### 10. Modular Architecture

* Feature: Make the app extensible by designing it with modular tool integrations.
* How:
  * Abstract tools like Google Search and Wikipedia into reusable modules.
  * Add a settings page for enabling/disabling tools.
* Enhancement:
  * Encourage community contributions and future scalability.

### Example: Adding Multi-Language Support

Here’s how you can add multi-language summaries to the current app:
```python
from langchain.prompts import PromptTemplate
from googletrans import Translator

translator = Translator()

# New prompt for multilingual summaries
multilang_prompt = PromptTemplate(
    input_variables=["topic", "google_results", "wikipedia_results", "language"],
    template=(
        "Summarize the topic '{topic}' using these search results from Google: {google_results}, "
        "and this summary from Wikipedia: {wikipedia_results}. "
        "Translate the summary into {language}."
    )
)

# Chain for multilingual summary
def explore_topic_multilang(topic: str, language: str):
    # Generate a search query
    query = search_query_chain.run(topic)
    
    # Fetch information from Google
    google_results = google_search.run(query)
    
    # Fetch information from Wikipedia
    wikipedia_results = fetch_wikipedia_summary(topic)
    
    # Summarize all the information and translate
    final_summary = summary_chain.run({
        "topic": topic,
        "google_results": google_results,
        "wikipedia_results": wikipedia_results,
        "language": language
    })
    
    return final_summary

# Add language selection in Streamlit
language = st.selectbox("Select the language for your summary:", ["English", "Spanish", "French", "German", "Chinese"])
translated_summary = explore_topic_multilang(topic, language)
st.write(translated_summary)
```

### What’s Next?

I will provide step-by-step instructions and code implementation for featured enhancements later. 

* Feature: [Multi Model Outputs](./AITopicExplorer-MultiModal.md)