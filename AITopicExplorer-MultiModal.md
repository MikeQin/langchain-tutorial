# AI Topic Explorer with Multi-Modal Outputs

Let’s enhance the **AI Topic Explorer** with **Multi-Modal Outputs** by adding `images` and `videos` alongside the `text summary`. This will make the application visually engaging and informative.

### Steps to Implement Multi-Modal Outputs

1. Add Images:
  * Use the Unsplash API to fetch relevant images.
  * Images will complement the textual summary with visual context.
2. Add Videos:
  * Use the YouTube Data API to fetch related videos.
  * Embed YouTube links in the application.
3. Combine Outputs:
  * Present a summary, images, and videos together for a rich experience.

### Updated Code with Multi-Modal Outputs

Install Additional Libraries

```bash
pip install unsplash-py youtube-search-python
```

**Full Code**

```python
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
import wikipediaapi
from unsplash_py import Unsplash
from youtube_search_python import VideosSearch
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

serpapi_api_key = os.getenv("SERPAPI_API_KEY")
unsplash_access_key = os.getenv("UNSPLASH_ACCESS_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = OpenAI(
    model="gpt-4", 
    temperature=0.7, 
    openai_api_key=openai_api_key
)

# Initialize tools
google_search = GoogleSearchAPIWrapper(serpapi_api_key=serpapi_api_key)
unsplash = Unsplash(access_key=unsplash_access_key)

# Wikipedia tool
def fetch_wikipedia_summary(query: str) -> str:
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page = wiki_wiki.page(query)
    if page.exists():
        return page.summary[:1000]
    else:
        return "No Wikipedia page found for this topic."

# YouTube search tool
def fetch_youtube_videos(query: str, max_results: int = 3) -> list:
    videos_search = VideosSearch(query, limit=max_results)
    return videos_search.result()["result"]

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
def explore_topic(topic: str) -> dict:
    # Generate a search query
    query = search_query_chain.run(topic)
    
    # Fetch information from Google
    google_results = google_search.run(query)
    
    # Fetch information from Wikipedia
    wikipedia_results = fetch_wikipedia_summary(topic)
    
    # Summarize all the information
    summary = summary_chain.run({
        "topic": topic,
        "google_results": google_results,
        "wikipedia_results": wikipedia_results
    })
    
    # Fetch images from Unsplash
    images = unsplash.search.photos(topic, per_page=3)
    image_urls = [image["urls"]["regular"] for image in images["results"]]
    
    # Fetch videos from YouTube
    videos = fetch_youtube_videos(topic)
    video_urls = [{"title": video["title"], "link": video["link"]} for video in videos]
    
    return {"summary": summary, "images": image_urls, "videos": video_urls}

# Streamlit app
st.title("AI Topic Explorer")
st.subheader("Explore topics with text, images, and videos!")

# User input
topic = st.text_input("Enter a topic to explore:", "")

if topic:
    with st.spinner("Exploring the topic..."):
        try:
            results = explore_topic(topic)
            st.success("Exploration Complete!")
            
            # Display summary
            st.header("Summary")
            st.write(results["summary"])
            
            # Display images
            st.header("Images")
            for img_url in results["images"]:
                st.image(img_url, use_column_width=True)
            
            # Display videos
            st.header("Videos")
            for video in results["videos"]:
                st.write(f"**{video['title']}**")
                st.video(video["link"])
        except Exception as e:
            st.error(f"An error occurred: {e}")

```

### How the Application Works

1. **Summary:** Generates a detailed summary using Google and Wikipedia.
2. **Images:** Fetches 3 relevant images from Unsplash based on the topic.
3. **Videos:** Fetches 3 related videos from YouTube and embeds them in the app.

### Environment Variables

Ensure you add the following to your .env file:

```bash
SERPAPI_API_KEY=your_serpapi_api_key
UNSPLASH_ACCESS_KEY=your_unsplash_access_key
OPENAI_API_KEY=your_openai_api_key
```

### To Run the Application

1. Save the code to `ai_topic_explorer_multimodal.py`.
2. Run:
```bash
streamlit run ai_topic_explorer_multimodal.py
```

### Output Example

For the topic **"Artificial Intelligence in Medicine"**, the app displays:

* A detailed text summary.
* High-quality images of medical technology.
* Embedded YouTube videos explaining AI applications in medicine.

## Next Steps

* Deploying this app to the cloud (e.g., Streamlit Cloud or Heroku)
* Further extending it (e.g., database integration for storing results)? 😊
* [Vector DB integration](./VectorDB.md)