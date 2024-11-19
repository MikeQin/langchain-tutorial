import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.utilities import GoogleSearchAPIWrapper
from youtubesearchpython import VideosSearch
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import faiss
import wikipediaapi
import requests
import os
import pickle

# Load environment variables
load_dotenv()

serpapi_api_key = os.getenv("SERPAPI_API_KEY")
unsplash_access_key = os.getenv("UNSPLASH_ACCESS_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not serpapi_api_key or not unsplash_access_key or not openai_api_key:
    st.error("Missing API keys. Please check your environment variables.")
    st.stop()

# Initialize LLM
llm = OpenAI(
    model="gpt-4", 
    temperature=0.7, 
    openai_api_key=openai_api_key
)

# Initialize tools
google_search = GoogleSearchAPIWrapper(serpapi_api_key=serpapi_api_key)

# Initialize embeddings and FAISS vector store
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load or initialize FAISS index
try:
    with open("faiss_index.pkl", "rb") as f:
        vectorstore = pickle.load(f)
except FileNotFoundError:
    st.warning("FAISS index not found. Creating a new index...")
    vectorstore = FAISS(embedding_model, faiss.IndexFlatL2(1536))
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()

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

# Fetch Unsplash images using REST API
def fetch_unsplash_images(query: str, per_page: int = 3) -> list:
    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {unsplash_access_key}"}
    params = {"query": query, "per_page": per_page}
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        return [result["urls"]["regular"] for result in data["results"]]
    else:
        return []

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
    # Generate an embedding for the query
    # query_embedding = vectorstore.embed_query(topic)
    query_embedding = embedding_model.embed_query(topic)
    
    # Search for similar topics in the vector database
    similar_results = vectorstore.similarity_search_by_vector(query_embedding, k=1)
    
    if similar_results:
        # Use the cached response if a similar query is found
        cached_response = similar_results[0].metadata.get("response", {})
        if cached_response:
           return cached_response.get("response", {})  # Return the cached response
    
    # Proceed with regular processing if no similar query is found

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
    images = fetch_unsplash_images(topic, per_page=3)

    # Fetch videos from YouTube
    videos = fetch_youtube_videos(topic)
    video_urls = [{"title": video["title"], "link": video["link"]} for video in videos]
    
    # Save the new response in the vector database
    result = {"summary": summary, "images": images, "videos": video_urls}
    vectorstore.add_texts(
        texts=[topic],
        metadatas=[{"response": result}],
        embeddings=[query_embedding]
    )

    with open("faiss_index.pkl", "wb") as f:
      pickle.dump(vectorstore, f)
    
    return result

# def explore_topic(topic: str) -> dict:
#     # Check vector database for similar queries
#     query_embedding = embedding_model.embed_query(topic)
#     search_results = vectorstore.similarity_search_with_score(topic, k=1)
#     if search_results and search_results[0][1] < 0.5:  # Similarity threshold
#         cached_result = search_results[0][0]
#         st.info("Using cached result!")
#         return cached_result
    
#     # Generate a search query
#     query = search_query_chain.run(topic)
    
#     # Fetch information from Google
#     google_results = google_search.run(query)
    
#     # Fetch information from Wikipedia
#     wikipedia_results = fetch_wikipedia_summary(topic)
    
#     # Summarize all the information
#     summary = summary_chain.run({
#         "topic": topic,
#         "google_results": google_results,
#         "wikipedia_results": wikipedia_results
#     })
    
#     # Fetch images from Unsplash
#     images = fetch_unsplash_images(topic, per_page=3)
    
#     # Fetch videos from YouTube
#     videos = fetch_youtube_videos(topic)
#     video_urls = [{"title": video["title"], "link": video["link"]} for video in videos]
    
#     # Save result in vector database
#     result = {"summary": summary, "images": images, "videos": video_urls}
#     vectorstore.add_texts([topic], [result])
#     with open("faiss_index.pkl", "wb") as f:
#         pickle.dump(vectorstore, f)
    
#     return result

# Streamlit app
st.title("AI Topic Explorer with Caching")
st.subheader("Explore topics with text, images, and videos efficiently!")

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
