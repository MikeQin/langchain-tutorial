# AI Topic Explorer App (with Vector Database)

```bash
# Create a virutal environment
python -m venv .venv

# Activate in Linux
source .venv/bin/activate

# Activate in Git Bash for Windows
source .venv/Scripts/activate

# Verify venv
which python

# Use requirements.txt
python -m pip install -r requirements.txt

# Upgrade pip
python.exe -m pip install --upgrade pip

# Example of manual installation
python -m pip install streamlit langchain openai python-dotenv google-search-results faiss-cpu requests wikipedia-api youtube-search-python

# Test the Application
streamlit run ai_topic_explorer_vectordb.py
```

## Adding Vector Database to Save API Call Tokens

Adding a vector database integration can help you efficiently save and retrieve cached responses for similar queries, saving API call tokens. We’ll integrate the FAISS library for vector similarity search and store query embeddings alongside their results.

### Install Required Libraries

Run the following command to install additional dependencies:
```bash
pip install faiss-cpu langchain
```

**Modified Code with Vector Database Integration**

```python
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
import wikipedia
import requests
import os
import pickle

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

# Initialize embeddings and FAISS vector store
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load or initialize FAISS index
try:
    with open("faiss_index.pkl", "rb") as f:
        vectorstore = pickle.load(f)
except FileNotFoundError:
    vectorstore = FAISS(embedding_model.embed_query, faiss.IndexFlatL2(1536))

# Wikipedia tool
def fetch_wikipedia_summary(query: str) -> str:
    wiki_wiki = wikipedia.Wikipedia('en')
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
    # Check vector database for similar queries
    query_embedding = embedding_model.embed_query(topic)
    search_results = vectorstore.similarity_search_with_score(topic, k=1)
    if search_results and search_results[0][1] < 0.5:  # Similarity threshold
        cached_result = search_results[0][0]
        st.info("Using cached result!")
        return cached_result
    
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
    
    # Save result in vector database
    result = {"summary": summary, "images": images, "videos": video_urls}
    vectorstore.add_texts([topic], [result])
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
    
    return result

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
```

### How This Works

1. FAISS Integration:

  * Queries are converted into embeddings using OpenAIEmbeddings.
  * FAISS is used to check for similar queries stored in a vector database.

2. Caching Results:

  * If a similar query exists in the database (similarity score < 0.5), cached results are returned, saving token expenses.
  * New results are saved in the vector database for future queries.

3. Persistent Storage:

  * The FAISS index is saved to faiss_index.pkl for reuse across sessions.

### Correct Usage
`query_embedding` should be used to perform a similarity search in the vector database. For example, if you're using `FAISS` or `Pinecone`, you'd search the database for embeddings similar to the `query_embedding` and retrieve the closest matches.

Here’s how you might integrate it into your workflow:
```python
def explore_topic(topic: str) -> dict:
    # Generate an embedding for the query
    query_embedding = vectorstore.embed_query(topic)
    
    # Search for similar topics in the vector database
    similar_results = vectorstore.similarity_search_by_vector(query_embedding, k=1)
    
    if similar_results:
        # Use the cached response if a similar query is found
        cached_response = similar_results[0].metadata.get("response")
        if cached_response:
            return cached_response  # Return the cached response
    
    # Proceed with regular processing if no similar query is found
    query = search_query_chain.run(topic)
    google_results = google_search.run(query)
    wikipedia_results = fetch_wikipedia_summary(topic)
    summary = summary_chain.run({
        "topic": topic,
        "google_results": google_results,
        "wikipedia_results": wikipedia_results
    })
    images = fetch_unsplash_images(topic, per_page=3)
    videos = fetch_youtube_videos(topic)
    video_urls = [{"title": video["title"], "link": video["link"]} for video in videos]
    
    # Save the new response in the vector database
    vectorstore.add_texts(
        texts=[topic],
        metadatas=[{"response": {"summary": summary, "images": images, "videos": video_urls}}],
        embeddings=[query_embedding]
    )
    
    return {"summary": summary, "images": images, "videos": video_urls}
```

### Explanation of the Code

1. Generate the Query Embedding:
  * The embedding is created for the input query using the vector database's embedding model.

2. Perform Similarity Search:
  * Searches the vector database for embeddings that are close to the current query embedding.

3. Use Cached Response:
  * If a similar embedding is found and has a cached response, return it to save API tokens.

4. Save New Results:
  * If no similar query exists, compute the results and save them, along with the embedding, in the vector database for future use.

### Why query_embedding Was Unused

In the current state, your code only computes the embedding but does not leverage it for caching or searching. By using it in a similarity search like above, you can ensure token savings by avoiding redundant API calls.

### Improved version of the explore_topic function is better 

Yes, the improved version of the explore_topic function is better because it leverages a vector database to save and retrieve cached responses. This approach:

1. Saves Tokens and API Costs:
  * By checking for similar embeddings in the database before making API calls, you avoid redundant requests to OpenAI and other APIs.

2. Enhances Efficiency:
  * Reusing previously computed responses for similar queries reduces processing time for frequent or related queries.

3. Provides a Scalable Solution:
  * As the database grows, the system becomes more efficient at answering recurring or overlapping queries.

Here’s a detailed look at why the improvements make the function better:

#### Improvements Over the Original Function
--------------------------------------------------------------------   
Feature	            Original explore_topic	  Improved explore_topic
--------------------------------------------------------------------      
Query Optimization	Always generates queries	Checks if a similar query exists before processing.
Token Savings	      No token caching	        Avoids token usage by leveraging cached results.
API Efficiency	    Always fetches new data	  Only fetches data if not found in cache.
Caching	            No caching of results	    Caches results in a vector database.

### Why This Version Is Better

1. Intelligent Caching:

  * If a similar topic exists, the system skips all API calls and returns the cached result, saving both time and cost.

2. Dynamic Learning:

  * As new queries are processed, the system "learns" and grows its cache, improving performance for related topics in the future.

3. Flexible Scalability:

  * You can use this approach with scalable vector database systems like Pinecone, FAISS, or Weaviate, ensuring it can handle large datasets efficiently.

4. Better User Experience:

  * Faster responses for cached queries improve the overall responsiveness of your application.

### The Key Differences `similarity_search_with_score` and `similarity_search_by_vector`

The key differences between `vectorstore.similarity_search_with_score(topic, k=1)` and `vectorstore.similarity_search_by_vector(query_embedding, k=1)` lie in their input, use case, and mechanism:

#### 1. Input Type
* `similarity_search_with_score`(topic, k=1):
  * Takes text input (topic).
  * Internally, the vector database generates an embedding for the input text using a predefined embedding function.
* `similarity_search_by_vector`(query_embedding, k=1):
  * Takes a precomputed embedding vector (query_embedding).
  * This assumes the embedding for the input query has already been generated externally.

#### 2. Use Case

1. `similarity_search_with_score(topic, k=1)`:
  * Best when you have raw text (topic) as input and you want the vector database to handle embedding generation for you.
  * Suitable for simpler workflows where embedding generation is tightly coupled with the vector database.

2. `similarity_search_by_vector(query_embedding, k=1)`:
  * Useful when you want manual control over how embeddings are generated (e.g., using a custom model or function).
  * Necessary when embeddings are generated by a source external to the vector database, such as OpenAI, Hugging Face, or another LLM.

#### 3. Mechanism
* `similarity_search_with_score(topic, k=1)`:
  * The text query is first embedded within the vector database using its built-in embedding function.
  * The resulting embedding is then compared against the stored vectors to find the most similar entries.

* `similarity_search_by_vector(query_embedding, k=1)`:
  * Directly compares the given embedding (query_embedding) with the stored embeddings.
  * No text-to-embedding transformation occurs within the vector database.

#### 4. Performance and Efficiency

* `similarity_search_with_score(topic, k=1)`:
  * Slightly slower for each query, as it performs on-the-fly embedding generation for the input text before searching.
* `similarity_search_by_vector(query_embedding, k=1)`:
  * Faster, as it skips the embedding step and directly searches using the provided embedding.

#### Example Scenarios

**Scenario 1:** Using the Vector Database's Built-in Embedding Function
If you want to rely on the vector database’s internal embedding model:

```python
search_results = vectorstore.similarity_search_with_score("What is AI?", k=1)
```

**Scenario 2:** Using an External Embedding Generator
If you generate embeddings using OpenAI or Hugging Face:

```python
query_embedding = vectorstore.embed_query("What is AI?")  # Generate embedding
search_results = vectorstore.similarity_search_by_vector(query_embedding, k=1)
```

#### Choosing Between the Two

1. Use `similarity_search_with_score(topic, k=1)`:

  * When you want simplicity and are okay with the database handling embeddings.
  * If you're exclusively using the vector database's embedding function.

2. Use `similarity_search_by_vector(query_embedding, k=1)`:

  * When you want flexibility and control over embedding generation.
  * If you’re integrating custom embedding models or external APIs like OpenAI.

Both methods work similarly in returning results, but **embedding control** is the primary deciding factor. Let me know which one fits your workflow better!

### Running the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

You’re all set! 😊

### Next Steps

* Take a look at the [code](./ai_topic_explorer_vectordb.py)
* [Final Code Check](./FinalCodeCheck.md)