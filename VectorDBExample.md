# Vector DB Example

Here’s a complete working code example for caching OpenAI API responses using a vector database. This implementation integrates FAISS, an open-source vector database, to store and retrieve embeddings locally.

### Complete Working Code

#### Install Required Libraries

```bash
pip install openai faiss-cpu numpy
```

#### Code

```python
import openai
import faiss
import numpy as np
from typing import Optional
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FAISS index
dimension = 1536  # Embedding size for text-embedding-ada-002
index = faiss.IndexFlatL2(dimension)  # L2 distance metric
cached_data = {}  # To store metadata (question-response pairs)

# Helper: Generate embedding
def get_embedding(text: str) -> np.ndarray:
    """Generate embeddings using OpenAI."""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response["data"][0]["embedding"], dtype=np.float32)

# Helper: Store question and response
def store_question_response(question: str, response: str):
    """Store the question and response in the vector database."""
    embedding = get_embedding(question)
    index.add(np.array([embedding]))  # Add to FAISS index
    cached_data[len(cached_data)] = {"question": question, "response": response}  # Add to metadata
    print(f"Stored: {question} -> {response}")

# Helper: Search similar questions
def search_similar_question(question: str, threshold: float = 0.8) -> Optional[dict]:
    """Search for a similar question in the vector database."""
    embedding = np.array([get_embedding(question)])
    distances, indices = index.search(embedding, k=1)  # Find the closest match
    if distances[0][0] < threshold:
        matched_index = indices[0][0]
        return cached_data.get(matched_index)
    return None

# Main: Get response with caching
def get_response_with_cache(question: str) -> str:
    """Fetch a response using the cache or the OpenAI API."""
    # Search for a similar question
    cached_result = search_similar_question(question)
    if cached_result:
        print("Using cached response.")
        return cached_result["response"]
    
    # If no similar question, call OpenAI API
    print("Fetching response from OpenAI...")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": question}]
    )["choices"][0]["message"]["content"]
    
    # Store the question and response
    store_question_response(question, response)
    return response

# Example usage
if __name__ == "__main__":
    # First question
    question1 = "What is the capital of France?"
    response1 = get_response_with_cache(question1)
    print(f"Response: {response1}")
    
    # Similar question
    question2 = "Which city is the capital of France?"
    response2 = get_response_with_cache(question2)
    print(f"Response: {response2}")
```

### How It Works

1. First Question:
  * If a question is new, it generates the response using OpenAI and stores the question, response, and embedding.

2. Similar Question:
  * For subsequent similar questions, it calculates the embedding and searches the FAISS index.
  * If a similar question is found (based on distance threshold), it fetches the cached response instead of making an API call.

### Features

1. Efficient Vector Search:
  * Uses FAISS to quickly find similar questions with low latency.

2. Token Savings:
  * Caches responses to avoid redundant API calls.

3. Customizable Threshold:
  * The threshold parameter controls similarity sensitivity (lower values = stricter match).

4. Extendable:
  * Easily migrate to cloud vector databases (e.g., Pinecone or Weaviate) for large-scale use.

### Next Steps

* Improvement Ideas:
  * Migrate to a cloud vector database for scalability.
  * Add an expiration mechanism for cached responses.
  * Integrate this with a UI like Streamlit for interactive use.
* Need Help?
  * Would you like guidance on deploying this or extending it further? 😊