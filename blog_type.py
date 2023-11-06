# Imports from the typing module
from typing import List, Tuple

# Imports for spaCy, random, and CSV
import spacy
import random
import csv

# Imports for the BERT model
from transformers import AutoTokenizer, AutoModel

# Imports for cosine similarity and asynchronous programming
from sklearn.metrics.pairwise import cosine_similarity
from asyncio import gather

# Imports for pickling, FAISS, and NumPy
import pickle
import faiss
import numpy as np

# Load blog types from a CSV file
blog_types = []

try:
    with open("blog_types.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            blog_types.append(row["Type"])

except Exception as e:
    print(f"Error loading blog types: {e}")

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Add a lemmatizer to the spaCy pipeline
nlp.add_pipe("lemmatizer")

# Get the list of stop words from spaCy
stop_words = nlp.Defaults.stop_words

# Load the BERT model
bert_model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
model = AutoModel.from_pretrained(bert_model_name)

# Cache the BERT embeddings of all blog titles
all_titles = []
all_embeddings = []

# ... (code to cache the BERT embeddings)

# Create a FAISS index to search for similar blog titles
def get_faiss_index() -> faiss.IndexFlatIP:
    """Creates a FAISS index to search for similar blog titles.

    Returns:
        A FAISS index.
    """

    index = faiss.IndexFlatIP(768)
    index.add(all_embeddings)
    return index

# Function to score a blog title against a document
async def score_title(doc: str, title: str) -> Tuple[float, str]:
    """Scores a blog title against a given document.

    Args:
        doc: The document to score the title against.
        title: The blog title to score.

    Returns:
        A tuple containing the similarity score and the blog title.
    """

    # Encode the document and title into BERT embeddings.
    doc_embedding = model(tokenizer(doc, return_tensors="pt").input_ids).last_hidden_state
    title_embedding = model(tokenizer(title, return_tensors="pt").input_ids).last_hidden_state

    # Calculate the cosine similarity between the two embeddings.
    score = cosine_similarity(doc_embedding.cpu().numpy(), title_embedding.cpu().numpy())[0]

    return score, title

# Function to search for the most similar blog titles to a keyword
def search_titles(keyword: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """Searches for the most similar blog titles to a given keyword.

    Args:
        keyword: The keyword to search for.
        top_k: The number of results to return.

    Returns:
        A list of tuples containing the blog title and the similarity score.
    """

    index = get_faiss_index()

    # Encode the keyword into a BERT embedding.
    keyword_embedding = model(tokenizer(keyword, return_tensors="pt").input_ids).last_hidden_state

    # Search for the most similar blog titles in the FAISS index.
    distances, indices = index.search(keyword_embedding.cpu().numpy(), top_k)

    # Get the blog titles and similarity scores from the FAISS results.
    results = [(blog_types[i], distances[0][i]) for i in indices[0]]

    return results

if __name__ == "__main__":

    # Search for the most similar blog titles to the keyword "machine learning"
    results = search_titles("machine learning")

    # If no matches are found, pick a random blog type.
    if not results:
        print("No matches, picking random type")
        blog_type = random.choice(blog_types)

    else:
        blog_type = results[0][0]

    # Export the result to a CSV file.
    with open("blog_type_picked.csv", "w") as f:
        writer = csv.writer(f)