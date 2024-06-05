
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import google.generativeai as genai


class Query(BaseModel):
    text: str


# Configure Google GenAI
GOOGLE_API_KEY = 'AIzaSyBN6_iW25NEfLLxScsDThFDJm6F3evMPb0'
genai.configure(api_key=GOOGLE_API_KEY)

text_model = genai.GenerativeModel(
    'gemini-pro',
    generation_config=genai.GenerationConfig(
        temperature=0.1,
    ),
)


# Function to refine query
def refine_query(query):
    prompt = f"""Input:
  User query: The causes of global warming and what effect it has on the environment.
  Prompt:

  I. Refine and Understand:

  Correct grammar and spelling: "The causes of global warming and what effects it has on the environment."
  Identify intent: The user is likely interested in articles that discuss the causes of global warming and its impact on the environment.
  Add synonyms and related terms: "global warming", "climate change", "greenhouse gases", "anthropogenic factors"
  II. Expand and Explore:

  Identify synonyms and related terms: "global warming" -> "climate change", "greenhouse gases", "anthropogenic factors"
  Suggest additional search terms based on intent: "environmental impact", "climate change effects", "mitigation strategies" 
  III. Present Options:

  Offer the refined query: "Climate change causes and environmental consequences."
  Suggest alternative queries with broader or narrower focus using identified synonyms and related terms. (e.g., "Anthropogenic factors in climate change" or "Effects of climate change on specific ecosystems")

  <Output>:

  Refined query: "Climate change causes and environmental consequences."
  Suggested alternative queries: "Anthropogenic factors in climate change", "Effects of climate change on specific ecosystems"

  Do the same for the following query: {query}
  Output just the Refined query and Suggested alternative queries. FOLLOW THE OUTPUT FORMAT"""

    response = text_model.generate_content([prompt])
    response.resolve()
    output = response.text.split('\n')
    ref_query = output[0].split('Refined query')[1][2:]
    alt_query = output[1].split(':')[1].strip()  # Extract the alternative queries part
    return ref_query, alt_query


# Load the pre-trained Sentence Transformer model
model = SentenceTransformer("thenlper/gte-base")

# Define data columns and file path
cols = ['id', 'title', 'abstract', 'update_date', 'authors_parsed', 'doi']
file_name = 'arxiv-metadata-oai-snapshot.json'

# Load the DataFrame containing document information
data = []
with open(file_name, encoding='latin-1') as f:
    for line in f:
        doc = json.loads(line)
        lst = [doc['id'], doc['title'], doc['abstract'], doc['update_date'], doc['authors_parsed'], doc['doi']]
        data.append(lst)

df_data = pd.DataFrame(data=data, columns=cols)



# Load the pre-built Annoy index
annoy_model = AnnoyIndex(768, metric='angular')
annoy_model.load('gte-base_annoy_100000_tree_500.ann')  # Replace with your Annoy index filename

# Initialize FastAPI application
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set this to your allowed origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.post("/search")
async def search(query: Query):
    """
    Search for similar documents based on a refined query.
    """

    # Refine the user query
    try:
        ref_query, alt_query = refine_query(query.text)  # Obtain refined and alternative queries
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refining the query: {e}")

    # Generate embedding for the refined query
    query_embed = model.encode(ref_query)

    # Perform semantic search using Annoy
    try:
        similar_item_ids = annoy_model.get_nns_by_vector(query_embed, 10, include_distances=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Annoy search: {e}")

    # Retrieve document information from DataFrame
    query_results = pd.DataFrame({
        'id': df_data.iloc[similar_item_ids[0]]['id'],
        'title': df_data.iloc[similar_item_ids[0]]['title'],
        'abstract': df_data.iloc[similar_item_ids[0]]['abstract'],
        'update_date': df_data.iloc[similar_item_ids[0]]['update_date'],
        'authors_parsed': df_data.iloc[similar_item_ids[0]]['authors_parsed'],
        'doi': df_data.iloc[similar_item_ids[0]]['doi'],
        'distance': similar_item_ids[1]
    })

    # Return search results, refined query, and alternative queries
    return {
        "refined_query": ref_query,
        "suggested_alternative_queries": alt_query,
        "results": query_results.to_dict(orient='records')
    }
