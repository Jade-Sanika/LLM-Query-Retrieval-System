# services.py

import os
import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- INITIALIZATION ---

# Load environment variables from .env file
load_dotenv()

# Configure Google Gemini client
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# --- CONSTANTS ---
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-flash-latest"
PINECONE_INDEX_NAME = "hackrx-rag-index-gemini"

# --- CORE FUNCTIONS ---

def setup_pinecone_index():
    """Checks if the Pinecone index exists, and creates it if it doesn't."""
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,  # Dimension for Gemini's embedding-001 model
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index created successfully.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")
    return pc.Index(PINECONE_INDEX_NAME)

# Get the index object
index = setup_pinecone_index()

def extract_text_from_pdf_url(pdf_url: str) -> str:
    """Downloads a PDF from a URL and extracts its text content."""
    print(f"Downloading and extracting text from {pdf_url}...")
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        doc = fitz.open(stream=response.content, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        print("Text extraction complete.")
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        raise

def chunk_text(text: str) -> list[str]:
    """Splits a long text into smaller, manageable chunks."""
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

# --- THIS FUNCTION IS UPDATED FOR MAXIMUM SPEED ---
def embed_and_index_chunks(chunks: list[str], document_url: str):
    """
    Generates embeddings for text chunks in batches and upserts them into Pinecone.
    This is much faster than processing one by one.
    """
    print(f"Embedding and indexing chunks in batches for document: {document_url}...")
    
    # Process chunks in batches of 100 (the max batch size for Gemini's API)
    for i in range(0, len(chunks), 100):
        batch_chunks = chunks[i:i + 100]
        
        try:
            # Create embeddings for the entire batch in one API call
            response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch_chunks,
                task_type="retrieval_document"
            )
            embeddings = response['embedding']

            # Prepare vectors for Pinecone upsert
            vectors_to_upsert = []
            for j, embedding in enumerate(embeddings):
                vectors_to_upsert.append({
                    "id": f"chunk-{i+j}",
                    "values": embedding,
                    "metadata": {"text": batch_chunks[j]}
                })
            
            # Upsert the batch of vectors to Pinecone
            index.upsert(vectors=vectors_to_upsert, namespace=document_url)
            print(f"Upserted batch {i//100 + 1}...")

        except Exception as e:
            print(f"Error processing batch starting at chunk {i}: {e}")
            continue
            
    print("Batch embedding and indexing complete.")


def find_relevant_clauses(query: str, document_url: str) -> str:
    """Finds and returns the most relevant text chunks for a given query using Gemini."""
    print(f"Finding relevant clauses for query: '{query}'...")
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = response['embedding']
    
    query_results = index.query(
        namespace=document_url,
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    
    context = "\n\n---\n\n".join([item['metadata']['text'] for item in query_results['matches']])
    print("Relevant context retrieved.")
    return context

def get_answer_from_llm(context: str, question: str) -> str:
    """Uses Gemini to generate an answer based on the provided context and question."""
    print("Generating answer from Gemini...")
    model = genai.GenerativeModel(LLM_MODEL)
    
    # --- NEW, MORE ROBUST PROMPT ---
    prompt = f"""
    You are a meticulous financial and legal document analyst. Your task is to answer the user's question with extreme precision, based ONLY on the provided context.

    Follow these steps rigorously:
    1.  First, carefully read the entire context to understand the information available.
    2.  Identify the exact sentences, clauses, or data points within the context that directly answer the user's question.
    3.  Synthesize these findings into a clear and concise answer. Do not add any information that is not explicitly stated in the context.
    4.  If, after careful review, you cannot find the answer within the provided context, you MUST respond with the exact phrase: "This information is not available in the provided context." Do not guess or infer.

    CONTEXT:
    ---
    {context}
    ---
    
    QUESTION: {question}
    
    ANSWER:
    """
    
    response = model.generate_content(prompt)
    print("Answer generated.")
    return response.text