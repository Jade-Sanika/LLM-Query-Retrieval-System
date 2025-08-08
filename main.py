# main.py

import asyncio
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from models import HackRxRequest, HackRxResponse
import services

# --- AUTHENTICATION & APP INITIALIZATION (same as before) ---
auth_scheme = HTTPBearer()
BEARER_TOKEN = "a504246eea16baacbea3cdca22dc9bd4dd7ecb2d29d8bbda6378eb74008863ec"

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """Validates the bearer token provided in the request header."""
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token"
        )
    return credentials

app = FastAPI(
    title="Intelligent Query-Retrieval System API",
    description="An API for answering questions about documents using a RAG pipeline."
)

# --- NEW: In-Memory Cache ---
# This set will store the URLs of documents we have already processed.
PROCESSED_DOCS_CACHE = set()

# --- API ENDPOINT ---
@app.post("/api/v1/hackrx/run", response_model=HackRxResponse, tags=["HackRx"])
async def run_submission(request: HackRxRequest, _=Depends(validate_token)):
    """
    This endpoint processes a document and answers a list of questions about it.
    It orchestrates the entire RAG pipeline and uses caching for speed.
    """
    try:
        doc_url = str(request.documents)
        
        # --- CACHING LOGIC ---
        # Check if we have already processed this document URL.
        if doc_url not in PROCESSED_DOCS_CACHE:
            print(f"New document URL received. Starting full processing for: {doc_url}")
            # 1. If not, perform the expensive processing steps.
            document_text = services.extract_text_from_pdf_url(doc_url)
            text_chunks = services.chunk_text(document_text)
            services.embed_and_index_chunks(text_chunks, doc_url)
            
            # 2. Add the URL to our cache to remember it for next time.
            PROCESSED_DOCS_CACHE.add(doc_url)
            print(f"Document processing complete. URL added to cache.")
        else:
            print(f"Cached document URL received. Skipping processing for: {doc_url}")

        # --- This part now runs for both cached and new documents ---
        print("Proceeding to Question Answering...")
        async def process_question(question: str):
            context = services.find_relevant_clauses(question, doc_url)
            answer = services.get_answer_from_llm(context, question)
            return answer

        tasks = [process_question(q) for q in request.questions]
        answers = await asyncio.gather(*tasks)
        
        return HackRxResponse(answers=answers)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")