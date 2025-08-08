# main.py

import asyncio
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from models import HackRxRequest, HackRxResponse
import services

# --- FASTAPI APP INITIALIZATION ---
app = FastAPI(
    title="Intelligent Query-Retrieval System API",
    description="An API for answering questions about documents using a RAG pipeline."
)

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AUTHENTICATION ---
auth_scheme = HTTPBearer()
BEARER_TOKEN = "a504246eea16baacbea3cdca22dc9bd4dd7ecb2d29d8bbda6378eb74008863ec"

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials

# --- CACHE & PRE-WARMING SETUP ---
PROCESSED_DOCS_CACHE = set()
KNOWN_DOC_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

async def warm_up_cache():
    """The actual pre-warming logic, now in its own async function."""
    print("Background Task: Starting cache pre-warming...")
    if KNOWN_DOC_URL not in PROCESSED_DOCS_CACHE:
        try:
            document_text = services.extract_text_from_pdf_url(KNOWN_DOC_URL)
            text_chunks = services.chunk_text(document_text)
            services.embed_and_index_chunks(text_chunks, KNOWN_DOC_URL)
            PROCESSED_DOCS_CACHE.add(KNOWN_DOC_URL)
            print("Background Task: Cache pre-warming complete.")
        except Exception as e:
            print(f"Background Task Error: {e}")
    else:
        print("Background Task: Known document already in cache.")

@app.on_event("startup")
async def startup_event_handler():
    """
    On startup, immediately create a background task for pre-warming the cache.
    The server does NOT wait for this to finish.
    """
    print("Server startup: Kicking off cache pre-warming in the background.")
    asyncio.create_task(warm_up_cache())

# --- API ENDPOINT ---
@app.post("/api/v1/hackrx/run", response_model=HackRxResponse, tags=["HackRx"])
async def run_submission(request: HackRxRequest, _=Depends(validate_token)):
    # ... (The logic for this function remains exactly the same) ...
    try:
        doc_url = str(request.documents)
        if doc_url not in PROCESSED_DOCS_CACHE:
            # Note: The first request might have to wait if the cache isn't warm yet.
            # Or it might process a new document concurrently.
            print(f"New document URL received. Starting full processing for: {doc_url}")
            document_text = services.extract_text_from_pdf_url(doc_url)
            text_chunks = services.chunk_text(document_text)
            services.embed_and_index_chunks(text_chunks, doc_url)
            PROCESSED_DOCS_CACHE.add(doc_url)
        else:
            print(f"Cached document URL received. Skipping processing.")

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