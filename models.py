# models.py

from pydantic import BaseModel, HttpUrl
from typing import List

class HackRxRequest(BaseModel):
    """Defines the structure of the incoming request payload."""
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    """Defines the structure of the outgoing response payload."""
    answers: List[str]