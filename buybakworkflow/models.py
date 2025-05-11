# models.py
from pydantic import BaseModel, Field


class ResearchTopic(BaseModel):
    query: str = Field(..., example="example query")
