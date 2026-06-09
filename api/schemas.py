from pydantic import BaseModel
from typing import Optional


class TransactionInput(BaseModel):
    amt: float
    category: str
    hour: int
    age: int
    distance_km: float
    amt_zscore_card: float
    gender: str = "M"
    city_pop: int = 50000
    is_fraud: Optional[int] = None


class ExplainResponse(BaseModel):
    transaction: TransactionInput
    risk_level: str
    explanation: str


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
