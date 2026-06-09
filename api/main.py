from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import TransactionInput, ExplainResponse, AskRequest, AskResponse
from .claude_client import explain_transaction, ask_question, get_risk_level

app = FastAPI(
    title="FraudSense AI",
    description="API de detección y explicación de fraude en tarjetas de crédito",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "FraudSense AI",
        "version": "1.0.0",
        "endpoints": ["/explain", "/ask", "/docs"],
    }


@app.post("/explain", response_model=ExplainResponse)
def explain(transaction: TransactionInput):
    """
    Recibe los datos de una transacción y devuelve una explicación
    en lenguaje natural de por qué es sospechosa o segura.
    """
    try:
        risk_level = get_risk_level(transaction)
        explanation = explain_transaction(transaction)
        return ExplainResponse(
            transaction=transaction,
            risk_level=risk_level,
            explanation=explanation,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Pregunta libre sobre el dataset, el modelo o los patrones de fraude.
    Claude responde con contexto del proyecto.
    """
    try:
        answer = ask_question(request.question)
        return AskResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
