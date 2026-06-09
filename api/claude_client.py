import os
import anthropic
from dotenv import load_dotenv
from .schemas import TransactionInput

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """Eres FraudSense AI, un analista experto en detección de fraude en tarjetas de crédito.
Tu rol es explicar en lenguaje claro y profesional por qué una transacción es sospechosa o segura.
Usás datos reales del modelo Random Forest entrenado sobre 1.85M transacciones (dataset Sparkov).

Contexto del modelo:
- Feature más importante: monto (amt) — 47.6% de importancia
- El monto promedio de fraude es $530 vs $67 en transacciones legítimas (8x más alto)
- Las horas de mayor riesgo son 0hs-3am y 22-23hs
- La categoría de mayor riesgo es shopping_net (online)
- amt_zscore_card mide cuántos desvíos estándar está el monto respecto al historial de esa tarjeta

Cuando expliques una transacción:
1. Indicá los factores de riesgo concretos con números
2. Compará con los patrones del dataset cuando sea relevante
3. Sé directo y útil, sin tecnicismos innecesarios
4. Respondé siempre en español"""


def get_risk_level(transaction: TransactionInput) -> str:
    score = 0
    if transaction.amt > 300:
        score += 2
    if transaction.amt > 500:
        score += 1
    if transaction.hour in range(0, 4) or transaction.hour in range(22, 24):
        score += 2
    if transaction.amt_zscore_card > 2:
        score += 2
    if transaction.amt_zscore_card > 5:
        score += 1
    if "net" in transaction.category.lower():
        score += 1

    if score >= 5:
        return "ALTO"
    elif score >= 3:
        return "MEDIO"
    else:
        return "BAJO"


def explain_transaction(transaction: TransactionInput) -> str:
    risk_level = get_risk_level(transaction)

    prompt = f"""Analizá esta transacción y explicá por qué tiene riesgo {risk_level}:

Datos de la transacción:
- Monto: ${transaction.amt:.2f}
- Categoría: {transaction.category}
- Hora: {transaction.hour}:00hs
- Edad del titular: {transaction.age} años
- Distancia comercio-domicilio: {transaction.distance_km:.1f} km
- Z-score del monto vs historial de la tarjeta: {transaction.amt_zscore_card:.2f}
- Población de la ciudad: {transaction.city_pop:,}

Nivel de riesgo calculado: {risk_level}

Explicá en 3-4 oraciones los factores clave de riesgo o seguridad de esta transacción."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


def ask_question(question: str) -> str:
    prompt = f"""El usuario pregunta sobre el proyecto de detección de fraude:

{question}

Contexto disponible:
- Dataset: 1,852,394 transacciones, 9,651 fraudes (0.52% tasa)
- Modelo: Random Forest con F1=0.88 y AUC-ROC=0.987
- Logistic Regression baseline: F1=0.13, AUC=0.8655
- Features: amt, amt_zscore_card, hour, category_enc, age, city_pop, distance_km, day_of_week, gender_enc
- Hallazgos clave: shopping_net = categoría más riesgosa (1.59%), fraude concentrado en 0-3am, monto fraude 8x mayor
- Técnicas usadas: class_weight='balanced', StandardScaler, train_test_split 80/20

Respondé de forma clara y concisa en español."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text
