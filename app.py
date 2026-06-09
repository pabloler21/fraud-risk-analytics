import gradio as gr
import httpx

API_URL = "http://127.0.0.1:8000"


def explain_transaction(amt, category, hour, age, distance_km, amt_zscore_card, gender, city_pop):
    payload = {
        "amt": amt,
        "category": category,
        "hour": hour,
        "age": age,
        "distance_km": distance_km,
        "amt_zscore_card": amt_zscore_card,
        "gender": gender,
        "city_pop": int(city_pop),
    }
    try:
        r = httpx.post(f"{API_URL}/explain", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        risk = data["risk_level"]
        explanation = data["explanation"]

        color = {"ALTO": "🔴", "MEDIO": "🟡", "BAJO": "🟢"}.get(risk, "⚪")
        return f"{color} **Riesgo {risk}**\n\n{explanation}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def ask_fraudsense(question):
    if not question.strip():
        return "Escribí una pregunta."
    try:
        r = httpx.post(f"{API_URL}/ask", json={"question": question}, timeout=30)
        r.raise_for_status()
        return r.json()["answer"]
    except Exception as e:
        return f"❌ Error: {str(e)}"


CATEGORIES = [
    "shopping_net", "misc_net", "grocery_pos", "shopping_pos",
    "gas_transport", "entertainment", "food_dining", "health_fitness",
    "personal_care", "home", "kids_pets", "travel",
]

with gr.Blocks(
    title="FraudSense AI",
    theme=gr.themes.Base(
        primary_hue="red",
        neutral_hue="slate",
    ),
    css="""
    .gradio-container { max-width: 900px; margin: auto; }
    h1 { text-align: center; }
    """,
) as demo:

    gr.Markdown("# 🔍 FraudSense AI\n**Detección y explicación de fraude en tarjetas de crédito**")

    with gr.Tabs():

        with gr.Tab("Analizar Transacción"):
            gr.Markdown("Ingresá los datos de la transacción para obtener un análisis de riesgo.")

            with gr.Row():
                with gr.Column():
                    amt = gr.Number(label="Monto ($)", value=523.45)
                    category = gr.Dropdown(label="Categoría", choices=CATEGORIES, value="shopping_net")
                    hour = gr.Slider(label="Hora del día", minimum=0, maximum=23, step=1, value=2)
                    age = gr.Slider(label="Edad del titular", minimum=18, maximum=90, step=1, value=34)

                with gr.Column():
                    distance_km = gr.Number(label="Distancia comercio-domicilio (km)", value=187.3)
                    amt_zscore_card = gr.Number(label="Z-score del monto (vs historial tarjeta)", value=4.8)
                    gender = gr.Radio(label="Género", choices=["M", "F"], value="F")
                    city_pop = gr.Number(label="Población de la ciudad", value=45000)

            btn_explain = gr.Button("Analizar", variant="primary")
            output_explain = gr.Markdown(label="Resultado")

            btn_explain.click(
                fn=explain_transaction,
                inputs=[amt, category, hour, age, distance_km, amt_zscore_card, gender, city_pop],
                outputs=output_explain,
            )

            gr.Examples(
                examples=[
                    [523.45, "shopping_net", 2, 34, 187.3, 4.8, "F", 45000],
                    [12.50, "grocery_pos", 14, 45, 2.1, 0.3, "M", 120000],
                    [890.00, "misc_net", 1, 28, 320.0, 7.2, "M", 8000],
                ],
                inputs=[amt, category, hour, age, distance_km, amt_zscore_card, gender, city_pop],
                label="Ejemplos (alto riesgo / bajo riesgo / muy alto riesgo)",
            )

        with gr.Tab("Preguntar a FraudSense"):
            gr.Markdown("Preguntá cualquier cosa sobre el dataset, el modelo o los patrones de fraude.")

            question = gr.Textbox(
                label="Tu pregunta",
                placeholder="¿Por qué las transacciones online tienen más fraude que las físicas?",
                lines=2,
            )
            btn_ask = gr.Button("Preguntar", variant="primary")
            output_ask = gr.Markdown(label="Respuesta")

            btn_ask.click(fn=ask_fraudsense, inputs=question, outputs=output_ask)

            gr.Examples(
                examples=[
                    ["¿Por qué el Random Forest tiene AUC de 0.987?"],
                    ["¿Qué significa el amt_zscore_card?"],
                    ["¿Por qué hay más fraude entre las 0 y las 3am?"],
                    ["¿Cuál es la diferencia entre precision y recall en este contexto?"],
                ],
                inputs=question,
                label="Preguntas de ejemplo",
            )

    gr.Markdown(
        "---\n*FraudSense AI — Entrenado sobre 1.85M transacciones · Random Forest F1=0.88 · AUC-ROC=0.987*"
    )

if __name__ == "__main__":
    demo.launch()
