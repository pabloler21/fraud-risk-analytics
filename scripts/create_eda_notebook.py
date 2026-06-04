import json

def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def md(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

cells = [
    md(
        "# Fase 3 — EDA: Fraud Risk Analytics\n"
        "\n"
        "Exploratory Data Analysis sobre el dataset de tarjetas de crédito (Sparkov/Kaggle).  \n"
        "**Base de datos:** PostgreSQL `fraud_db` — 1,852,394 transacciones | 9,651 fraudes (0.52%).\n"
        "\n"
        "Estructura del notebook:\n"
        "1. Imports y conexión a Postgres\n"
        "2. Carga de datos\n"
        "3. Inspección básica\n"
        "4. Feature engineering\n"
        "5. Visualizaciones"
    ),

    md(
        "## 1. Imports y conexión\n"
        "\n"
        "`sqlalchemy` crea un engine que `pd.read_sql()` acepta directamente — "
        "no hace falta manejar cursores con `psycopg2`."
    ),

    code(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sqlalchemy import create_engine\n"
        "\n"
        'engine = create_engine("postgresql://soporte@localhost:5432/fraud_db")\n'
        "\n"
        'sns.set_theme(style="whitegrid")\n'
        "%matplotlib inline\n"
        'print("Imports OK")'
    ),

    md(
        "## 2. Carga de datos\n"
        "\n"
        "Se seleccionan las columnas relevantes desde Postgres.  \n"
        "`parse_dates` convierte las columnas de fecha a `datetime64` automáticamente."
    ),

    code(
        'query = """\n'
        "SELECT row_id, trans_date_trans_time, cc_num, merchant, category,\n"
        "       amt, gender, city, state, city_pop, job, dob,\n"
        "       lat, long, merch_lat, merch_long, is_fraud\n"
        "FROM transactions\n"
        '"""\n'
        "\n"
        'df = pd.read_sql(query, engine, parse_dates=["trans_date_trans_time", "dob"])\n'
        'print(f"Filas: {len(df):,} | Columnas: {df.shape[1]}")'
    ),

    md(
        "## 3. Inspección básica\n"
        "\n"
        "Verificamos tipos de datos, nulos y la distribución de la variable target (`is_fraud`)."
    ),

    code("df.info()"),

    code(
        "# Nulos por columna\n"
        "df.isnull().sum()"
    ),

    code(
        '# Balance del target — clase muy desbalanceada\n'
        'fraud_counts = df["is_fraud"].value_counts()\n'
        'fraud_pct = df["is_fraud"].mean() * 100\n'
        "print(fraud_counts)\n"
        'print(f"\\nTasa de fraude: {fraud_pct:.2f}%")'
    ),

    code(
        "# Estadísticas del monto separadas por clase\n"
        'df.groupby("is_fraud")["amt"].describe().round(2)'
    ),

    md(
        "## 4. Feature Engineering\n"
        "\n"
        "Variables derivadas que no están en los datos crudos pero son señales clásicas de fraude:\n"
        "\n"
        "| Feature | Por qué importa |\n"
        "|---|---|\n"
        "| `hour` | Fraude se concentra 0h–3h |\n"
        "| `day_of_week` | Patrones distintos en fin de semana |\n"
        "| `age` | Perfiles de edad distintos por tipo de fraude |\n"
        "| `distance_km` | Fraude ocurre lejos del domicilio del titular |\n"
        "| `amt_zscore_card` | Monto anómalo relativo al historial de la tarjeta |"
    ),

    code(
        "# Variables temporales\n"
        'df["hour"] = df["trans_date_trans_time"].dt.hour\n'
        'df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek  # 0=lunes\n'
        "\n"
        "# Edad al momento de la transacción\n"
        'df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365\n'
        "\n"
        "# Distancia titular→comercio (aprox. euclidiana en grados × 111 km/grado)\n"
        'df["distance_km"] = np.sqrt(\n'
        '    (df["lat"] - df["merch_lat"]) ** 2 +\n'
        '    (df["long"] - df["merch_long"]) ** 2\n'
        ") * 111\n"
        "\n"
        "# Z-score del monto por tarjeta\n"
        'card_stats = df.groupby("cc_num")["amt"].agg(["mean", "std"]).rename(\n'
        '    columns={"mean": "card_mean", "std": "card_std"}\n'
        ")\n"
        "df = df.join(card_stats, on=\"cc_num\")\n"
        'df["amt_zscore_card"] = (df["amt"] - df["card_mean"]) / df["card_std"].replace(0, np.nan)\n'
        "\n"
        'print("Features creados OK")\n'
        'df[["hour", "day_of_week", "age", "distance_km", "amt_zscore_card"]].describe().round(2)'
    ),

    md("## 5. Visualizaciones\n\n### 5.1 Tasa de fraude por categoría de comercio"),

    code(
        "import os; os.makedirs(\"../outputs\", exist_ok=True)\n"
        "\n"
        "fraud_by_cat = (\n"
        '    df.groupby("category")["is_fraud"]\n'
        '    .agg(fraud_rate="mean", fraud_n="sum", total="count")\n'
        '    .sort_values("fraud_rate", ascending=False)\n'
        ")\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(11, 5))\n"
        'sns.barplot(x=fraud_by_cat.index, y=fraud_by_cat["fraud_rate"] * 100, ax=ax, color="steelblue")\n'
        'ax.axhline(df["is_fraud"].mean() * 100, color="red", linestyle="--", label="Promedio global")\n'
        'ax.set_title("Tasa de fraude por categoría (%)", fontsize=13)\n'
        'ax.set_xlabel("Categoría")\n'
        'ax.set_ylabel("% fraude")\n'
        'ax.tick_params(axis="x", rotation=45)\n'
        "ax.legend()\n"
        "plt.tight_layout()\n"
        'plt.savefig("../outputs/fraud_by_category.png", dpi=150)\n'
        "plt.show()"
    ),

    md("### 5.2 Tasa de fraude por hora del día"),

    code(
        'fraud_by_hour = df.groupby("hour")["is_fraud"].mean() * 100\n'
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        'sns.lineplot(x=fraud_by_hour.index, y=fraud_by_hour.values, marker="o", ax=ax)\n'
        'ax.axhline(df["is_fraud"].mean() * 100, color="red", linestyle="--", label="Promedio global")\n'
        'ax.set_title("Tasa de fraude por hora del día", fontsize=13)\n'
        'ax.set_xlabel("Hora")\n'
        'ax.set_ylabel("% fraude")\n'
        "ax.legend()\n"
        "plt.tight_layout()\n"
        'plt.savefig("../outputs/fraud_by_hour.png", dpi=150)\n'
        "plt.show()"
    ),

    md("### 5.3 Distribución del monto — fraude vs. legítimo"),

    code(
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        'for label, color, name in [(0, "steelblue", "Legítimo"), (1, "crimson", "Fraude")]:\n'
        '    subset = df[df["is_fraud"] == label]["amt"]\n'
        "    sns.kdeplot(subset[subset < 500], label=name, color=color, fill=True, alpha=0.3, ax=ax)\n"
        "\n"
        'ax.set_title("Distribución del monto (amt < $500)", fontsize=13)\n'
        'ax.set_xlabel("Monto ($)")\n'
        "ax.legend()\n"
        "plt.tight_layout()\n"
        'plt.savefig("../outputs/amt_distribution.png", dpi=150)\n'
        "plt.show()"
    ),

    md("### 5.4 Distancia titular→comercio"),

    code(
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        'for label, color, name in [(0, "steelblue", "Legítimo"), (1, "crimson", "Fraude")]:\n'
        '    subset = df[df["is_fraud"] == label]["distance_km"]\n'
        "    sns.kdeplot(subset[subset < 200], label=name, color=color, fill=True, alpha=0.3, ax=ax)\n"
        "\n"
        'ax.set_title("Distancia titular → comercio (km)", fontsize=13)\n'
        'ax.set_xlabel("Distancia (km)")\n'
        "ax.legend()\n"
        "plt.tight_layout()\n"
        'plt.savefig("../outputs/distance_distribution.png", dpi=150)\n'
        "plt.show()"
    ),

    md("### 5.5 Heatmap de correlaciones con `is_fraud`"),

    code(
        'num_cols = ["amt", "hour", "day_of_week", "age", "distance_km", "amt_zscore_card", "city_pop", "is_fraud"]\n'
        "corr = df[num_cols].corr()\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 6))\n"
        'sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)\n'
        'ax.set_title("Correlación entre features numéricos", fontsize=13)\n'
        "plt.tight_layout()\n"
        'plt.savefig("../outputs/correlation_heatmap.png", dpi=150)\n'
        "plt.show()"
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "fraud-intelligence",
            "language": "python",
            "name": "fraud-intelligence"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

path = r"c:\Users\pablo\Desktop\fraud-risk-analytics\notebooks\01_eda.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook creado OK")
