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
        "# Fase 4 — ML Baseline: Detección de Fraude\n"
        "\n"
        "Objetivo: entrenar modelos de clasificación y evaluarlos con métricas apropiadas para clases desbalanceadas.\n"
        "\n"
        "Estructura:\n"
        "1. Imports y carga de datos\n"
        "2. Preparación del dataset (features + encoding + split)\n"
        "3. Baseline — Logistic Regression\n"
        "4. Modelo principal — Random Forest\n"
        "5. Evaluación comparativa\n"
        "6. Feature importance"
    ),

    md(
        "## 1. Imports y carga de datos\n"
        "\n"
        "Misma conexión que el EDA. Se agregan imports de scikit-learn para el pipeline de ML."
    ),

    code(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sqlalchemy import create_engine\n"
        "\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.metrics import (\n"
        "    classification_report,\n"
        "    confusion_matrix,\n"
        "    roc_auc_score,\n"
        "    RocCurveDisplay,\n"
        "    PrecisionRecallDisplay,\n"
        "    ConfusionMatrixDisplay,\n"
        ")\n"
        "\n"
        "engine = create_engine(\"postgresql://soporte:Momochiyo23.@localhost:5432/fraud_db\")\n"
        "sns.set_theme(style=\"whitegrid\")\n"
        "%matplotlib inline\n"
        'print("Imports OK")'
    ),

    code(
        'query = """\n'
        "SELECT trans_date_trans_time, cc_num, category, amt,\n"
        "       gender, city_pop, dob,\n"
        "       lat, lon, merch_lat, merch_long, is_fraud\n"
        "FROM transactions\n"
        '"""\n'
        "df = pd.read_sql(query, engine, parse_dates=[\"trans_date_trans_time\", \"dob\"])\n"
        'print(f"Cargadas {len(df):,} filas")'
    ),

    md(
        "## 2. Preparación del dataset\n"
        "\n"
        "### 2.1 Feature engineering\n"
        "\n"
        "Mismas variables derivadas que el EDA."
    ),

    code(
        "# Temporales\n"
        "df[\"hour\"] = df[\"trans_date_trans_time\"].dt.hour\n"
        "df[\"day_of_week\"] = df[\"trans_date_trans_time\"].dt.dayofweek\n"
        "\n"
        "# Edad\n"
        "df[\"age\"] = (df[\"trans_date_trans_time\"] - df[\"dob\"]).dt.days // 365\n"
        "\n"
        "# Distancia titular→comercio\n"
        "df[\"distance_km\"] = np.sqrt(\n"
        "    (df[\"lat\"] - df[\"merch_lat\"]) ** 2 +\n"
        "    (df[\"lon\"] - df[\"merch_long\"]) ** 2\n"
        ") * 111\n"
        "\n"
        "# Z-score del monto por tarjeta\n"
        "card_stats = df.groupby(\"cc_num\")[\"amt\"].agg([\"mean\", \"std\"]).rename(\n"
        "    columns={\"mean\": \"card_mean\", \"std\": \"card_std\"}\n"
        ")\n"
        "df = df.join(card_stats, on=\"cc_num\")\n"
        "df[\"amt_zscore_card\"] = (df[\"amt\"] - df[\"card_mean\"]) / df[\"card_std\"].replace(0, np.nan)\n"
        "df[\"amt_zscore_card\"] = df[\"amt_zscore_card\"].fillna(0)\n"
        "\n"
        'print("Features OK")'
    ),

    md(
        "### 2.2 Encoding de variables categóricas\n"
        "\n"
        "`category` y `gender` son texto — los modelos solo aceptan números."
    ),

    code(
        "le_cat = LabelEncoder()\n"
        "le_gen = LabelEncoder()\n"
        "\n"
        "df[\"category_enc\"] = le_cat.fit_transform(df[\"category\"])\n"
        "df[\"gender_enc\"] = le_gen.fit_transform(df[\"gender\"])\n"
        "\n"
        "print(\"Categorías:\", dict(zip(le_cat.classes_, le_cat.transform(le_cat.classes_))))\n"
        "print(\"Géneros:\", dict(zip(le_gen.classes_, le_gen.transform(le_gen.classes_))))"
    ),

    md(
        "### 2.3 Selección de features y split train/test\n"
        "\n"
        "80% entrenamiento, 20% test. `stratify=y` garantiza que ambos splits tengan la misma proporción de fraudes."
    ),

    code(
        "FEATURES = [\n"
        "    \"amt\", \"hour\", \"day_of_week\", \"age\",\n"
        "    \"distance_km\", \"amt_zscore_card\",\n"
        "    \"city_pop\", \"category_enc\", \"gender_enc\"\n"
        "]\n"
        "\n"
        "X = df[FEATURES]\n"
        "y = df[\"is_fraud\"]\n"
        "\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42, stratify=y\n"
        ")\n"
        "\n"
        'print(f"Train: {len(X_train):,} filas | Test: {len(X_test):,} filas")\n'
        'print(f"Fraudes en train: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")\n'
        'print(f"Fraudes en test:  {y_test.sum():,} ({y_test.mean()*100:.2f}%)")'
    ),

    md(
        "### 2.4 Escalado de features\n"
        "\n"
        "Logistic Regression requiere features en la misma escala. Random Forest no lo necesita, pero escalar no lo perjudica.\n"
        "\n"
        "**Regla clave:** el scaler se entrena solo con `X_train` — nunca con `X_test`. Si escalás con todo el dataset, estás filtrando información del futuro al modelo (data leakage)."
    ),

    code(
        "scaler = StandardScaler()\n"
        "X_train_sc = scaler.fit_transform(X_train)\n"
        "X_test_sc = scaler.transform(X_test)\n"
        'print("Escalado OK")'
    ),

    md(
        "## 3. Baseline — Logistic Regression\n"
        "\n"
        "`class_weight='balanced'` le indica al modelo que los errores en fraudes valen 191x más que en legítimos."
    ),

    code(
        "lr = LogisticRegression(\n"
        "    class_weight=\"balanced\",\n"
        "    max_iter=1000,\n"
        "    random_state=42\n"
        ")\n"
        "lr.fit(X_train_sc, y_train)\n"
        "\n"
        "y_pred_lr = lr.predict(X_test_sc)\n"
        "y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]\n"
        "\n"
        'print("=== Logistic Regression ===")\n'
        "print(classification_report(y_test, y_pred_lr, target_names=[\"Legítimo\", \"Fraude\"]))\n"
        'print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_lr):.4f}")'
    ),

    md("## 4. Modelo principal — Random Forest"),

    code(
        "rf = RandomForestClassifier(\n"
        "    n_estimators=100,\n"
        "    class_weight=\"balanced\",\n"
        "    random_state=42,\n"
        "    n_jobs=-1\n"
        ")\n"
        "rf.fit(X_train, y_train)\n"
        "\n"
        "y_pred_rf = rf.predict(X_test)\n"
        "y_prob_rf = rf.predict_proba(X_test)[:, 1]\n"
        "\n"
        'print("=== Random Forest ===")\n'
        "print(classification_report(y_test, y_pred_rf, target_names=[\"Legítimo\", \"Fraude\"]))\n"
        'print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_rf):.4f}")'
    ),

    md(
        "## 5. Evaluación comparativa\n"
        "\n"
        "### 5.1 Matrices de confusión"
    ),

    code(
        "import os; os.makedirs(\"../outputs\", exist_ok=True)\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "\n"
        "for ax, y_pred, title in [\n"
        "    (axes[0], y_pred_lr, \"Logistic Regression\"),\n"
        "    (axes[1], y_pred_rf, \"Random Forest\"),\n"
        "]:\n"
        "    ConfusionMatrixDisplay(\n"
        "        confusion_matrix(y_test, y_pred),\n"
        "        display_labels=[\"Legítimo\", \"Fraude\"]\n"
        "    ).plot(ax=ax, colorbar=False, cmap=\"Blues\")\n"
        "    ax.set_title(title)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig(\"../outputs/confusion_matrices.png\", dpi=150)\n"
        "plt.show()"
    ),

    md("### 5.2 Curvas ROC"),

    code(
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "\n"
        "RocCurveDisplay.from_predictions(y_test, y_prob_lr, name=\"Logistic Regression\", ax=ax)\n"
        "RocCurveDisplay.from_predictions(y_test, y_prob_rf, name=\"Random Forest\", ax=ax)\n"
        "\n"
        "ax.plot([0, 1], [0, 1], \"k--\", label=\"Aleatorio\")\n"
        "ax.set_title(\"Curva ROC\", fontsize=13)\n"
        "plt.tight_layout()\n"
        "plt.savefig(\"../outputs/roc_curves.png\", dpi=150)\n"
        "plt.show()"
    ),

    md("### 5.3 Curvas Precision-Recall"),

    code(
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "\n"
        "PrecisionRecallDisplay.from_predictions(y_test, y_prob_lr, name=\"Logistic Regression\", ax=ax)\n"
        "PrecisionRecallDisplay.from_predictions(y_test, y_prob_rf, name=\"Random Forest\", ax=ax)\n"
        "\n"
        "ax.set_title(\"Curva Precision-Recall\", fontsize=13)\n"
        "plt.tight_layout()\n"
        "plt.savefig(\"../outputs/precision_recall_curves.png\", dpi=150)\n"
        "plt.show()"
    ),

    md(
        "## 6. Feature Importance\n"
        "\n"
        "¿Qué variables usa más el Random Forest para tomar decisiones?"
    ),

    code(
        "importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "sns.barplot(x=importances.values, y=importances.index, ax=ax, color=\"steelblue\")\n"
        "ax.set_title(\"Feature Importance — Random Forest\", fontsize=13)\n"
        "ax.set_xlabel(\"Importancia\")\n"
        "plt.tight_layout()\n"
        "plt.savefig(\"../outputs/feature_importance.png\", dpi=150)\n"
        "plt.show()\n"
        "\n"
        "print(importances.round(4))"
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

path = r"c:\Users\pablo\Desktop\fraud-risk-analytics\notebooks\02_modeling.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook creado OK")
