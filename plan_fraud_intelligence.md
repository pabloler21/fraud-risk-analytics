# Fraud Detection & Risk Intelligence — Plan de ejecución (Windows)

Proyecto de portfolio nivel junior: **Data Analyst en dominio de fraude + plus de AI Engineering**.
Stack: PostgreSQL · Python (pandas/numpy/matplotlib/seaborn/scikit-learn) · Power BI · FastAPI + Pydantic + Claude API.

Dataset: **Credit Card Transactions Fraud Detection (Sparkov)** — `kartik2112/fraud-detection` en Kaggle.

Tiempo estimado: **~6 días full-time** (ver el cronograma y el uso de Claude Code al final). Marcá cada paso con `[x]` a medida que avanzás.

---

## FASE 0 — Setup del proyecto (Día 1 — primeros 30 min)

Postgres y Power BI ya están instalados. Solo armás la carpeta y el entorno de Python con `uv`.

### Carpeta y VS Code

Creá una carpeta en el escritorio (ej. `fraud-intelligence`) y abrila con VS Code (`File → Open Folder`, o desde la terminal `code .` dentro de la carpeta).

Estructura a crear adentro:

```
fraud-intelligence/
├── data/raw/            # CSVs de Kaggle (NO subir a git)
├── data/processed/
├── notebooks/
├── sql/
├── api/
├── models/
├── powerbi/
├── .env
├── .gitignore
└── README.md
```

### Entorno de Python con uv

En la terminal integrada de VS Code (PowerShell), parado en la carpeta del proyecto:

```powershell
uv init
uv add pandas numpy matplotlib seaborn scikit-learn jupyter `
       psycopg2-binary sqlalchemy joblib `
       fastapi uvicorn anthropic pydantic python-dotenv kaggle gradio
```

`uv init` crea el `pyproject.toml` y `uv add` te arma el `.venv` y registra las dependencias solo. Para correr cosas: `uv run jupyter notebook`, `uv run uvicorn api.main:app --reload`, etc. En VS Code, cuando abras un notebook, elegí el kernel del `.venv` que creó uv.

> Las dependencias de arriba cubren todo el proyecto. Si querés, separá las de la API (`fastapi`, `uvicorn`, `anthropic`, etc.) recién en la Fase 6; da igual agregarlas todas ahora.

### Lo ya instalado: solo tené a mano

- **PostgreSQL**: necesitás la contraseña del usuario `postgres` y el puerto (por defecto `5432`). Si no la recordás, reseteala desde pgAdmin antes de seguir.
- **Power BI Desktop**: nada que preparar todavía, lo usás en la Fase 5.

### Git

`.gitignore` (mínimo): `.venv/`, `data/`, `.env`, `*.pkl`, `__pycache__/`, `.ipynb_checkpoints/`

Inicializá el repo al final: `git init`.

---

## FASE 1 — Conseguir los datos y cargarlos en Postgres (Día 1 — mañana)

### 1.1 Descargar el dataset de Kaggle

1. Creá cuenta en kaggle.com si no tenés.
2. Andá a **Settings → API → Create New Token**. Te baja un `kaggle.json`.
3. Movelo a `C:\Users\<tu_usuario>\.kaggle\kaggle.json`.
4. Descargá y descomprimí:

```powershell
uv run kaggle datasets download -d kartik2112/fraud-detection -p data\raw
Expand-Archive data\raw\fraud-detection.zip -DestinationPath data\raw
```

Quedan dos archivos: `fraudTrain.csv` (~1,3M filas) y `fraudTest.csv` (~556k filas).

### 1.2 Conocer el esquema

Columnas (en orden): índice sin nombre, `trans_date_trans_time`, `cc_num`, `merchant`, `category`, `amt`, `first`, `last`, `gender`, `street`, `city`, `state`, `zip`, `lat`, `long`, `city_pop`, `job`, `dob`, `trans_num`, `unix_time`, `merch_lat`, `merch_long`, `is_fraud`.

Mirada rápida en Python (notebook o consola) para entender qué tenés:

```python
import pandas as pd
df = pd.read_csv("data/raw/fraudTrain.csv")
print(df.shape)
print(df["is_fraud"].value_counts(normalize=True))  # ~0,5% fraude → desbalance real
df.head()
```

### 1.3 Crear la base y la tabla

`sql/01_schema.sql`:

```sql
CREATE DATABASE fraud_db;
-- conectarse a fraud_db y luego:
CREATE TABLE transactions (
    row_id                INTEGER,
    trans_date_trans_time TIMESTAMP,
    cc_num                BIGINT,
    merchant              TEXT,
    category              TEXT,
    amt                   NUMERIC(10,2),
    first_name            TEXT,
    last_name             TEXT,
    gender                CHAR(1),
    street                TEXT,
    city                  TEXT,
    state                 CHAR(2),
    zip                   TEXT,
    lat                   NUMERIC(9,6),
    lon                   NUMERIC(9,6),
    city_pop              INTEGER,
    job                   TEXT,
    dob                   DATE,
    trans_num             TEXT,
    unix_time             BIGINT,
    merch_lat             NUMERIC(9,6),
    merch_long            NUMERIC(9,6),
    is_fraud              SMALLINT
);
```

> El orden de columnas debe coincidir con el CSV. La primera columna sin nombre entra como `row_id`.

### 1.4 Cargar los CSV

Desde `psql` (abrí "SQL Shell (psql)" del menú inicio, o `psql -U postgres -d fraud_db`). Usá `\copy` (cliente, no requiere permisos de servidor):

```sql
\copy transactions FROM 'C:/Users/<tu_usuario>/Desktop/fraud-intelligence/data/raw/fraudTrain.csv' WITH (FORMAT csv, HEADER true);
\copy transactions FROM 'C:/Users/<tu_usuario>/Desktop/fraud-intelligence/data/raw/fraudTest.csv' WITH (FORMAT csv, HEADER true);
```

> Usá barras normales `/` en la ruta aunque estés en Windows. Verificá: `SELECT count(*), sum(is_fraud) FROM transactions;`

---

## FASE 2 — SQL: detección de patrones (Día 1 — tarde)

Acá demostrás la skill core del rol: **encontrar patrones sospechosos con SQL**. Guardá todo en `sql/03_patterns.sql` y dejá comentarios explicando qué busca cada query (eso después va al README).

Queries clave a escribir (te dejo el esqueleto, completalas vos):

**1. Tasa de fraude por categoría** — ¿dónde se concentra el fraude?
```sql
SELECT category,
       count(*) AS total,
       sum(is_fraud) AS fraudes,
       round(100.0 * sum(is_fraud) / count(*), 3) AS tasa_fraude_pct
FROM transactions
GROUP BY category
ORDER BY tasa_fraude_pct DESC;
```

**2. Fraude por hora del día** — los patrones temporales son oro en fraude.
```sql
SELECT extract(hour FROM trans_date_trans_time) AS hora,
       sum(is_fraud) AS fraudes,
       round(100.0 * sum(is_fraud) / count(*), 3) AS tasa_pct
FROM transactions
GROUP BY hora ORDER BY hora;
```

**3. Velocidad de transacciones (window function)** — cuántas operaciones hizo la misma tarjeta en la hora previa. Esta es la query que muestra nivel.
```sql
SELECT trans_num, cc_num, trans_date_trans_time, amt, is_fraud,
       count(*) OVER (
           PARTITION BY cc_num ORDER BY unix_time
           RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
       ) AS txns_ultima_hora
FROM transactions
ORDER BY txns_ultima_hora DESC;
```

**4. Monto vs. promedio histórico de la tarjeta** — detectar gastos anómalos.
```sql
SELECT trans_num, cc_num, amt, is_fraud,
       round(avg(amt) OVER (PARTITION BY cc_num), 2) AS promedio_tarjeta,
       round(amt / nullif(avg(amt) OVER (PARTITION BY cc_num), 0), 1) AS veces_sobre_promedio
FROM transactions
ORDER BY veces_sobre_promedio DESC NULLS LAST;
```

**5. Vista de transacciones sospechosas** — combiná reglas en una `VIEW` que después consume Power BI y el agente de IA.
```sql
CREATE VIEW suspicious_transactions AS
SELECT * FROM ( /* tu query con las señales combinadas */ ) t
WHERE veces_sobre_promedio > 5 OR txns_ultima_hora > 5;
```

Objetivo de la fase: 6-8 queries comentadas + 1 vista. Eso ya es un entregable de portfolio por sí solo.

---

## FASE 3 — EDA en Python (Día 2)

Notebook `notebooks/01_eda.ipynb`. El EDA = "reconocer patrones sutiles en grandes volúmenes", que es literalmente la tarea del analista de fraude.

Pasos:

1. **Cargar y parsear**: leer el CSV, convertir `trans_date_trans_time` y `dob` a fecha.
2. **Feature engineering** (creá estas columnas, las vas a reusar en ML y en el dashboard):
   - `hour` = hora de la transacción
   - `day_of_week`
   - `age` = años entre `dob` y la transacción
   - `distance_km` = distancia entre cliente (`lat`,`long`) y comercio (`merch_lat`,`merch_long`) con fórmula de Haversine
   - `amt_zscore_card` = qué tan lejos está el monto del promedio de esa tarjeta
3. **Visualizaciones** (matplotlib/seaborn) — al menos estas cinco:
   - Barras del desbalance de clases (mostrá que el fraude es ~0,5%)
   - Distribución de montos fraude vs. legítimo (escala log en el eje)
   - Línea de tasa de fraude por hora
   - Barras de tasa de fraude por categoría
   - Heatmap de correlación entre variables numéricas
4. **Conclusiones escritas**: 3-4 bullets de hallazgos en lenguaje de negocio (ej: "el fraude se concentra entre las 22h y 3am y en categorías X e Y"). Esto es lo que más valoran.
5. Guardá el dataset con features en `data/processed/transactions_features.csv`.

---

## FASE 4 — Machine Learning liviano (Día 3 — mañana)

Notebook `notebooks/02_modeling.ipynb`. **No sobrevendas**: un baseline honesto que muestra que entendés el problema.

1. **Split**: usá `fraudTrain.csv` para entrenar y `fraudTest.csv` para evaluar (ya viene separado).
2. **Preprocesamiento**: escalar `amt`, `distance_km`, `age`; codificar `category` (one-hot de las categorías top). Armá un `Pipeline` de sklearn.
3. **Modelos**: arrancá con `LogisticRegression(class_weight="balanced")` y luego `RandomForestClassifier(class_weight="balanced")`. Nada de XGBoost/SMOTE — no hace falta para jr.
4. **Métricas correctas** (esto es lo que te separa del jr promedio): NO uses accuracy. Reportá:
   - Matriz de confusión
   - Precision, Recall, F1
   - ROC-AUC y **PR-AUC** (la curva precision-recall es la que importa con clases desbalanceadas)
   - Graficá la curva precision-recall
5. **Interpretá**: sacá la importancia de features del Random Forest y explicá en negocio qué señales pesan más.
6. **Guardá el modelo**: `joblib.dump(pipeline, "models/fraud_model.pkl")`.

Frase para el README: *"el modelo atrapa el X% del fraude revisando solo el Y% de las transacciones"* — traducí siempre a impacto de negocio.

---

## FASE 5 — Dashboard en Power BI (Día 3 tarde – Día 4)

1. Abrí **Power BI Desktop** → *Get Data* → *PostgreSQL database*. Servidor `localhost`, base `fraud_db`. (Si pide el provider Npgsql, instalalo desde el link que te ofrece Power BI.)
2. Importá la tabla `transactions` y la vista `suspicious_transactions`.
3. **Medidas DAX** a crear:
   - `Tasa de Fraude = DIVIDE(SUM(is_fraud), COUNTROWS(transactions))`
   - `Pérdida por Fraude ($) = CALCULATE(SUM(amt), transactions[is_fraud] = 1)`
   - `Transacciones Marcadas = COUNTROWS(suspicious_transactions)`
4. **Páginas del dashboard**:
   - **Overview**: KPIs arriba (tasa de fraude, pérdida total $, transacciones marcadas), tendencia temporal, mapa por estado.
   - **Patrones**: fraude por categoría, por hora, por monto. Slicers interactivos.
   - **Investigación**: tabla de transacciones sospechosas con *drill-through* al detalle de una transacción.
5. Guardá como `powerbi/fraud_dashboard.pbix` y exportá screenshots a PDF para el portfolio.

---

## FASE 6 — La capa de IA (el plus que sorprende) (Día 5)

Backend FastAPI con dos endpoints. Tu stack conocido: `anthropic` + `pydantic` + `fastapi`. Sin LangGraph.

`.env`: `ANTHROPIC_API_KEY=...` y las credenciales de la base.

### 6.1 Endpoint `/explain` — explica una transacción marcada

El analista manda una transacción y Claude la traduce a una explicación en lenguaje natural, **anclada a los datos reales** (así no alucina).

- Definí un modelo Pydantic `TransactionInput` (monto, hora, categoría, distancia, veces_sobre_promedio, txns_ultima_hora, etc.).
- Definí un modelo Pydantic `FraudExplanation` (campos: `risk_level`, `explanation`, `key_signals: list[str]`).
- En el prompt a Claude: pasá los números reales y pedile que devuelva SOLO JSON con esos campos. Parsealo con Pydantic.
- Usá el modelo Haiku para esto (es barato y rápido, perfecto para explicaciones cortas).

### 6.2 Endpoint `/ask` — NL2SQL

El analista pregunta en español → Claude genera el SQL → lo ejecutás contra Postgres → devolvés filas + un resumen.

- En el system prompt incluí el **esquema de la tabla** (nombres de columnas y tipos).
- **Guardrails obligatorios** (esto es lo que demuestra "IA responsable"):
  - Creá un usuario de Postgres **read-only** para estas queries (`GRANT SELECT`). Nunca el superusuario.
  - Validá en código que el SQL generado empiece con `SELECT` y no contenga `INSERT/UPDATE/DELETE/DROP`.
  - Inyectá un `LIMIT` si no lo tiene.
- Devolvé las filas y, opcional, un resumen en lenguaje natural del resultado.

### 6.3 UI de demo (opcional pero suma)

Armá una interfaz mínima con **Gradio** (ya lo manejás del hackathon): un campo para preguntar en lenguaje natural y un panel que muestre la explicación de una transacción. Con eso tenés algo clickeable para mostrar en la entrevista.

---

## FASE 7 — Portfolio y cierre (Día 6)

1. **README.md** profesional: problema de negocio, arquitectura (diagrama simple dato→SQL→ML→dashboard→IA), screenshots del dashboard, resultados del modelo en lenguaje de negocio, cómo correrlo.
2. **GitHub**: subí todo a tu cuenta (`pabloler21`). Que el `data/` quede ignorado; en el README explicá cómo bajar el dataset de Kaggle.
3. **Deploy del API**: Render o Railway (ya los usaste). Dejá el endpoint vivo para el demo.
4. **Post de LinkedIn**: contá el proyecto enfocando el ángulo híbrido — "analista de fraude + asistente con LLM que explica las decisiones". Ese contraste es tu diferenciador.

---

## Resumen de qué demuestra cada parte (para la entrevista)

| Componente | Skill que valida |
|---|---|
| SQL + window functions | Detección de patrones, consulta de datos |
| EDA en Python | Reconocer patrones sutiles, comunicación visual |
| ML baseline + métricas correctas | Criterio analítico (precision/recall, no accuracy) |
| Dashboard Power BI | Visualización y storytelling de negocio |
| Capa de IA (explainer + NL2SQL + guardrails) | El plus de AI Engineering que ningún jr lleva |

El posicionamiento honesto: proyecto sólido de **data analyst en fraude** (lo que el rol jr realmente pide) con una herramienta de IA encima que resuelve un dolor real del analista. Ese es el "wow".

---

## Cronograma full-time (~6 días)

| Día | Foco |
|---|---|
| **Día 1** | Setup (30 min) + datos en Postgres (mañana) + SQL de patrones (tarde) |
| **Día 2** | EDA completo: features, gráficos, conclusiones de negocio |
| **Día 3** | ML baseline + métricas (mañana) → empezar Power BI (tarde) |
| **Día 4** | Terminar dashboard Power BI (3 páginas + DAX + screenshots) |
| **Día 5** | Capa de IA: FastAPI + endpoints `/explain` y `/ask` + guardrails |
| **Día 6** | UI de demo (Gradio) + README + GitHub + deploy + post de LinkedIn |

Es ajustado pero realista a tiempo completo. Si un día se estira, el colchón natural es recortar la UI de Gradio (Fase 6.3) y el deploy (Fase 7.3), que son opcionales.

---

## Dónde usar Claude Code (y dónde NO)

Claude Code te acelera lo aburrido, pero hay partes que **tenés que escribir y entender vos** porque son exactamente lo que te van a preguntar en la entrevista. Si Claude Code te las escribe y no las entendés, el proyecto se cae cuando te pidan que lo expliques.

**Delegá a Claude Code (boilerplate, velocidad):**
- Script de carga de los CSV a Postgres y el `CREATE TABLE`
- Función de Haversine para `distance_km`
- Scaffolding de FastAPI: estructura de `main.py`, routers, conexión a la base, `.env`, carga de `.gitignore`
- Modelos Pydantic (`TransactionInput`, `FraudExplanation`)
- Boilerplate de matplotlib/seaborn (los ejes, títulos, layout) una vez que vos decidiste QUÉ graficar
- Sintaxis puntual: DAX, parámetros de sklearn, comandos de uv/psql
- Debugging de errores y el README final

**Hacelo vos (es tu skill y lo defendés en la entrevista):**
- La **lógica de las queries SQL** de detección de patrones (Fase 2). Es la skill core del rol; tenés que poder explicar cada window function.
- La **interpretación del EDA** (Fase 3): qué patrón viste y qué significa para el negocio. El gráfico lo puede hacer Claude Code; el insight es tuyo.
- La **lectura de las métricas de ML** (Fase 4): por qué precision/recall y no accuracy, qué significa tu matriz de confusión en plata.
- Los **prompts del explainer y el NL2SQL** (Fase 6). Acá tu experiencia con structured outputs y anti-alucinación es el diferenciador; no lo tercericés.

Regla simple: si es algo que un entrevistador te puede pedir que expliques en vivo, escribilo vos y usá Claude Code para revisarlo, no para generarlo desde cero.
