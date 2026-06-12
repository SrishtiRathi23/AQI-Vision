FROM python:3.10-slim

# System deps needed by lightgbm/xgboost/shap wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Cloud Run sets $PORT; default to 8080 locally.
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
