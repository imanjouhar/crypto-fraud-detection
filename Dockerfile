FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py visualize.py ./
COPY models/ models/
COPY visualizations/ visualizations/
EXPOSE 5000
ENV API_KEY=aml-secret-key-2024
CMD ["python", "main.py", "api"]
