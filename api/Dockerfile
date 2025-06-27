FROM python:3.9-slim

WORKDIR /app

COPY api.py api.py

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]