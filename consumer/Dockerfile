FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY consumer.py .
COPY model.pkl .
COPY preproc.pkl .

CMD ["python", "-u", "consumer.py"]