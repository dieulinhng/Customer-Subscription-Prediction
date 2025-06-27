from fastapi import FastAPI, Request
import pika, json

app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    message = json.dumps(data)
    
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='rabbitmq')
    )
    channel = connection.channel()
    channel.queue_declare(queue='predict_queue')
    channel.basic_publish(exchange='', routing_key='predict_queue', body=message)
    connection.close()

    return {"Sent to queue 🔥"}