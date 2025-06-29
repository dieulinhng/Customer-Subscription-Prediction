import pika
import joblib
import pandas as pd
import json
import time

print("🚀 Consumer run.")
max_retries = 10
connection = None
for i in range(max_retries):
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq', port=5672, heartbeat=600))
        print("✅ Successfully connected to RabbitMQ!")
        break
    except Exception as e:
        print(f"❌ Failed to connect to RabbitMQ ({i+1}/{max_retries}): {e}")
        time.sleep(3)
else:
    print("❌ Could not connect to RabbitMQ after retries. Exiting.")
    exit(1)

# Load model và preprocessor
model = joblib.load('model.pkl')
preproc = joblib.load('preproc.pkl')

def callback(ch, method, properties, body):
    print("Message received from queue:", body)
    if isinstance(body, bytes):
        data = json.loads(body.decode('utf-8'))
    else:
        data = json.loads(body)
    feature_order = [
        'Age', 'Gender', 'Category', 'Purchase_Amount_(USD)', 'Size', 'Season', 'Review_Rating'
        , 'Shipping_Type', 'Promo_Code_Used', 'Previous_Purchases', 'Payment_Method'
        , 'Frequency_of_Purchases', 'Region', 'Color_Group'
    ]
    input_dict = {field: data[field] for field in feature_order}
    input_df = pd.DataFrame([input_dict])
    X = preproc.transform(input_df)
    prob = model.predict_proba(X)[0][1]
    label = model.predict(X)[0]
    
    label_map = {0: "Not Subcribed", 1: "Subcribed"}
    label_str = label_map.get(label)
    
    print(f"✅ Probability: {prob:.2f}")
    print(f"✅ Label: {label_str}")
    ch.basic_ack(delivery_tag=method.delivery_tag)

connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq', port=5672))
channel = connection.channel()
channel.queue_declare(queue='predict_queue')

channel.basic_consume(
    queue='predict_queue',
    on_message_callback=callback,
    auto_ack=False
)

print("Waiting for messages 🔍")
channel.start_consuming()