from confluent_kafka import Consumer, KafkaError
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from transformers import AutoTokenizer, AutoModel
import torch
import json
import uuid

print("Iniciando Worker de Arquitectura Distribuida...")

# 1. Conexión a Base de Datos Vectorial de Alto Flujo (Qdrant en Rust)
qdrant = QdrantClient(host="localhost", port=6333)

# Nuestra Inteligencia Artificial escupe vectores de exactamente 384 dimensiones.
# Le ordenamos a la base de datos crear una tabla indexada con algoritmo Coseno.
qdrant.recreate_collection(
    collection_name="financial_documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)
print("Colección HNSW en Qdrant configurada.")

# 2. Inicializar Modelo de Lenguaje (Local y Privado)
print("Cargando Motor Neuronal HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str):
    """ Función interna para volver texto a las 384 dimensiones. """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1)
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).squeeze().tolist()

# 3. Consumidor (Consumer) de Kafka
conf = {
    'bootstrap.servers': 'localhost:9092',
    # Todos los workers que compartan este ID se dividirán el trabajo para escalar.
    # Así Meta procesa millones de fotos: prenden la computadora y solita pide su rebanada de datos a Kafka.
    'group.id': 'embedding-cluster-1', 
    'auto.offset.reset': 'earliest' # Comenzar desde los viejos si se trabó y recién lo enciendes
}
consumer = Consumer(conf)
# Nos suscribimos al "canal de radio" del que es dueño nuestra Ingestion API
consumer.subscribe(["raw-documents"])

print("🟢 Worker en línea escuchando asíncronamente a Kafka...")

try:
    while True: # Un worker real jamás se apaga
        
        # Pide un mensaje a Kafka esperando máximo 1 segundo
        msg = consumer.poll(1.0)
        
        if msg is None: continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF: continue
            else:
                print(f"Error nativo de Kafka: {msg.error()}")
                break
                
        # 4. Desempaquetar el mensaje crudo entregado por el Productor
        raw_val = msg.value().decode('utf-8')
        payload = json.loads(raw_val)
        
        # Obtenemos la data (respetando tu llave doc.id)
        doc_id = payload.get("doc.id") 
        content = payload.get("content")
        print(f"\n[KAFKA STREAM] Recibido Doc: {doc_id} -> Procesando Inteligencia Artificial...")
        
        # 5. Pipeline RAG: Convertimos palabras a números, ¡ahora!
        vector = get_embedding(content)
        
        # 6. Upsert (Guardar) a Qdrant, asociando los metadatos y el Vector
        qdrant.upsert(
            collection_name="financial_documents",
            points=[
                PointStruct(
                    id=str(uuid.uuid4()), # UUID único del bloque matemático
                    vector=vector,
                    payload={"doc_id": doc_id, "content": content}
                )
            ]
        )
        print(f"✅ Vector indexado en Base de Datos para Doc: {doc_id}")

except KeyboardInterrupt:
    print("Apagando Cluster de Gracia...")
finally:
    consumer.close()
