from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from confluent_kafka import Producer
import json
import uuid
app = FastAPI(title="Kafka Ingestion Service(Productor)")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

conf = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'ingestion-fastapi'

}
producer = Producer(conf)

class DocumentPayload(BaseModel):
    id: str = None
    content: str
def delivery_report(err, msg):
    """
    Callback asíncrono. Cuando Kafka recibe exitosamente el mensaje, llama a esta función.
    Si el cluster falla o no hay espacio, 'err' nos avisa.
    Práctica obligatoria de "Zero Data Loss" en el nivel Enterprise.
    """
    if err is not None:
        print(f"❌ FALLO CRÍTICO: No se entregó a Kafka: {err}")
    else:
          print(f"✅ ÉXITO: Documento resguardado en el tópico '{msg.topic()}' [Partición: {msg.partition()}]")
    


@app.post("/ingest")
async def ingest_document(doc: DocumentPayload):
    """
    Recibe un documento y en MILISEGUNDOS lo inyecta a un Tópico de Kafka para que 
    los Embedders (Workers) lo consuman a su propio ritmo.
    """
    if not doc.id:
        doc.id = str(uuid.uuid4())
        
    try:
        # Convertimos el objeto Python a JSON crudo (Bytes)
        # Kafka solo entiende bytes, no objetos Python directamente
        mensaje_json = json.dumps({"doc.id": doc.id, "content": doc.content})
        
        
        producer.produce(
            topic="raw-documents",
            key=doc.id.encode('utf-8'),
            
            value=mensaje_json.encode('utf-8'),
            callback=delivery_report
        )
        
        producer.poll(0)
        
        return {
            "status": "accepted",
            "message": "Documento enviado a la cola de procesamiento en tiempo real en Kafka.",
            "doc_id": doc.id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))