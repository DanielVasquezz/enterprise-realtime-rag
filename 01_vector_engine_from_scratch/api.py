from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from core import VectorEngine
import json
import redis 
import numpy as np

from worker import celery_app, procesar_ingesta


app = FastAPI(
    title="Custom RAG Vector Engine Async",
    description="Motor de embeddings de Alto Rendimiento asíncrono con Celery y Redis",
    version="2.0.0"
)


# Initialize the engine globally at startup
# WARNING: In a real production system over Kubernetes, this loading phase
# happens once per worker during application boot.
print("Loading Neural Network into memory. Please wait...")
engine = VectorEngine()
print("Model loaded successfully.")


redis_client = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)

class DocumentPayload(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = {}

class QueryPayload(BaseModel):
    query: str
    top_k: int = 3

@app.post("/ingest", status_code=202)
async def ingest_document(doc: DocumentPayload):
    """
    Recibe un documento e inmediatamente delega la creación de chunks y embeddings
    a Celery (Worker de fondo). Retorna un HTTP 202 Inmediato.
    """
    try:
        # Aquí es donde transferimos la carga al Worker usando .delay()
        task = procesar_ingesta.delay(doc.id, doc.content, doc.metadata)
        
        return {
            "status": "accepted",
            "message": "Enviado al worker para ingesta asíncrona.",
            "task_id": task.id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
    
@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Ruta creada para que Front-ends puedan consultar el estado de un job
    """
    task_result = celery_app.AsyncResult(task_id)
    
    return{
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result if
    task_result.status == 'SUCCESS' else None
    }

@app.post("/search")
async def search(query: QueryPayload):
    """
    Revisa nuestra base de datos NoSQL de Redis, recupera los tensores JSON
    y realiza la comparación de similaridad de coseno contra todos.
    """
    try:
        # 1. Convertimos la query del usuario en un tensor vectorial
        query_vector = engine.generate_embedding(query.query)
        results = []
        
        # 2. Obtenemos TODOS los documentos alojados en la memoria de Redis
        keys = redis_client.keys("doc:*")
        
        if not keys:
            raise HTTPException(
                status_code=400, 
                detail="La Base de datos está vacía o el worker no ha terminado aún."
            )
            
        # 3. Extraemos y comparamos iterativamente
        for key in keys:
            raw_data = redis_client.get(key)
            record = json.loads(raw_data)
            
            # Reconstruimos la lista matemática a NumPy array
            record_embedding = np.array(record["embedding"])
            
            # Calculamos Similitud
            similarity = engine.cosine_similarity(query_vector, record_embedding)
            
            results.append({
                "chunk_id": record["chunk_id"],
                "content": record["content"],
                "similarity_score": similarity,
                "doc_id": record["doc_id"],
                "metadata": record["metadata"]
            })
            
        # 4. Ordenamos por el que tenga el puntaje matemático más alto (reverse=True)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "status": "success",
            "query": query.query,
            "results": results[:query.top_k]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # HTTPS Readiness Consideration:
    # To run this in production with HTTPS directly on uvicorn, append: 
    # ssl_keyfile="/path/to/key.pem", ssl_certfile="/path/to/cert.pem"
    # Usually in Big Tech, TLS termination is handled by an Ingress controller (Nginx/Traefik).
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
