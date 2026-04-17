import json
import redis
from celery import Celery
from core import VectorEngine

# 1. Configuración de Celery
# Celery necesita dos lugares: un "broker" para recibir mensajes y un "backend" para guardar resultados.
# Aquí conectamos Celery a nuestro contenedor de Redis en dos bases de datos diferentes (0 y 1).
celery_app = Celery(
    'vector_worker',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

# 2. Cliente nativo de Redis
# Además de Celery, iniciamos un cliente particular de Redis para guardar nuestra "base de datos" de embeddings (reemplaza la lista en memoria `database = []`).
redis_client = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)



# 3. Carga del Modelo en el Worker
# MUY IMPORTANTE: Cargamos el modelo una única vez cuando el "trabajador" (worker) enciende.
# ¡No debemos cargar el modelo en cada petición!
print("INICIANDO WORKER: Cargando red neuronal en memoria...")
engine = VectorEngine()
print("Modelo cargado exitosamente en este proceso (worker).")

@celery_app.task(bind=True, name="ingest_document_task")
def procesar_ingesta(self, doc_id: str, content: str, metadata: dict):
    """
    Esta es la función que se ejecutará en 'background'.
    Recibe el texto, lo divide, genera los embeddings y lo guarda en Redis.
    """
    try:
        # 1. Crear pedazos (chunks) de texto para no perder contexto
        chunks = engine.create_chunks(content)
        chunk_records = []
        
        # 2. Iterar por cada fragmento
        for idx, chunk in enumerate(chunks):
            # Transformamos el texto humano en un vector dimensional (números)
            embedding = engine.generate_embedding(chunk)
            
            # Formateamos nuestro registro para la base de datos
            record = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{idx}",
                "content": chunk,
                # JSON no entiende de arrays de numpy, hay que convertirlo a lista nativa de Python (.tolist())
                "embedding": embedding.tolist(), 
                "metadata": metadata
            }
            
            # 3. Guardar directamente en Redis (serializado como JSON)
            # Usamos el formato Clave/Valor. La clave será doc:{chunk_id}
            redis_client.set(f"doc:{record['chunk_id']}", json.dumps(record))
            chunk_records.append(record["chunk_id"])
            
        # Lo que devolvemos aquí será capturado por 'backend' de Celery y visto como el "Resultado Final"
        return {
            "status": "success",
            "message": f"Se procesaron {len(chunks)} chunks en background.",
            "chunks_procesados": chunk_records
        }
        
    except Exception as e:
        # En caso de error, el Job fallará y devolverá este mensaje
        return {"status": "error", "message": str(e)}
