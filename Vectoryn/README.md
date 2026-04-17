#Real-time intelligence engine for structured and unstructured data.

# Enterprise Distributed RAG (Kafka + Qdrant + Llama-3)


## Arquitectura del Proyecto
Este proyecto es un sistema de Búsqueda Vectorial (RAG) diseñado con estándares de Big Tech (Event-Driven Architecture) para soportar alto tráfico y garantizar escalabilidad asíncrona real.

### Stack Tecnológico Principal:
1. **Infraestructura de Datos**: Apache Kafka (Redpanda) para encolamiento asíncrono y Qdrant Vector DB para búsquedas matemáticas en O(log N).
2. **Microservicios API**: FastAPI (Python) ejecutando Ingesta y Recuperación (Retrieval) en puertos aislados.
3. **Cerebro AI**: Llama-3 (Groq API) y Sentence-Transformers Locales para Embedding.
4. **Cliente Interfaz**: Vanilla JS + CSS3 Glassmorphism conectando por Streaming Nativo (Server-Sent Events).

## Diagrama del Flujo de Datos
1. **Ingesta**: [Frontend] -> POST (JSON) -> [FastAPI 8000] -> [Kafka Topic `raw-documents`]
2. **Procesamiento**: [Kafka] -> [Worker Python Background] -> HuggingFace Embedding Local -> [Qdrant HNSW Index]
3. **Consulta**: [Frontend] -> POST -> [FastAPI 8001] -> Búsqueda Coseno en Qdrant -> Inyección de Prompt a Groq -> [SSE Stream Frontend]

## Cómo correrlo localmente
```bash
# 1. Levantar Clústeres
docker compose up -d

# 2. Iniciar Worker Asíncrono
source venv/bin/activate
python workers/embedder.py

# 3. Iniciar Puerta de Ingesta
uvicorn ingestion.main:app --port 8000

# 4. Iniciar Puerta de Consulta (Search)
uvicorn search.api:app --port 8001
```
