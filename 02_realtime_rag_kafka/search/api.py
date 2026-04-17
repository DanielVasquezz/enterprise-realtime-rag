from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
import os

app = FastAPI(title="Retrieval API (RAG Vector Search + LLM)")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 1. Base de Datos Cruda
qdrant = QdrantClient(host="localhost", port=6333)

load_dotenv()

# 2. El LLM (Llama-3 alojado en la granja de servidores de Groq)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))  

print("Cargando Motor Semántico Local para Inferencia Rápidas...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_query_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1)
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).squeeze().tolist()

class QueryPayload(BaseModel):
    query: str
    top_k: int = 3

from fastapi.responses import StreamingResponse

@app.post("/search")
async def retrieve_knowledge(query: QueryPayload):
    try:
        # FASE 1: (Retrieval) - Vectores en milisegundos
        query_vector = get_query_embedding(query.query)
        
        search_result = qdrant.search(
            collection_name="financial_documents",
            query_vector=query_vector,
            limit=query.top_k
        )
        
        context_texts = [hit.payload.get("content", "") for hit in search_result]
        context_unido = "\n\n---\n\n".join(context_texts)
        
        prompt_maestro = f"""
        Eres un Arquitecto de Inteligencia Artificial Financiero. Tu trabajo es responder a la PREGUNTA DEL USUARIO, utilizando ÚNICAMENTE la siguiente información de CONTEXTO que hemos extraído de nuestra base de datos confidencial.
        Si la respuesta a la pregunta no está en este contexto, debes decir estrictamente: "No tengo suficiente información almacenada para responder esto." No inventes información.
        Habla como un experto, conciso y directo al grano.
        
        CONTEXTO DE LA BASE DE DATOS:
        {context_unido}
        
        PREGUNTA DEL USUARIO: 
        {query.query}
        """
        
        # FASE 2: Generation con bandera en TRUE para Streaming Nativo (SSE)
        chat_completion_stream = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Eres un experto analista riguroso."},
                {"role": "user", "content": prompt_maestro}
            ],
            model="llama-3.1-8b-instant", 
            temperature=0.2,
            stream=True # ESTO ES EL SECRETO MILITAR PARA EL STREAMING
        )
        
        # Un Yield Generator en Python es como un loop asíncrono que no bloquea la RAM
        def event_stream():
            for chunk in chat_completion_stream:
                if chunk.choices[0].delta.content is not None:
                    # Empujamos la letra al cable tan rápido como Llama-3 la crea
                    yield chunk.choices[0].delta.content
                    
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
