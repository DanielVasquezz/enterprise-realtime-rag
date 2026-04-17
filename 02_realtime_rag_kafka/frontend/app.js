const ingestForm = document.getElementById('ingestForm');
const chatForm = document.getElementById('chatForm');
const chatBox = document.getElementById('chatBox');
const ingestStatus = document.getElementById('ingestStatus');

ingestForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const docId = document.getElementById('docId').value;
    const content = document.getElementById('docContent').value;
    const btnText = document.querySelector('#ingestBtn .btn-text');
    
    btnText.textContent = "Hackeando Kafka...";
    
    try {
        await fetch('http://localhost:8000/ingest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: docId, content: content })
        });
        
        ingestStatus.textContent = "✅ Documento capturado";
        setTimeout(() => ingestStatus.textContent = "", 3000);
        document.getElementById('docContent').value = "";
    } catch (error) {
        ingestStatus.textContent = "❌ Error de conexión";
    } finally {
        btnText.textContent = "Subir a Kafka";
    }
});

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const queryInput = document.getElementById('queryInput');
    const query = queryInput.value;
    if (!query) return;
    
    queryInput.value = ""; 
    appendMessage(query, 'user');
    const loadingId = appendTypingIndicator();
    
    try {
        // Hacemos el llamado a tu API pidiéndole el recurso nativo (Response)
        const response = await fetch('http://localhost:8001/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, top_k: 2 })
        });
        
        document.getElementById(loadingId).remove();
        
        // Dibujamos el cascarón de la respuesta vacía
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ai-message`;
        msgDiv.innerHTML = `<div class="avatar">AI</div><div class="msg-content"></div>`;
        chatBox.appendChild(msgDiv);
        const textContainer = msgDiv.querySelector('.msg-content');
        
        // ==========================================
        // OPTIMIZACIÓN 3: Server Sent Events Decoder
        // ==========================================
        const reader = response.body.getReader(); // Lee los bytes conformen entran al cable de RED
        const decoder = new TextDecoder('utf-8');
        let accumulatedText = "";
        
        // Loop que escucha el Pipeline Asíncrono real de Groq -> Python -> Fetch
        while (true) {
            const { done, value } = await reader.read();
            if (done) break; // Groq cerró la conexión
            
            // Decodificamos el Byte Array
            const chunk = decoder.decode(value, { stream: true });
            accumulatedText += chunk;
            
            // OPTIMIZACIÓN 1: Usamos innerText (Nativo del DOM en C++) 
            // Esto rinde muchísimo mejor y nos ahorra el replace('<br>') que crasheaba
            // a Google Chrome al hacerlo 100 veces por segundo. 
            textContainer.innerText = accumulatedText;
            chatBox.scrollTop = chatBox.scrollHeight;
            
            // Con esto quitamos el setInterval()! Llama-3 domina la velocidad ahora.
        }
    } catch (error) {
        document.getElementById(loadingId)?.remove();
        appendMessage("Conexión RAG perdida.", 'ai');
    }
});

function appendMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}-message`;
    const isAi = sender === 'ai';
    msgDiv.innerHTML = `<div class="avatar">${isAi ? 'AI' : 'TÚ'}</div><div class="msg-content">${text}</div>`;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msgDiv;
}

function appendTypingIndicator() {
    const id = "loading-" + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ai-message`;
    msgDiv.id = id;
    msgDiv.innerHTML = `<div class="avatar">AI</div><div class="msg-content typing-indicator"><span></span><span></span><span></span></div>`;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return id;
}
