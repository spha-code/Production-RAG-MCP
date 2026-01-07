import os
import time
import requests
from dotenv import load_dotenv

# Load environment variables manually as before
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
# The official Router endpoint for chat
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

# Reliable open-source models
PRIMARY_MODEL = "Qwen/Qwen2.5-7B-Instruct"
FALLBACK_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

def call_hf_api(messages: list, model_id: str) -> str:
    """Standard REST call to the HF Router"""
    if not HF_TOKEN:
        return "❌ HF_TOKEN missing from .env"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 800,
        "temperature": 0.7,
        "provider": "auto"  # Crucial for avoiding 'model not supported' errors
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=45)
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content'].strip()
        
        # Handle 503 (Model loading) with a retry
        if response.status_code == 503:
            print(f"⏳ {model_id} is waking up... retrying in 15s")
            time.sleep(15)
            return call_hf_api(messages, model_id)
            
        return f"❌ API Error ({response.status_code}): {response.text}"

    except Exception as e:
        return f"❌ Request failed: {str(e)}"

def ask_local(question: str, chunks: list[str] = None) -> str:
    """Main function to integrate with your RAG logic"""
    if not question: return "Please ask a question."
    
    # Context handling (keeping your limit of 2 chunks)
    context = "\n".join(chunks[:2]) if chunks else "No additional context."
    
    messages = [
        {
            "role": "system", 
            "content": f"You are a helpful assistant. Use this context: {context}"
        },
        {
            "role": "user", 
            "content": question
        }
    ]

    # Try primary, fallback on error
    result = call_hf_api(messages, PRIMARY_MODEL)
    if result.startswith("❌"):
        print(f"⚠️ Primary failed, switching to {FALLBACK_MODEL}")
        result = call_hf_api(messages, FALLBACK_MODEL)
        
    return result

# Compatibility Aliases
ask_gemini = ask_local
LLM_READY = True if HF_TOKEN else False