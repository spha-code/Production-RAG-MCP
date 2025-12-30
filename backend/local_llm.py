# backend/local_llm.py
import os
import time
import requests
import re
from dotenv import load_dotenv

print("Loading Gemini LLM...")

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY missing from .env file")
    print("💡 Add to .env: GEMINI_API_KEY=your_key_here")
    GEMINI_READY = False
else:
    print(f"✅ Gemini API key loaded")
    GEMINI_READY = True

# Available Gemini models
AVAILABLE_MODELS = [
    "models/gemini-2.5-flash",      # Fastest, newest
    "models/gemini-2.0-flash",      # Good alternative
    "models/gemini-2.0-flash-001",  # Stable version
]

def filter_relevant_chunks(chunks: list[str], question: str) -> list[str]:
    """
    Keep only chunks actually relevant to the question
    Returns: Filtered list of chunks (max 2)
    """
    if not chunks or len(chunks) == 0:
        return []
    
    # Clean question
    question_clean = re.sub(r'[^\w\s]', '', question.lower())
    question_words = set(question_clean.split())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                  'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were',
                  'what', 'who', 'where', 'when', 'why', 'how', 'do', 'does',
                  'can', 'could', 'would', 'should', 'will', 'you', 'me', 'my'}
    question_words = question_words - stop_words
    
    # If question is too short, just limit chunks
    if len(question_words) < 2:
        return chunks[:2]
    
    relevant_chunks = []
    
    for chunk in chunks[:4]:  # Check top 4 from ChromaDB
        chunk_clean = re.sub(r'[^\w\s]', '', chunk.lower())
        
        # Calculate match score
        matches = 0
        for word in question_words:
            if word in chunk_clean:
                matches += 1
        
        # Keep if decent match
        if matches >= 1 or (question_words and matches / len(question_words) >= 0.3):
            relevant_chunks.append(chunk)
    
    return relevant_chunks[:2]

def calculate_token_limit(question: str) -> int:
    """
    Determine appropriate token limit based on question type
    """
    question_lower = question.lower()
    
    # Very short questions
    if len(question.split()) <= 3:
        return 500  # "who is X?" - brief answer
    
    # Detailed explanations need more tokens
    detail_keywords = ['explain', 'describe', 'tell me about', 'how does', 'what are']
    if any(keyword in question_lower for keyword in detail_keywords):
        return 1200  # Detailed explanation
    
    # Lists
    list_keywords = ['list', 'examples', 'types of', 'name all']
    if any(keyword in question_lower for keyword in list_keywords):
        return 1000
    
    # Default for medium questions
    return 800

def call_gemini_api(prompt: str, question: str = None, model_name: str = None) -> str:
    """
    Direct REST API call to Gemini with complete responses
    """
    if not GEMINI_API_KEY:
        return "❌ Gemini API key not configured."
    
    model_to_use = model_name or AVAILABLE_MODELS[0]
    
    # Calculate appropriate token limit
    token_limit = calculate_token_limit(question) if question else 800
    
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_to_use}:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": token_limit,  # Dynamic token limit
            "topP": 0.9,
            "topK": 40,
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                
                # Check response completion
                finish_reason = candidate.get("finishReason", "")
                if finish_reason == "MAX_TOKENS":
                    print(f"⚠️ Response may be incomplete (token limit: {token_limit})")
                elif finish_reason == "SAFETY":
                    return "⚠️ Response blocked by safety filters."
                
                # Extract text
                if "content" in candidate and "parts" in candidate["content"]:
                    text = candidate["content"]["parts"][0]["text"]
                    
                    # Check for obvious cut-offs
                    if text.strip().endswith(('.', '!', '?')):
                        return text  # Properly ended
                    elif len(text) > 50:  # Has some content
                        return text + "..."  # Add ellipsis if cut off
                    else:
                        return text
                else:
                    return "Unexpected response format."
            else:
                return "No response generated."
                
        else:
            # Try alternative model if first fails
            if model_name is None and response.status_code >= 400:
                for alt_model in AVAILABLE_MODELS[1:2]:
                    try:
                        alt_url = f"https://generativelanguage.googleapis.com/v1beta/{alt_model}:generateContent?key={GEMINI_API_KEY}"
                        alt_response = requests.post(alt_url, json=payload, timeout=20)
                        if alt_response.status_code == 200:
                            alt_data = alt_response.json()
                            if "candidates" in alt_data and alt_data["candidates"]:
                                text = alt_data["candidates"][0]["content"]["parts"][0]["text"]
                                print(f"✅ Switched to {alt_model}")
                                return text
                    except:
                        continue
            
            error_msg = response.text[:150] if response.text else "No error details"
            return f"❌ API error ({response.status_code}): {error_msg}"
            
    except requests.exceptions.Timeout:
        return "⚠️ Request timeout. Please try again."
    except Exception as e:
        return f"❌ Request failed: {str(e)[:80]}"

def ask_local(question: str, chunks: list[str] = None) -> str:
    """
    Main function - generates complete responses using Gemini API
    """
    if not GEMINI_READY:
        return "❌ Gemini API not configured. Check your .env file."
    
    if not question or not question.strip():
        return "Please ask a question."
    
    start_time = time.time()
    question = question.strip()
    
    # Filter chunks for relevance
    filtered_chunks = []
    if chunks and len(chunks) > 0:
        filtered_chunks = filter_relevant_chunks(chunks, question)
    
    has_relevant_context = len(filtered_chunks) > 0
    
    # Build optimized prompt for complete responses
    if has_relevant_context:
        context = "\n".join(filtered_chunks)[:1000]
        
        prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. FIRST check if the document context contains relevant information
2. If YES, provide a COMPLETE answer based primarily on the context
3. If NO or information is incomplete, supplement with your general knowledge
4. Provide a thorough, detailed answer - do not cut off mid-sentence
5. Structure your answer clearly with complete sentences

COMPLETE ANSWER:"""
        
        mode = "doc+"
    else:
        prompt = f"""USER QUESTION: {question}

INSTRUCTIONS:
Provide a COMPLETE, detailed answer to the question above.
Include relevant details, examples, and context.
Write in full paragraphs with complete sentences.

ANSWER:"""
        mode = "general"
    
    # Call Gemini API
    response = call_gemini_api(prompt, question)
    
    elapsed = time.time() - start_time
    
    # Log performance
    original_count = len(chunks) if chunks else 0
    filtered_count = len(filtered_chunks)
    char_count = len(response) if response else 0
    
    print(f"✅ Gemini ({mode}): {elapsed*1000:.0f}ms | "
          f"Chunks: {original_count}→{filtered_count} | "
          f"Chars: {char_count}")
    
    return response

def ask_gemini(question: str, chunks: list[str] = None) -> str:
    """
    Alias for ask_local - maintained for compatibility
    """
    return ask_local(question, chunks)

# For health check compatibility
LLM = GEMINI_READY

# Test function
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GEMINI LLM TEST SUITE")
    print("="*60)
    
    if not GEMINI_READY:
        print("\n❌ Setup incomplete:")
        print("   1. Get API key: https://makersuite.google.com/app/apikey")
        print("   2. Add to .env: GEMINI_API_KEY=your_key_here")
        exit(1)
    
    print(f"\n✅ Gemini API ready")
    print(f"📋 Available models: {', '.join(AVAILABLE_MODELS)}")
    
    # Test cases with expected complete responses
    test_cases = [
        ("Who is Sou Fujimoto?", [], "short biography"),
        ("Explain quadratic functions", [], "detailed explanation"),
        ("What is AI?", [], "complete definition"),
    ]
    
    all_passed = True
    
    for i, (question, chunks, expected_type) in enumerate(test_cases, 1):
        print(f"\n[{i}] Test: '{question}'")
        print(f"   Expected: {expected_type}")
        
        start = time.time()
        response = ask_local(question, chunks)
        elapsed = (time.time() - start) * 1000
        
        if not response:
            print(f"   ❌ No response")
            all_passed = False
        elif response.startswith("❌") or response.startswith("⚠️"):
            print(f"   ❌ Error: {response}")
            all_passed = False
        else:
            # Check if response seems complete
            ends_properly = response.strip().endswith(('.', '!', '?'))
            has_length = len(response.split()) >= 15  # At least 15 words
            
            if ends_properly and has_length:
                status = "✅ Complete"
            elif has_length:
                status = "⚠️ Possibly cut"
            else:
                status = "❌ Too short"
            
            preview = response[:80] + "..." if len(response) > 80 else response
            print(f"   {status}: {preview}")
            print(f"   ⏱️  {elapsed:.0f}ms | Words: {len(response.split())}")
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed! Gemini is responding with complete answers.")
    else:
        print("❌ Some tests failed.")
    print("="*60)