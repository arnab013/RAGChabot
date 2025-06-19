import requests, time, json
try:
    from config import GOOGLE_API_KEY, GOOGLE_ENDPOINT, GOOGLE_MODEL
except ImportError:
    from .config import GOOGLE_API_KEY, GOOGLE_ENDPOINT, GOOGLE_MODEL

def chat(messages, **gen_params):
    """
    Chat function using Google Gemini API.
    messages: list[dict] → [{"role":"user", "content":"..."}]
    gen_params: temperature, max_tokens, top_p, etc.
    """
    return chat_gemini(messages, **gen_params)

def chat_gemini(messages, model=GOOGLE_MODEL, **gen_params):
    """
    Chat with Google Gemini API.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is required for Gemini")
    
    # Convert messages to Gemini format
    gemini_contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })
    
    # Set up generation config with increased token limit for comprehensive responses
    generation_config = {
        "temperature": gen_params.get("temperature", 0.7),
        "maxOutputTokens": gen_params.get("max_tokens", 12384),  # Increased from 8192 to 16384
        "topP": gen_params.get("top_p", 0.95)
    }
    
    payload = {
        "contents": gemini_contents,
        "generationConfig": generation_config
    }
    
    url = f"{GOOGLE_ENDPOINT}?key={GOOGLE_API_KEY}"
    
    while True:
        resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
        
        if resp.status_code == 429:       # rate-limit
            retry = int(resp.headers.get("Retry-After", "5"))
            print(f"Rate-limited, sleeping {retry}s")
            time.sleep(retry)
            continue
            
        if resp.status_code != 200:
            print(f"Gemini API error: {resp.status_code}, {resp.text}")
            resp.raise_for_status()
            
        response_data = resp.json()
        
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            candidate = response_data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return candidate["content"]["parts"][0]["text"]
        
        raise ValueError(f"Unexpected Gemini response format: {response_data}")
