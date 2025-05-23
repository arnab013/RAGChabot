import requests, time
from .config import MISTRAL_API_KEY, MISTRAL_ENDPOINT, MIXTRAL_MODEL

HEADERS = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json"
}

def chat(messages, model=MIXTRAL_MODEL, **gen_params):
    """
    messages: list[dict] → [{"role":"user", "content":"..."}]
    gen_params: temperature, max_tokens, top_p, etc.
    """
    payload = {"model": model, "messages": messages, **gen_params}
    while True:
        resp = requests.post(MISTRAL_ENDPOINT, headers=HEADERS, json=payload, timeout=60)
        if resp.status_code == 429:       # rate-limit
            retry = int(resp.headers.get("Retry-After", "5"))
            print(f"Rate-limited, sleeping {retry}s")
            time.sleep(retry)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
