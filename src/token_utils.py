import re
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")  # reasonably close for Mixtral
def count_tokens(text: str) -> int:
    return len(_enc.encode(text))

def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))
