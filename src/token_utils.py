def count_tokens(text: str) -> int:
    """
    Simple token counter - approximates tokens as word count * 1.3
    This is a rough estimate but sufficient for our use case
    """
    if not text:
        return 0
    # Rough approximation: 1 token ≈ 0.75 words
    words = len(text.split())
    return int(words * 1.3)

def count_words(text: str) -> int:
    """Count words in text"""
    if not text:
        return 0
    return len(text.split())
