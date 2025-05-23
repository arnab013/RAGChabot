from .llm_clients import chat
from .token_utils import count_tokens

MAX_CTX = 60_000    # safe Mixtral window
CHUNK   = 4_096     # tokens per map chunk


def map_reduce_summarise(query: str, passages: list[str]) -> str:
    """
    Summarise passages only if they exceed MAX_CTX.
    Preserve the leading “[ID] "title" ||” header of each passage.
    """
    # total tokens
    total = sum(count_tokens(p) for p in passages)
    if total < MAX_CTX:
        return "\n\n".join(passages)

    # split headers & bodies
    headers, bodies = [], []
    for p in passages:
        if "||" in p:
            head, body = p.split("||", 1)
        else:
            head, body = p, ""
        headers.append(head.strip())
        bodies.append(body.strip())

    # MAP phase: chunk bodies into ~CHUNK-token pieces
    maps, buf, buf_tok = [], [], 0
    for body in bodies:
        t = count_tokens(body)
        if buf_tok + t > CHUNK:
            maps.append("\n\n".join(buf))
            buf, buf_tok = [], 0
        buf.append(body)
        buf_tok += t
    if buf:
        maps.append("\n\n".join(buf))

    # summarise each map-chunk
    partials = []
    for i, chunk in enumerate(maps):
        prompt = [
            {"role": "system",
             "content": f"Summarise chunk {i+1}/{len(maps)} relevant to: '{query}'."},
            {"role": "user", "content": chunk},
        ]
        partials.append(chat(prompt, temperature=0.2, max_tokens=512))

    # REDUCE phase: combine partial summaries
    final_summary = chat(
        [
            {"role": "system",
             "content": "Combine the following partial summaries into one coherent passage."},
            {"role": "user", "content": "\n\n".join(partials)},
        ],
        temperature=0.2,
        max_tokens=768,
    )

    # reattach headers
    return "\n\n".join(f"{h} || {final_summary}" for h in headers)
