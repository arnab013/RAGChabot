import json, re
from typing import List, Dict
from .llm_clients import chat

WHITELIST = [
    "publication_number", "publication_kind", "publication_date", "ipc", "cpc",
    "title_en", "claims", "abstract_text", "description_text", "prior_art",
    "reference", "parent", "pct_publication_number",
    "designated_states_contracting", "designated_states_extension",
    "designated_states_validation", "sdg_number", "analysis_explanation",
    "ipc_tech_field", "ipc_technologies", "applicant_names",
    "applicant_countries", "applicant_count", "inventor_names",
    "inventor_countries", "inventor_count", "parent_publication_number"
]

META_PROMPT = f"""
You are an expert at translating patent-search questions into a STRICT JSON spec.

Return JSON with keys:
  rewritten_query : string
  column_priority : ordered subset of columns from this whitelist:
    {', '.join(WHITELIST)}
  filters         : list of {{column, op, value}} filters
  aggregation     : OPTIONAL dict with keys
                    - group_by: column to aggregate on
                    - top_k: integer

Examples:

1) “Show me SDG 6 patents about membrane desalination.”
{{
  "rewritten_query":"membrane desalination",
  "column_priority":["sdg_number","ipc_technologies","abstract_text","title_en"],
  "filters":[
    {{"column":"sdg_number","op":"eq","value":6}},
    {{"column":"ipc_technologies","op":"contains","value":"membrane"}},
    {{"column":"abstract_text","op":"contains","value":"desalination"}}
  ]
}}

2) “Show me SDG 6 patents technologies”
{{
  "rewritten_query":"technology themes in SDG 6 patents",
  "column_priority":["sdg_number","ipc_technologies","analysis_explanation"],
  "filters":[{{"column":"sdg_number","op":"eq","value":6}}],
  "aggregation":{{"group_by":"ipc_technologies","top_k":10}}
}}

3) “List SDG 13 filings on direct-air CO₂ capture”
{{
  "rewritten_query":"direct air capture patents",
  "column_priority":["sdg_number","abstract_text","analysis_explanation"],
  "filters":[
    {{"column":"sdg_number","op":"eq","value":13}},
    {{"column":"abstract_text","op":"contains","value":"direct air capture"}}
  ]
}}

4) “Give me SDG 9 robotic-manufacturing patents published between 2022 and 2024”
{{
  "rewritten_query":"robotic manufacturing patents",
  "column_priority":["sdg_number","publication_date","abstract_text"],
  "filters":[
    {{"column":"sdg_number","op":"eq","value":9}},
    {{"column":"publication_date","op":"between","value":["2022-01-01","2024-12-31"]}},
    {{"column":"abstract_text","op":"contains","value":"robotic"}},
    {{"column":"abstract_text","op":"contains","value":"manufacturing"}}
  ]
}}

(…and so on for family lookups, prior-art, claim summaries, status checks…)

Supported ops: eq, neq, contains, startswith, in, gte, lte, between.
ONLY output the JSON—no extra commentary.
"""

def rewrite(chat_hist: List[Dict[str, str]], user_msg: str) -> Dict:
    messages = [{"role":"system","content":META_PROMPT}] \
             + chat_hist[-10:] \
             + [{"role":"user","content":user_msg}]
    raw = chat(messages, temperature=0.0, max_tokens=512)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"rewritten_query":user_msg,"column_priority":[],"filters":[]}
    try:
        data = json.loads(m.group(0))
        # sanitize
        data["column_priority"] = [c for c in data.get("column_priority",[]) if c in WHITELIST]
        data["filters"] = [f for f in data.get("filters",[]) if f.get("column") in WHITELIST]
        return data
    except json.JSONDecodeError:
        return {"rewritten_query":user_msg,"column_priority":[],"filters":[]}
