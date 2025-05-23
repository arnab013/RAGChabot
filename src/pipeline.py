# src/pipeline.py

import re
from collections import deque
from typing import Any, Dict, List

import pandas as pd

from .query_rewrite    import rewrite
from .retrieval        import PassageRetriever
from .filter_ops       import apply_filter
from .stats_engine     import top_k_group, group_by_year
from .summarise        import map_reduce_summarise
from .llm_clients      import chat
from .token_utils      import count_tokens

MAX_CTX_TOKENS  = 60_000
PROMPT_OVERHEAD = 2_000


class RAGPipeline:
    """Conversation-level orchestrator with special-case branches,
    aggregation, and multi-stage passage-RAG fallbacks."""

    def __init__(self,
                 retriever: PassageRetriever,
                 max_history: int = 5,
                 debug: bool     = False):
        self.retriever        = retriever
        self.chat_history     = deque(maxlen=max_history * 2)
        self.debug            = debug
        self._last_ctx_tokens = 0
        # for “this category” and multi-turn context
        self._last_filters     = []
        self._last_aggregation = None

    def _filter_df(self,
                   df: pd.DataFrame,
                   filters: List[Dict[str, Any]]) -> pd.DataFrame:
        """Return the subset of df passing all filters."""
        if not filters:
            return df
        mask = []
        for _, row in df.iterrows():
            ok = True
            for f in filters:
                ok &= apply_filter(row.get(f["column"], ""), f["op"], f["value"])
            mask.append(ok)
        return df.loc[mask]
    

    def ask(self, user_msg: str) -> str:

        # ─── 0. Innovate-on-patent branch ────────────────────────────────
        m_imp = re.search(
            r"(?:improve|innovate|build on|extend)\s+(?:the\s+)?(?:patent\s+)?(.+?)\s*\(?(\d{4,})\)?",
            user_msg, re.I
        )
        if m_imp:
            patent_id = m_imp.group(2)
            df = self.retriever.df
            row = df[df["publication_number"].astype(str) == patent_id]
            if row.empty:
                return f"Sorry, I don’t have patent {patent_id}."
            # extract context fields
            title    = row.iloc[0]["title_en"] or ""
            abstract = row.iloc[0]["abstract_text"] or ""
            claims   = row.iloc[0]["claims"] or ""
            analysis = row.iloc[0]["analysis_explanation"] or ""
            # build brainstorming prompt
            system = {
                "role": "system",
                "content": (
                    "You are a domain expert in R&D. "
                    "I'll give you a patent; propose at least five concrete "
                    "improvements or spin-off innovations based on its core idea."
                )
            }
            user_ctx = {
                "role": "user",
                "content": (
                    f"Patent {patent_id}: {title}\n\n"
                    f"Abstract:\n{abstract}\n\n"
                    f"Claims:\n{claims}\n\n"
                    f"Inventor's analysis:\n{analysis}\n\n"
                    "Please brainstorm improvements or new applications."
                )
            }
            answer = chat([system, user_ctx], temperature=0.7, max_tokens=512)
            self.chat_history.extend([
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": answer},
            ])
            return answer

        
        # ─── 0. Inherit “this category” filters if referenced
        if "this category" in user_msg.lower():
            filters     = list(self._last_filters)
            aggregation = self._last_aggregation
        else:
            filters     = []
            aggregation = None

        # ─── 1. Rewrite NL → structured spec (once per turn)
        rw = rewrite(list(self.chat_history), user_msg)
        rq           = rw.get("rewritten_query", user_msg)
        # merge inherited + new filters
        for f in rw.get("filters", []):
            if f not in filters:
                filters.append(f)
        col_priority  = rw.get("column_priority", [])
        aggregation   = aggregation or rw.get("aggregation")

        # ─── 2. Force SDG-N filter if mentioned
        m_sdg = re.search(r"\bsdg\s*(\d+)\b", user_msg, re.I)
        if m_sdg:
            sdg_val = int(m_sdg.group(1))
            f_sdg = {"column":"sdg_number","op":"eq","value":sdg_val}
            if not any(f["column"]=="sdg_number" for f in filters):
                filters.insert(0, f_sdg)

        # persist for multi-turn
        self._last_filters     = list(filters)
        self._last_aggregation = aggregation

        # ─── A. Metadata+“what’s new” branch
        if ("inventor" in user_msg.lower()
            and "applicant" in user_msg.lower()
            and "new" in user_msg.lower()):
            # find last patent ID from conversation
            last = next(
                (h["content"] for h in reversed(self.chat_history)
                 if h["role"]=="assistant" and "(" in h["content"]),
                None
            )
            pid_m = re.search(r"\((\d+)\)", last) if last else None
            if pid_m:
                pid = pid_m.group(1)
                df  = self.retriever.df
                row = df[df["publication_number"].astype(str)==pid]
                if not row.empty:
                    inv = row.iloc[0].get("inventor_names") or "not provided"
                    app = row.iloc[0].get("applicant_names") or "not provided"
                    new = (row.iloc[0].get("analysis_explanation")
                           or row.iloc[0].get("abstract_text")
                           or "not provided")
                    answer = (
                        f"({pid}) Inventor(s): {inv}; Applicant(s): {app}.\n"
                        f"New in this invention: {new}"
                    )
                else:
                    answer = "I don’t have enough information on that patent."
                self.chat_history.extend([
                    {"role":"user",      "content":user_msg},
                    {"role":"assistant", "content":answer},
                ])
                return answer

        # ─── B. Inventor-perspective branch
        if re.search(r"\binventor\b", user_msg, re.I) and "this patent" in user_msg.lower():
            last = next(
                (h["content"] for h in reversed(self.chat_history)
                 if h["role"]=="assistant" and "(" in h["content"]),
                None
            )
            pid_m = re.search(r"\((\d+)\)", last) if last else None
            if pid_m:
                pid = pid_m.group(1)
                df  = self.retriever.df
                row = df[df["publication_number"].astype(str)==pid]
                if not row.empty and pd.notna(row.iloc[0].get("analysis_explanation")):
                    expl = row.iloc[0]["analysis_explanation"]
                    answer = f"({pid}) according to the inventor: {expl}"
                else:
                    answer = "I don’t have enough information from the inventor’s explanation."
                self.chat_history.extend([
                    {"role":"user",      "content":user_msg},
                    {"role":"assistant", "content":answer},
                ])
                return answer

        # ─── C. Summarise independent claims
        m_claim = re.search(r"claims (?:of|for)\s+([A-Z0-9]+)", user_msg, re.I)
        if m_claim:
            pid = m_claim.group(1)
            df  = self.retriever.df
            row = df[df["publication_number"].astype(str)==pid]
            if row.empty or not row.iloc[0].get("claims"):
                return "I don’t have enough information to summarize the claims."
            claims = row.iloc[0]["claims"]
            prompt = [
                {"role":"system", "content":"Summarise these patent claims in plain English."},
                {"role":"user",   "content":claims},
            ]
            ans = chat(prompt, temperature=0.0, max_tokens=512)
            self.chat_history.extend([
                {"role":"user",      "content":user_msg},
                {"role":"assistant", "content":ans},
            ])
            return ans

        # ─── D. Prior-art lookup
        m_prior = re.search(r"(?:prior[- ]art|cited by)\s+([A-Z0-9]+)", user_msg, re.I)
        if m_prior:
            pid = m_prior.group(1)
            df  = self.retriever.df
            row = df[df["publication_number"].astype(str)==pid]
            if row.empty or not row.iloc[0].get("prior_art"):
                return "I don’t have enough information on prior art."
            arts = [a.strip() for a in re.split(r"[;,]", row.iloc[0]["prior_art"]) if a.strip()]
            bullets = []
            for a in arts:
                match = df[df["publication_number"].astype(str)==a]
                title = f"“{match.iloc[0]['title_en']}”" if not match.empty else ""
                bullets.append(f"• ({a}) {title}")
            ans = "\n".join(bullets)
            self.chat_history.extend([
                {"role":"user",      "content":user_msg},
                {"role":"assistant", "content":ans},
            ])
            return ans

        # ─── E. Family / parent lookup
        if re.search(r"\b(?:family|parent)\b", user_msg, re.I):
            pid_match = re.search(r"([A-Z0-9]+)", user_msg)
            if pid_match:
                pid = pid_match.group(1)
                df  = self.retriever.df
                fam = df[
                    (df["parent_publication_number"].astype(str)==pid) |
                    (df["publication_number"].astype(str)==pid)
                ]
                if fam.empty:
                    return "I don’t have enough information on this patent family."
                bullets = [
                    f"• ({r['publication_number']}) {r['title_en']} — filed {r['publication_date']}"
                    for _, r in fam.iterrows()
                ]
                ans = "\n".join(bullets)
                self.chat_history.extend([
                    {"role":"user",      "content":user_msg},
                    {"role":"assistant", "content":ans},
                ])
                return ans

        # ─── F. “How … filed” → year-by-year counts
        if re.search(r"\bhow\b.*\bfiled\b", user_msg, re.I):
            df_sub = self._filter_df(self.retriever.df, filters)
            freqs  = group_by_year(df_sub, "publication_date")
            if not freqs:
                return "I don’t have enough information in the provided patents."
            bullets = [f"• {yr}: {cnt} patents" for yr, cnt in freqs.items()]
            ans = "\n".join(bullets)
            self.chat_history.extend([
                {"role":"user",      "content":user_msg},
                {"role":"assistant", "content":ans},
            ])
            return ans

        # ─── G. “Latest/Recent” inventions → date-sorted list
        if re.search(r"\b(latest|recent)\b", user_msg, re.I):
            from datetime import datetime
            def ordn(n:int)->str:
                if 10 <= (n%100) <= 20: s="th"
                else: s={1:"st",2:"nd",3:"rd"}.get(n%10,"th")
                return f"{n}{s}"

            df_sub = self._filter_df(self.retriever.df, filters)
            dates  = pd.to_datetime(df_sub["publication_date"], errors="coerce")
            df_s   = df_sub.loc[dates.sort_values(ascending=False).index][:10]
            bullets = []
            for _, r in df_s.iterrows():
                try:
                    d = datetime.strptime(str(r["publication_date"]), "%Y%m%d")
                except:
                    d = pd.to_datetime(r["publication_date"], errors="coerce")
                mth, day, yr = d.strftime("%B"), ordn(d.day), d.year
                bullets.append(
                    f"• ({r['publication_number']}) {r['title_en']} — "
                    f"Patent published on {mth} {day}, {yr}"
                )
            ans = "\n".join(bullets) or "I don’t have enough information."
            self.chat_history.extend([
                {"role":"user",      "content":user_msg},
                {"role":"assistant", "content":ans},
            ])
            return ans

        # ─── H. Aggregation branch (guarded against empty dict)
        if aggregation and isinstance(aggregation, dict) and aggregation.get("group_by"):
            df_sub = self._filter_df(self.retriever.df, filters)
            grp    = aggregation.get("group_by", "ipc_technologies")
            top_k  = aggregation.get("top_k", 10)

            if grp == "publication_date" or "each year" in user_msg.lower():
                freqs, is_year = group_by_year(df_sub, "publication_date"), True
            else:
                freqs, is_year = top_k_group(df_sub, grp, top_k), False

            if not freqs:
                return "I don’t have enough information in the provided patents."

            exemplars = {}
            for key in freqs:
                if is_year:
                    yrs  = pd.to_datetime(df_sub["publication_date"], errors="coerce").dt.year
                    mask = yrs == int(key)
                else:
                    col_ser = df_sub[grp]
                    if pd.api.types.is_string_dtype(col_ser):
                        mask = col_ser.fillna("").astype(str) \
                                     .str.contains(str(key), case=False, na=False)
                    else:
                        mask = col_ser == key
                if mask.any():
                    r = df_sub.loc[mask].iloc[0]
                    exemplars[key] = (str(r["publication_number"]), r["title_en"])

            lower = user_msg.lower()
            is_cnt = any(w in lower for w in ("top", "count", "how many", "number"))
            bullets = []
            for k, cnt in freqs.items():
                if k in exemplars:
                    pid, ttl = exemplars[k]
                    if is_cnt:
                        bullets.append(f"• {k} — {cnt} patents (e.g. ({pid}) “{ttl}”)")
                    else:
                        bullets.append(f"• {k} — {cnt} patents; example: ({pid}) “{ttl}”")
                else:
                    bullets.append(f"• {k} — {cnt} patents")
            ans = "\n".join(bullets)
            self.chat_history.extend([
                {"role":"user",      "content":user_msg},
                {"role":"assistant", "content":ans},
            ])
            return ans

        # ─── I. Passage-RAG with multi-stage fallback
        def try_search(filt, cols):
            return self.retriever.search(
                rq,
                max_passages   = 400,
                filters        = filt,
                column_order   = cols,
                top_k_return   = 60,
            )

        passages = try_search(filters, col_priority)
        if not passages and self.debug:
            print("⚠️ No hits with initial filters+priority → relaxing")
        if not passages:
            passages = try_search([], col_priority)
        if not passages and self.debug:
            print("⚠️ No hits with priority only → pure semantic")
        if not passages:
            passages = try_search([], [])
        if not passages:
            return "I don’t have enough information in the provided patents."

        # dedupe + token-budget fit
        seen, ctx, tok = set(), [], 0
        budget = MAX_CTX_TOKENS - PROMPT_OVERHEAD
        for p in passages:
            pid = p["publication_number"]
            if pid in seen:
                continue
            t = count_tokens(p["text"])
            if tok + t > budget:
                break
            seen.add(pid)
            ctx.append(p)
            tok += t
        if self.debug:
            print(f"[debug] picked {len(ctx)} chunks, {tok} tokens")

        raw_ctx = [
            f"[{p['publication_number']}] \"{p['title']}\" || {p['text']}"
            for p in ctx
        ]
        context = map_reduce_summarise(rq, raw_ctx)
        self._last_ctx_tokens = count_tokens(context)

        include_app = "applicant" in user_msg.lower() or "country" in user_msg.lower()
        allowed     = ", ".join(p["publication_number"] for p in ctx) or "NONE"
        fields      = ["publication_number", "title_en", "publication_date"]
        if include_app:
            fields.append("applicant_countries")

        system_prompt = (
            f"You may cite ONLY these publication numbers: {allowed}. "
            "If none answer, reply: 'I don’t have enough information.'\n"
            "Format bullets as: (" + ", ".join(fields) + ") — short note."
        )
        messages = (
            [{"role":"system","content":system_prompt}] +
            list(self.chat_history) +
            [{"role":"user","content":f"QUESTION: {user_msg}\n\nCONTEXT:\n{context}"}]
        )
        final_ans = chat(messages, temperature=0.0, max_tokens=512)

        self.chat_history.extend([
            {"role":"user",      "content":user_msg},
            {"role":"assistant", "content":final_ans},
        ])
        return final_ans
