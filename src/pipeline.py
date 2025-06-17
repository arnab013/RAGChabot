# src/pipeline.py

import re
from collections import Counter
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from query_rewrite    import rewrite
from retrieval        import PassageRetriever
from filter_ops       import apply_filter
from stats_engine     import top_k_group, group_by_year
from summarise        import map_reduce_summarise
from llm_clients      import chat
from token_utils      import count_tokens

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

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            df = self.retriever.df
            if df is None or df.empty:
                return {"error": "No data available", "total_patents": 0}
            
            stats = {
                "total_patents": len(df),
                "by_country": {},
                "by_year": {},
                "by_decade": {},
                "by_technology": {},
                "by_inventor": {},
                "by_applicant": {},
                "date_range": {},
                "available_columns": list(df.columns)
            }
            
            # Country statistics (check multiple possible column names)
            country_cols = ['country', 'applicant_countries', 'inventor_countries']
            for col in country_cols:
                if col in df.columns:
                    country_data = df[col].dropna()
                    if not country_data.empty:
                        # Handle multiple countries separated by semicolons
                        all_countries = []
                        for countries in country_data:
                            if isinstance(countries, str):
                                all_countries.extend([c.strip() for c in countries.split(';') if c.strip()])
                        if all_countries:
                            from collections import Counter
                            stats["by_country"] = dict(Counter(all_countries).most_common(15))
                        break
            
            # Publication year statistics
            date_cols = ['publication_date', 'filing_date', 'priority_date']
            for col in date_cols:
                if col in df.columns:
                    dates = df[col].dropna()
                    if not dates.empty:
                        # Extract years from various date formats
                        years = dates.astype(str).str.extract(r'(\d{4})')[0].dropna()
                        if not years.empty:
                            from collections import Counter
                            year_counts = Counter(years)
                            stats["by_year"] = dict(year_counts.most_common(10))
                            
                            # Decade analysis
                            decades = [(int(year) // 10 * 10) for year in years if year.isdigit()]
                            decade_counts = Counter([f"{decade}s" for decade in decades])
                            stats["by_decade"] = dict(decade_counts.most_common())
                            
                            # Date range
                            stats["date_range"] = {
                                "earliest": min(years),
                                "latest": max(years)
                            }
                        break
            
            # Technology analysis from titles and abstracts
            tech_cols = ['title_en', 'title', 'abstract_text', 'abstract']
            tech_keywords = []
            for col in tech_cols:
                if col in df.columns:
                    text_data = df[col].dropna()
                    if not text_data.empty:
                        tech_keywords.extend(self._extract_technology_keywords(text_data))
                        break
            
            if tech_keywords:
                from collections import Counter
                stats["by_technology"] = dict(Counter(tech_keywords).most_common(15))
            
            # Inventor analysis
            inventor_cols = ['inventor_names', 'inventor', 'inventors']
            for col in inventor_cols:
                if col in df.columns:
                    inventors = df[col].dropna()
                    if not inventors.empty:
                        all_inventors = []
                        for inv_list in inventors:
                            if isinstance(inv_list, str):
                                all_inventors.extend([inv.strip() for inv in inv_list.split(';') if inv.strip()])
                        if all_inventors:
                            from collections import Counter
                            stats["by_inventor"] = dict(Counter(all_inventors).most_common(15))
                        break
            
            # Applicant/Assignee analysis
            applicant_cols = ['applicant_names', 'applicant', 'assignee', 'assignees']
            for col in applicant_cols:
                if col in df.columns:
                    applicants = df[col].dropna()
                    if not applicants.empty:
                        all_applicants = []
                        for app_list in applicants:
                            if isinstance(app_list, str):
                                all_applicants.extend([app.strip() for app in app_list.split(';') if app.strip()])
                        if all_applicants:
                            from collections import Counter
                            stats["by_applicant"] = dict(Counter(all_applicants).most_common(15))
                        break
            
            return stats
            
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Error getting database stats: {e}")
            return {"error": str(e), "total_patents": 0}

    def _extract_technology_keywords(self, text_series) -> List[str]:
        """Extract technology keywords from patent texts"""
        tech_patterns = {
            'Artificial Intelligence': ['artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'ai', 'ml'],
            'Internet of Things': ['internet of things', 'iot', 'connected device', 'smart device', 'sensor network'],
            'Blockchain': ['blockchain', 'cryptocurrency', 'distributed ledger', 'bitcoin', 'smart contract'],
            'Renewable Energy': ['solar', 'wind energy', 'renewable', 'clean energy', 'photovoltaic', 'sustainable energy'],
            'Biotechnology': ['biotech', 'genetic', 'protein', 'dna', 'pharmaceutical', 'bio', 'medical'],
            'Automotive': ['vehicle', 'automotive', 'car', 'electric vehicle', 'autonomous', 'transportation'],
            'Medical Device': ['medical device', 'diagnostic', 'therapeutic', 'implant', 'health monitoring'],
            'Communication': ['wireless', 'telecommunication', '5g', 'antenna', 'signal', 'network'],
            'Manufacturing': ['manufacturing', 'production', 'assembly', 'industrial', 'automation'],
            'Agriculture': ['agriculture', 'farming', 'crop', 'irrigation', 'precision agriculture'],
            'Environmental': ['environmental', 'pollution', 'waste', 'recycling', 'carbon', 'emission'],
            'Software': ['software', 'algorithm', 'computing', 'data processing', 'application'],
            'Nanotechnology': ['nano', 'nanotechnology', 'nanomaterial', 'nanoparticle'],
            'Robotics': ['robot', 'robotics', 'automation', 'robotic', 'autonomous system']
        }
        
        keywords = []
        for text in text_series:
            if pd.isna(text):
                continue
            text_lower = str(text).lower()
            for category, patterns in tech_patterns.items():
                if any(pattern in text_lower for pattern in patterns):
                    keywords.append(category)
                    break  # Only count once per text
        
        return keywords

    def handle_stats_query(self, query: str) -> str:
        """Handle database statistics and counting queries"""
        query_lower = query.lower()
        stats = self.get_database_stats()
        
        if stats.get("error"):
            return f"I'm sorry, but I encountered an issue accessing the database: {stats['error']}. Please try again or contact support if this persists."
        
        # Total patents query
        if any(phrase in query_lower for phrase in ["how many patents", "total patents", "number of patents", "database size"]):
            response = f"📊 **Database Overview**\n\n"
            response += f"I currently have access to **{stats['total_patents']:,} patents** in my database.\n\n"
            
            if stats.get('date_range'):
                response += f"**Date Range:** {stats['date_range']['earliest']} - {stats['date_range']['latest']}\n\n"
            
            # Add quick overview of available categories
            overview_items = []
            if stats.get('by_country'):
                top_country = list(stats['by_country'].items())[0]
                overview_items.append(f"Countries: {len(stats['by_country'])} (top: {top_country[0]} with {top_country[1]} patents)")
            
            if stats.get('by_technology'):
                top_tech = list(stats['by_technology'].items())[0]
                overview_items.append(f"Technologies: {len(stats['by_technology'])} (top: {top_tech[0]} with {top_tech[1]} patents)")
            
            if overview_items:
                response += "**Quick Overview:**\n" + "\n".join([f"• {item}" for item in overview_items]) + "\n\n"
            
            response += "💡 *Ask me about specific categories like 'patents by country', 'patents by year', or 'technology breakdown' for detailed analysis!*"
            return response
        
        # Country-based queries
        elif any(phrase in query_lower for phrase in ["by country", "country", "countries", "which countries"]):
            if not stats.get('by_country'):
                return "I apologize, but I don't have country information available in the current dataset. The available data columns are: " + ", ".join(stats.get('available_columns', []))
            
            response = f"🌍 **Patents by Country**\n\n"
            response += f"**Total Countries Represented:** {len(stats['by_country'])}\n\n"
            
            for i, (country, count) in enumerate(list(stats['by_country'].items())[:10], 1):
                percentage = (count / stats['total_patents']) * 100
                response += f"{i}. **{country}**: {count:,} patents ({percentage:.1f}%)\n"
            
            if len(stats['by_country']) > 10:
                response += f"\n*... and {len(stats['by_country']) - 10} more countries*"
            
            return response
        
        # Time-based queries
        elif any(phrase in query_lower for phrase in ["by year", "yearly", "timeline", "over time", "per year"]):
            if not stats.get('by_year'):
                return "I apologize, but I don't have publication date information available in the current dataset."
            
            response = f"📅 **Patents by Year**\n\n"
            response += f"**Years Covered:** {stats['date_range']['earliest']} - {stats['date_range']['latest']}\n\n"
            
            # Show recent years
            response += "**Recent Years:**\n"
            sorted_years = sorted(stats['by_year'].items(), reverse=True)
            for year, count in sorted_years[:10]:
                response += f"• **{year}**: {count:,} patents\n"
            
            # Show decade breakdown if available
            if stats.get('by_decade'):
                response += f"\n**By Decade:**\n"
                for decade, count in sorted(stats['by_decade'].items(), reverse=True):
                    response += f"• **{decade}**: {count:,} patents\n"
            
            return response
        
        # Technology-based queries
        elif any(phrase in query_lower for phrase in ["technology", "technologies", "technical field", "domain", "tech"]):
            if not stats.get('by_technology'):
                return "I apologize, but I don't have enough technology classification data available. This might be because the patent titles/abstracts don't contain recognizable technology keywords."
            
            response = f"🔬 **Patents by Technology Domain**\n\n"
            response += f"**Technology Areas Identified:** {len(stats['by_technology'])}\n\n"
            
            for i, (tech, count) in enumerate(list(stats['by_technology'].items())[:12], 1):
                percentage = (count / stats['total_patents']) * 100
                response += f"{i}. **{tech}**: {count:,} patents ({percentage:.1f}%)\n"
            
            return response
        
        # Inventor-based queries
        elif any(phrase in query_lower for phrase in ["inventor", "inventors", "who invented", "by inventor"]):
            if not stats.get('by_inventor'):
                return "I apologize, but I don't have detailed inventor information available in the current dataset."
            
            response = f"👨‍🔬 **Top Inventors**\n\n"
            response += f"**Total Inventors:** {len(stats['by_inventor'])}\n\n"
            
            for i, (inventor, count) in enumerate(list(stats['by_inventor'].items())[:10], 1):
                response += f"{i}. **{inventor}**: {count:,} patents\n"
            
            return response
        
        # Company/Applicant queries
        elif any(phrase in query_lower for phrase in ["company", "companies", "assignee", "applicant", "organization"]):
            if not stats.get('by_applicant'):
                return "I apologize, but I don't have assignee/applicant information available in the current dataset."
            
            response = f"🏢 **Top Patent Assignees/Companies**\n\n"
            response += f"**Total Organizations:** {len(stats['by_applicant'])}\n\n"
            
            for i, (applicant, count) in enumerate(list(stats['by_applicant'].items())[:10], 1):
                response += f"{i}. **{applicant}**: {count:,} patents\n"
            
            return response
        
        # General breakdown query
        else:
            response = f"📊 **Database Statistics Overview**\n\n"
            response += f"**Total Patents:** {stats['total_patents']:,}\n\n"
            
            available_breakdowns = []
            if stats.get('by_country'): available_breakdowns.append(f"**Countries**: {len(stats['by_country'])} countries")
            if stats.get('by_year'): available_breakdowns.append(f"**Years**: {len(stats['by_year'])} years")
            if stats.get('by_technology'): available_breakdowns.append(f"**Technologies**: {len(stats['by_technology'])} domains")
            if stats.get('by_inventor'): available_breakdowns.append(f"**Inventors**: {len(stats['by_inventor'])} inventors")
            if stats.get('by_applicant'): available_breakdowns.append(f"**Companies**: {len(stats['by_applicant'])} organizations")
            
            if available_breakdowns:
                response += "**Available Breakdowns:**\n" + "\n".join([f"• {breakdown}" for breakdown in available_breakdowns]) + "\n\n"
                response += "💡 *Ask me for specific breakdowns! Examples:*\n"
                response += "• 'Show me patents by country'\n"
                response += "• 'How many patents per year?'\n"
                response += "• 'What technologies are covered?'\n"
                response += "• 'Top inventors in the database'"
            
            return response

    def ask(self, user_msg: str, conversation_context: List[Dict] = None) -> str:
        
        # Update internal chat history with conversation context if provided
        if conversation_context:
            # Convert conversation context to internal format and update chat_history
            self.chat_history.clear()
            for msg in conversation_context[-10:]:  # Keep last 10 messages
                self.chat_history.append(msg)

        # ─── Check for database statistics queries first ─────────────────
        stats_keywords = [
            "how many patents", "total patents", "number of patents", "database size",
            "by country", "by year", "by technology", "by inventor", "by applicant",
            "statistics", "breakdown", "categories", "patents per", "distribution",
            "which countries", "what technologies", "top inventors", "companies"
        ]
        
        if any(keyword in user_msg.lower() for keyword in stats_keywords):
            return self.handle_stats_query(user_msg)

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
            answer = chat([system, user_ctx], temperature=0.7, max_tokens=5000)
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
            ans = chat(prompt, temperature=0.0, max_tokens=5000)
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
        allowed     = ", ".join(f"EP{p['publication_number']}" for p in ctx) or "NONE"
        fields      = ["publication_number", "title_en", "publication_date"]
        if include_app:
            fields.append("applicant_countries")

        system_prompt = (
            f"You are GoalDigger, a helpful and polite AI assistant specializing in patent research. "
            f"You may cite ONLY these publication numbers: {allowed}. "
            "When you cannot find relevant information to answer a user's query, politely acknowledge this by saying: "
            "'I apologize, but I don't have enough information in my current database to answer your specific question about [topic]. "
            "However, I'd be happy to help you with other patent-related queries or provide information on related topics if available.' "
            "When presenting patent information, use natural language format: "
            "The patent EP[number] titled '[title]' was published in [date] which [details]... "
            "Always prefix patent numbers with 'EP' (e.g., EP123456). "
            "Present information in a conversational, flowing manner. "
            "Be courteous, acknowledge limitations gracefully, and offer alternative assistance when possible. "
            "Organize multiple patents clearly with proper headings and complete sentences."
        )
        messages = (
            [{"role":"system","content":system_prompt}] +
            list(self.chat_history) +
            [{"role":"user","content":f"QUESTION: {user_msg}\n\nCONTEXT:\n{context}"}]
        )
        final_ans = chat(messages, temperature=0.0, max_tokens=5000)

        self.chat_history.extend([
            {"role":"user",      "content":user_msg},
            {"role":"assistant", "content":final_ans},
        ])
        return final_ans
