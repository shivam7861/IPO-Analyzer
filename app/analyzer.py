"""
app/analyzer.py
Groq-powered analysis engine.
Analyzes 7 RHP sections in sequence then synthesizes a final investment verdict.
"""

import asyncio
import json
import logging
import os
import re
from typing import AsyncGenerator, Dict


def normalize_groq_key(groq_key: str | None) -> str:
    """Trim whitespace from the Groq API key and return an empty string if missing."""
    if groq_key is None:
        return ""
    return str(groq_key).strip()

from groq import AsyncGroq

from .rag import retrieve_relevant_chunks

logger = logging.getLogger(__name__)

# ── Section analysis configs ──────────────────────────────────────────────────
ANALYSIS_SECTIONS = [
    {
        "key": "objects_of_issue",
        "label": "Objects of the Issue",
        "icon": "🎯",
        "query": "objects of issue fresh capital OFS offer for sale proceeds utilization debt repayment",
        "prompt": """You are an expert Indian equity research analyst. Analyze the "Objects of the Issue" section of this IPO's Red Herring Prospectus.

SECTION TEXT:
{text}

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "score": <integer 1-10, where 10 is most investor-friendly>,
  "summary": "<2-3 sentence summary>",
  "fresh_capital_pct": <number or null>,
  "ofs_pct": <number or null>,
  "use_of_proceeds": ["<item1>", "<item2>"],
  "positives": ["<point>"],
  "negatives": ["<point>"],
  "red_flags": ["<serious concern if any>"],
  "analyst_note": "<one sharp takeaway for retail investor>"
}}

Scoring guide:
- High OFS % (>60%): investors cashing out → score down significantly
- Fresh capital for capex / growth: score up
- Vague uses like "general corporate purposes": score down
- Debt repayment: neutral-to-good if leverage will drop meaningfully""",
    },
    {
        "key": "risk_factors",
        "label": "Risk Factors",
        "icon": "⚠️",
        "query": "risk factors material business litigation regulatory promoter pledge customer concentration",
        "prompt": """You are an expert Indian equity research analyst. Analyze the "Risk Factors" section of this IPO's RHP.

SECTION TEXT:
{text}

Respond ONLY with valid JSON:
{{
  "score": <integer 1-10, 10 = very low risk>,
  "summary": "<2-3 sentence overview of risk profile>",
  "critical_risks": [
    {{"risk": "<name>", "description": "<one line>", "severity": "HIGH|MEDIUM|LOW"}}
  ],
  "risk_flags": {{
    "customer_concentration": true|false,
    "promoter_pledge": true|false,
    "regulatory_dependency": true|false,
    "pending_litigation": true|false,
    "single_product_dependence": true|false,
    "high_debt": true|false,
    "forex_risk": true|false
  }},
  "positives": ["<low-risk areas>"],
  "negatives": ["<high-concern risks>"],
  "red_flags": ["<anything unusual vs typical Indian company in this sector>"],
  "analyst_note": "<most important risk the retail investor must know>"
}}""",
    },
    {
        "key": "industry_overview",
        "label": "Industry Overview",
        "icon": "🏭",
        "query": "market size CAGR growth drivers competition industry structure tailwinds headwinds",
        "prompt": """You are an expert Indian equity research analyst. Analyze the "Industry Overview" section of this IPO's RHP.
Note: This research is usually commissioned (and paid for) by the company from CRISIL/CARE/similar — factor in appropriate skepticism.

SECTION TEXT:
{text}

Respond ONLY with valid JSON:
{{
  "score": <integer 1-10, 10 = most attractive industry>,
  "summary": "<2-3 sentence industry summary>",
  "industry_name": "<primary industry>",
  "market_size": "<e.g. ₹45,000 Cr in FY24>",
  "cagr": "<e.g. 18% CAGR FY24-FY29>",
  "research_commissioned_by": "<firm name e.g. CRISIL, CARE, Frost & Sullivan>",
  "growth_drivers": ["<driver>"],
  "challenges": ["<challenge>"],
  "competitive_intensity": "LOW|MEDIUM|HIGH",
  "positives": ["<favorable factors>"],
  "negatives": ["<unfavorable factors>"],
  "analyst_note": "<key industry insight>"
}}""",
    },
    {
        "key": "business",
        "label": "Business Overview",
        "icon": "🏢",
        "query": "revenue segment customer concentration competitive advantage moat geographic business model product",
        "prompt": """You are an expert Indian equity research analyst. Analyze the "Our Business" section of this IPO's RHP.

SECTION TEXT:
{text}

Respond ONLY with valid JSON:
{{
  "score": <integer 1-10, 10 = strongest business>,
  "summary": "<2-3 sentence plain-English description>",
  "what_company_does": "<one sentence>",
  "revenue_segments": [
    {{"segment": "<name>", "contribution": "<% or amount>"}}
  ],
  "customer_concentration": {{
    "top_customer_pct": <number or null>,
    "top5_customers_pct": <number or null>,
    "is_concentrated": true|false
  }},
  "geographic_mix": "<e.g. 80% domestic, 20% exports>",
  "competitive_moats": ["<moat1>"],
  "notable_clients": ["<client name if mentioned>"],
  "positives": ["<strength>"],
  "negatives": ["<weakness>"],
  "red_flags": ["<concern>"],
  "analyst_note": "<key business insight>"
}}""",
    },
    {
        "key": "financials",
        "label": "Financial Analysis",
        "icon": "📊",
        "query": "revenue profit EBITDA operating cash flow debt interest coverage working capital growth trend",
        "prompt": """You are an expert Indian equity research analyst. Analyze the financial statements from this IPO's RHP.
Focus on: 3-year trends, OCF vs net profit quality, and leverage.

SECTION TEXT:
{text}

Respond ONLY with valid JSON:
{{
  "score": <integer 1-10, 10 = strongest financials>,
  "summary": "<2-3 sentence financial summary>",
  "revenue_trend": [
    {{"year": "<FY>", "revenue": "<₹ Cr>", "growth_pct": <number or null>}}
  ],
  "profit_trend": [
    {{"year": "<FY>", "pat": "<₹ Cr>", "pat_margin_pct": <number or null>}}
  ],
  "ebitda_margin_latest": "<% or null>",
  "cash_flow": {{
    "operating_cf_latest": "<₹ Cr or null>",
    "net_profit_latest": "<₹ Cr or null>",
    "quality": "GOOD|MODERATE|POOR",
    "comment": "<e.g. OCF consistently > PAT, healthy conversion>"
  }},
  "debt": {{
    "total_debt": "<₹ Cr or null>",
    "de_ratio": "<D/E or null>",
    "interest_coverage": "<x or null>",
    "is_overleveraged": true|false
  }},
  "growth_momentum": "ACCELERATING|STABLE|DECELERATING",
  "positives": ["<financial strength>"],
  "negatives": ["<financial weakness>"],
  "red_flags": ["<OCF-PAT gap, inventory spike, receivables explosion, etc.>"],
  "analyst_note": "<most important financial insight>"
}}""",
    },
    {
        "key": "related_party",
        "label": "Related Party Transactions",
        "icon": "🔗",
        "query": "related party transactions promoter loans purchases sales rent value leakage conflict interest",
        "prompt": """You are an expert Indian equity research analyst. Analyze the "Related Party Transactions" section of this IPO's RHP.

SECTION TEXT:
{text}

Respond ONLY with valid JSON:
{{
  "score": <integer 1-10, 10 = cleanest / no concerns>,
  "summary": "<2-3 sentence assessment>",
  "significant_transactions": [
    {{"party": "<name>", "nature": "<type>", "amount": "<₹ value>", "concern": "LOW|MEDIUM|HIGH"}}
  ],
  "flags": {{
    "loans_to_promoters": true|false,
    "purchases_from_promoter_entities": true|false,
    "sales_to_promoter_entities": true|false,
    "rent_to_promoters": true|false,
    "management_remuneration_excessive": true|false
  }},
  "positives": ["<clean area>"],
  "negatives": ["<concerning transaction>"],
  "red_flags": ["<serious value leakage if any>"],
  "analyst_note": "<key concern or reassurance for investor>"
}}""",
    },
    {
        "key": "management",
        "label": "Management & Promoters",
        "icon": "👤",
        "query": "promoters directors management litigation SEBI penalty criminal case pledge share background experience",
        "prompt": """You are an expert Indian equity research analyst. Analyze the Management and Promoters section of this IPO's RHP.

SECTION TEXT:
{text}

Respond ONLY with valid JSON:
{{
  "score": <integer 1-10, 10 = exemplary management>,
  "summary": "<2-3 sentence management assessment>",
  "promoter_background": "<brief description>",
  "management_quality": "STRONG|ADEQUATE|WEAK",
  "litigation": {{
    "criminal_cases": true|false,
    "sebi_actions": true|false,
    "regulatory_penalties": true|false,
    "significant_civil_cases": true|false,
    "details": ["<notable case if any>"]
  }},
  "shareholding": {{
    "pre_ipo_pct": "<% or null>",
    "post_ipo_pct": "<% or null>",
    "shares_pledged": true|false,
    "pledge_pct": "<% or null>"
  }},
  "key_people": [
    {{"name": "<name>", "role": "<designation>", "tenure": "<years at company>"}}
  ],
  "positives": ["<positive signal>"],
  "negatives": ["<concern>"],
  "red_flags": ["<serious red flag if any>"],
  "analyst_note": "<key management insight for investor>"
}}""",
    },
]

# ── Final verdict prompt ──────────────────────────────────────────────────────
VERDICT_PROMPT = """You are a senior equity research analyst at a top Indian institutional brokerage.
Based on the complete section-by-section analysis below, provide a definitive investment verdict on this IPO.

ANALYSES:
{all_analyses}

Respond ONLY with valid JSON:
{{
  "verdict": "INVEST"|"CAUTION"|"AVOID",
  "overall_score": <number 1-10, one decimal>,
  "verdict_summary": "<3-4 sentence honest recommendation>",
  "top_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "top_weaknesses": ["<weakness 1>", "<weakness 2>", "<weakness 3>"],
  "ideal_investor": "<what type of investor this suits>",
  "avoid_if": "<type of investor / risk tolerance that should skip>",
  "valuation_note": "<comment on likely valuation based on financials seen — be honest>",
  "key_monitorables": ["<post-listing metric to watch>"],
  "section_scores": {{
    "objects_of_issue": <score>,
    "risk_factors": <score>,
    "industry_overview": <score>,
    "business": <score>,
    "financials": <score>,
    "related_party": <score>,
    "management": <score>
  }},
  "one_liner": "<One brutally honest sentence summing up this IPO>"
}}

Be direct. Retail investors need honest guidance, not marketing.
Verdict guide: INVEST = good fundamentals, fair/attractive valuation; CAUTION = mixed signals, wait and watch; AVOID = red flags or excessive valuation."""


class IPOAnalyzer:
    """Runs all 7 section analyses + final verdict using Groq LLM."""

    def __init__(self, sections: Dict[str, str], groq_key: str | None = None):
        api_key = normalize_groq_key(groq_key or os.environ.get("GROQ_API_KEY", ""))
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Paste your key in the UI or add it to .env"
            )
        self.client = AsyncGroq(api_key=api_key)
        self.model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.sections = sections
        self.all_analyses: Dict[str, dict] = {}
        logger.info(f"IPOAnalyzer initialized. Model: {self.model}")

    # ── Streaming entry point ─────────────────────────────────────────────────

    async def analyze_stream(self) -> AsyncGenerator[dict, None]:
        """Yield SSE-ready dicts for each analysis step."""

        for sec_cfg in ANALYSIS_SECTIONS:
            key = sec_cfg["key"]

            yield {
                "event": "section_start",
                "data": json.dumps({"key": key, "label": sec_cfg["label"], "icon": sec_cfg["icon"]}),
            }

            try:
                result = await self._analyze_section(sec_cfg)
            except Exception as exc:
                logger.error(f"Error analyzing {key}: {exc}")
                result = {
                    "section_key": key,
                    "label": sec_cfg["label"],
                    "icon": sec_cfg["icon"],
                    "score": 5,
                    "summary": f"Analysis could not be completed: {exc}",
                    "positives": [],
                    "negatives": [],
                    "red_flags": [str(exc)],
                    "analyst_note": "Section analysis failed.",
                }

            self.all_analyses[key] = result
            yield {"event": "section_complete", "data": json.dumps(result)}

        # Final verdict
        yield {"event": "verdict_start", "data": json.dumps({"message": "Synthesizing final verdict..."})}
        try:
            verdict = await self._generate_verdict()
        except Exception as exc:
            logger.error(f"Verdict generation failed: {exc}")
            verdict = {"error": str(exc), "verdict": "CAUTION", "overall_score": 5}

        self.all_analyses["verdict"] = verdict
        yield {"event": "verdict_complete", "data": json.dumps(verdict)}

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _analyze_section(self, sec_cfg: dict) -> dict:
        sec_text = self.sections.get(sec_cfg["key"], "")
        if not sec_text or len(sec_text.strip()) < 50:
            return {
                "section_key": sec_cfg["key"],
                "label": sec_cfg["label"],
                "icon": sec_cfg["icon"],
                "score": 5,
                "summary": "This section was not found in the uploaded document.",
                "positives": [],
                "negatives": ["Section not detected in RHP"],
                "red_flags": [],
                "analyst_note": "Could not locate this section. Verify the PDF is a standard RHP.",
            }

        # BM25-retrieve most relevant chunks for very long sections
        context = retrieve_relevant_chunks(
            text=sec_text,
            query=sec_cfg["query"],
            max_chars=14_000,
        )

        prompt = sec_cfg["prompt"].format(text=context)
        raw = await self._call_groq(prompt)
        parsed = self._parse_json(raw)
        parsed.setdefault("section_key", sec_cfg["key"])
        parsed.setdefault("label", sec_cfg["label"])
        parsed.setdefault("icon", sec_cfg["icon"])
        return parsed

    async def _generate_verdict(self) -> dict:
        # Build a concise summary of all section analyses
        summaries = {}
        for key, analysis in self.all_analyses.items():
            summaries[key] = {
                "score": analysis.get("score"),
                "summary": analysis.get("summary", ""),
                "positives": analysis.get("positives", [])[:3],
                "negatives": analysis.get("negatives", [])[:3],
                "red_flags": analysis.get("red_flags", [])[:2],
                "analyst_note": analysis.get("analyst_note", ""),
            }

        prompt = VERDICT_PROMPT.format(all_analyses=json.dumps(summaries, indent=2))
        raw = await self._call_groq(prompt, max_tokens=2000)
        return self._parse_json(raw)

    async def _call_groq(self, prompt: str, max_tokens: int = 1800) -> str:
        logger.info(f"Calling Groq ({self.model}), prompt length: {len(prompt):,} chars")
        try:
            resp = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert Indian equity research analyst specializing in IPO analysis. "
                                "Always respond with ONLY valid JSON. No markdown fences, no preamble, no explanations."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1,
                ),
                timeout=120.0,  # 2-minute hard timeout per call
            )
            result = resp.choices[0].message.content.strip()
            logger.info(f"Groq response received: {len(result):,} chars")
            return result
        except asyncio.TimeoutError:
            raise RuntimeError("Groq API call timed out after 120 seconds. Check your API key and network.")

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse JSON from LLM response, stripping any markdown wrappers."""
        text = text.strip()
        # Strip ```json ... ``` wrappers if present
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Last resort: grab the outermost {...}
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
        return {"error": "JSON parse failed", "raw_response": text[:400]}
