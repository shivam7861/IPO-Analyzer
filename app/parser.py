"""
app/parser.py
Parses an RHP PDF using Docling and extracts the 7 key analysis sections.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Section configuration ────────────────────────────────────────────────────
SECTIONS: Dict[str, Dict] = {
    "objects_of_issue": {
        "label": "Objects of the Issue",
        "patterns": [
            r"objects?\s+of\s+(?:the\s+)?(?:issue|offer)",
            r"use\s+of\s+(?:net\s+)?proceeds",
            r"utilization\s+of\s+(?:net\s+)?proceeds",
        ],
    },
    "risk_factors": {
        "label": "Risk Factors",
        "patterns": [r"risk\s+factors?"],
    },
    "industry_overview": {
        "label": "Industry Overview",
        "patterns": [
            r"industry\s+(?:overview|description|analysis)",
            r"market\s+overview",
            r"industry\s+and\s+market",
            r"sector\s+overview",
        ],
    },
    "business": {
        "label": "Our Business",
        "patterns": [
            r"our\s+business",
            r"business\s+overview",
            r"description\s+of\s+(?:our\s+)?business",
            r"business\s+description",
        ],
    },
    "financials": {
        "label": "Financial Statements",
        "patterns": [
            r"(?:restated\s+)?(?:summary\s+of\s+)?financial\s+(?:statements?|information|results|data)",
            r"financial\s+(?:summary|highlights)",
            r"summary\s+(?:of\s+)?(?:restated\s+)?(?:standalone|consolidated)\s+financial",
            r"(?:standalone|consolidated)\s+(?:restated\s+)?financial\s+statements?",
        ],
    },
    "related_party": {
        "label": "Related Party Transactions",
        "patterns": [
            r"related\s+party\s+transactions?",
            r"transactions?\s+with\s+related\s+part(?:y|ies)",
        ],
    },
    "management": {
        "label": "Management & Promoters",
        "patterns": [
            r"our\s+(?:promoters?|management)",
            r"promoters?\s+(?:and\s+)?(?:promoter\s+)?(?:group|background|information)",
            r"directors?\s+and\s+(?:key\s+)?management",
            r"key\s+managerial\s+personnel",
            r"management\s+(?:profile|team|overview)",
        ],
    },
}


class RHPParser:
    """Parses a Red Herring Prospectus PDF and extracts relevant sections."""

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self._last_markdown: str = ""   # saved here so caller can persist it

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_sections(self) -> Dict[str, str]:
        """
        Main entry point.
        Returns: {section_key: text, 'total_pages': int, 'company_name': str}
        """
        logger.info("[Parser] ── BEGIN extract_sections ──")
        logger.info("[Parser] Input file: %s", self.pdf_path)
        logger.info("[Parser] File size: %.2f MB", self.pdf_path.stat().st_size / 1_048_576)

        import time
        t0 = time.perf_counter()

        logger.info("[Parser] Starting Docling PDF conversion...")
        markdown, total_pages = self._convert_pdf()
        self._last_markdown = markdown   # store for external saving

        t1 = time.perf_counter()
        logger.info("[Parser] Docling conversion done in %.1fs", t1 - t0)
        logger.info("[Parser] Extracting sections from parsed document...")
        sections = self._extract_from_markdown(markdown)

        t2 = time.perf_counter()
        logger.info("[Parser] Section extraction done in %.2fs", t2 - t1)

        sections["total_pages"] = total_pages
        sections["company_name"] = self._detect_company_name(markdown)

        found = [k for k in sections if k not in ("total_pages", "company_name") and sections[k]]
        logger.info("[Parser] ── END extract_sections (total %.1fs) ──", t2 - t0)
        logger.info("[Parser] Company: %s | Pages: %s | Sections found: %s",
                    sections['company_name'], total_pages, found)
        return sections

    # ── Private helpers ───────────────────────────────────────────────────────

    def _convert_pdf(self) -> Tuple[str, int]:
        """Run Docling on the PDF and return (markdown_text, page_count)."""
        logger.info("[Parser] Loading Docling DocumentConverter...")
        from docling.document_converter import DocumentConverter

        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import PdfFormatOption

            opts = PdfPipelineOptions()
            opts.do_ocr = False            # RHPs are text PDFs — skip OCR
            opts.do_table_structure = True  # Keep tables (needed for financials)

            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
            logger.info("[Parser] Pipeline configured: OCR=OFF, TableStructure=ON")
        except Exception as e:
            logger.warning("[Parser] Could not configure pipeline (%s) — using default converter", e)
            converter = DocumentConverter()

        logger.info("[Parser] Starting Docling.convert() — this is the slow step for large PDFs...")
        import time
        t0 = time.perf_counter()
        result = converter.convert(str(self.pdf_path))
        elapsed = time.perf_counter() - t0
        logger.info("[Parser] Docling.convert() finished in %.1fs", elapsed)

        doc = result.document

        total_pages: int = 0
        fn = getattr(doc, "num_pages", None)
        if callable(fn):
            try:
                total_pages = fn()
            except Exception:
                pass
        if not total_pages:
            try:
                total_pages = len(result.pages)
            except Exception:
                total_pages = 0

        logger.info("[Parser] Exporting document to Markdown...")
        t1 = time.perf_counter()
        markdown = doc.export_to_markdown()
        logger.info("[Parser] Markdown export done in %.2fs — %d pages, %s chars",
                    time.perf_counter() - t1, total_pages, f"{len(markdown):,}")
        return markdown, total_pages

    def _extract_from_markdown(self, markdown: str) -> Dict[str, str]:
        """
        Two-pass section extraction.

        Pass 1 — Heading candidates
          Every heading matching a section pattern is collected (numbered
          sub-items like "6. Our business..." are filtered out).

        Pass 2 — Section-boundary content extraction
          For each candidate, text is extracted from that heading until the
          NEXT occurrence of ANY OTHER section's pattern in the markdown.
          This means Risk Factors ends at Our Business (not at its own
          internal sub-headings), Our Business ends at Financials, etc.

          Among all candidates for a section, the one with the most content
          wins — TOC stubs always have far less content than real chapters.
        """
        import time
        t0 = time.perf_counter()
        logger.info("[Parser] Starting section extraction from %s chars of markdown", f"{len(markdown):,}")

        lines = markdown.split("\n")
        logger.info("[Parser] Document has %d lines", len(lines))

        headings = self._find_headings(lines)
        logger.info("[Parser] Found %d total headings in document", len(headings))

        # Build char-offset table: line i starts at line_offsets[i]
        line_offsets: List[int] = [0] * (len(lines) + 1)
        for i, ln in enumerate(lines):
            line_offsets[i + 1] = line_offsets[i] + len(ln) + 1  # +1 for \n

        # Pre-scan: every section pattern hit in the whole markdown, tagged
        # with section key, sorted by char position.
        all_hits: List[Tuple[int, str]] = []
        for sec_key, cfg in SECTIONS.items():
            for pat in cfg["patterns"]:
                for m in re.finditer(pat, markdown, re.IGNORECASE):
                    all_hits.append((m.start(), sec_key))
        all_hits.sort(key=lambda x: x[0])
        logger.info("[Parser] Pre-scanned %d total pattern hits across all sections", len(all_hits))

        def section_end(start_char: int, current_key: str, max_chars: int = 150_000) -> int:
            cap = min(len(markdown), start_char + max_chars)
            skip = start_char + 300  # skip past the heading text itself
            for pos, key in all_hits:
                if pos < skip:
                    continue
                if pos >= cap:
                    break
                if key == current_key:
                    continue
                # Confirm heading context: must be preceded by \n
                before = markdown[max(0, pos - 5): pos]
                if "\n" in before or pos < 10:
                    return pos
            return cap

        # Pass 1 — collect heading candidates
        section_candidates: Dict[str, List[Dict]] = {k: [] for k in SECTIONS}
        skipped_numbered = 0
        for h in headings:
            if self._is_numbered_subitem(h["text"]):
                skipped_numbered += 1
                continue
            h_lower = h["text"].lower()
            for sec_key, cfg in SECTIONS.items():
                for pat in cfg["patterns"]:
                    if re.search(pat, h_lower, re.IGNORECASE):
                        section_candidates[sec_key].append({
                            **h,
                            "char_pos": line_offsets[h["line"]],
                        })
                        break

        logger.info("[Parser] Pass 1 complete: %d numbered sub-items skipped", skipped_numbered)
        for sec_key, cands in section_candidates.items():
            logger.info("[Parser]   %-22s → %d candidate heading(s)", sec_key, len(cands))

        # Pass 2 — extract & pick best
        result: Dict[str, str] = {}
        MIN_CHARS = 400
        logger.info("[Parser] Pass 2: extracting content for each section...")

        for sec_key, candidates in section_candidates.items():
            best_text = ""
            best_match = None
            for match in candidates:
                end = section_end(match["char_pos"], sec_key)
                text = markdown[match["char_pos"]: end]
                if len(text) > len(best_text):
                    best_text = text
                    best_match = match

            if best_text.strip() and len(best_text) >= MIN_CHARS:
                result[sec_key] = best_text
                logger.info(
                    "[Parser]   ✅ %-22s → %s chars (line %d, '%s', %d candidate(s))",
                    sec_key, f"{len(best_text):,}",
                    best_match["line"], best_match["text"][:50], len(candidates)
                )
            elif candidates:
                logger.warning(
                    "[Parser]   ⚠️  %-22s → all %d candidate(s) too short (best=%d chars), trying fallback",
                    sec_key, len(candidates), len(best_text)
                )
            else:
                logger.warning("[Parser]   ❌ %-22s → no heading candidates found", sec_key)

        # Fallback for any section still not found
        fallback_used = []
        for sec_key, cfg in SECTIONS.items():
            if sec_key not in result:
                text = self._text_fallback(markdown, cfg["patterns"])
                if text:
                    result[sec_key] = text
                    fallback_used.append(sec_key)
                    logger.info("[Parser]   🔍 %-22s → text fallback used (%s chars)",
                                sec_key, f"{len(text):,}")
                else:
                    logger.error("[Parser]   ❌ %-22s → COMPLETELY MISSING from document", sec_key)

        import time
        logger.info("[Parser] Pass 2 complete. Sections extracted: %d/%d | Fallback used: %s",
                    len(result), len(SECTIONS), fallback_used or "none")
        return result

    def _find_headings(self, lines: List[str]) -> List[Dict]:
        """Extract headings (# markdown or **bold** lines) with line numbers."""
        headings = []
        for i, line in enumerate(lines):
            s = line.strip()
            m = re.match(r"^(#{1,4})\s+(.+)$", s)
            if m:
                headings.append({"line": i, "level": len(m.group(1)), "text": m.group(2).strip()})
                continue
            # Bold-only line — Docling often outputs chapter headings this way
            m = re.match(r"^\*\*([^*]{3,80})\*\*$", s)
            if m:
                headings.append({"line": i, "level": 3, "text": m.group(1).strip()})
        return headings

    @staticmethod
    def _is_numbered_subitem(text: str) -> bool:
        """
        Returns True for numbered/lettered sub-items that are NOT chapter headings,
        e.g. "6. Our business is subject to...", "ii. Regulatory risks", "A. Background"
        """
        return bool(re.match(r"^\s*(?:\d{1,2}|[ivxlIVXL]{1,5}|[A-E])[.)\s]\s*\S", text))

    def _text_fallback(self, markdown: str, patterns: List[str], window: int = 30_000) -> Optional[str]:
        """Last-resort: find pattern by text and return a window of chars around it."""
        for pat in patterns:
            m = re.search(pat, markdown, re.IGNORECASE)
            if m:
                start = max(0, m.start() - 200)
                end = min(len(markdown), m.start() + window)
                return markdown[start:end]
        return None

    def _detect_company_name(self, markdown: str) -> str:
        """Best-effort extraction of company name from the first ~3000 chars."""
        snippet = markdown[:3000]
        m = re.search(
            r"([A-Z][A-Za-z0-9\s&.\-]{3,60}(?:LIMITED|LTD\.?|PRIVATE\s+LIMITED|PVT\.?\s*LTD\.?))",
            snippet,
        )
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        return "Unknown Company"
