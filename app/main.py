"""
app/main.py
FastAPI backend — handles PDF upload, background analysis, and SSE streaming.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, ListFlowable

from .analyzer import normalize_groq_key

load_dotenv()

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="IPO Lens — AI RHP Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Storage ────────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# task_id → {status, file_path, parsed_md_path, filename, groq_key, queue, report, timings}
tasks: Dict[str, dict] = {}

STATIC_DIR = Path("static")

logger.info("IPO Lens server initialised. Upload dir: %s", UPLOAD_DIR.resolve())


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Accept a PDF upload, save it to disk, and return a task_id."""
    logger.info("=== NEW UPLOAD REQUEST ===")
    logger.info("Filename: %s | Content-Type: %s", file.filename, file.content_type)

    if not file.filename.lower().endswith(".pdf"):
        logger.warning("Rejected non-PDF file: %s", file.filename)
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    task_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{task_id}.pdf"

    t0 = time.perf_counter()
    content = await file.read()
    elapsed = time.perf_counter() - t0

    dest.write_bytes(content)
    size_mb = len(content) / 1_048_576

    logger.info("PDF saved → %s  (%.2f MB, read in %.2fs)", dest, size_mb, elapsed)
    logger.info("Task ID assigned: %s", task_id)

    tasks[task_id] = {
        "status": "uploaded",
        "file_path": str(dest),
        "parsed_md_path": str(UPLOAD_DIR / f"{task_id}_parsed.md"),
        "filename": file.filename,
        "size_bytes": len(content),
        "groq_key": None,
        "queue": asyncio.Queue(),
        "report": None,
        "timings": {"upload_done": time.time()},
    }

    logger.info("Task registered. Ready to stream at /api/stream/%s", task_id)
    return {"task_id": task_id, "filename": file.filename, "size_bytes": len(content)}


@app.get("/api/stream/{task_id}")
async def stream_analysis(task_id: str, request: Request, groq_key: str = ""):
    """
    SSE endpoint — starts the analysis pipeline and streams events to the client.
    Optional query param: groq_key (falls back to GROQ_API_KEY env var).
    """
    logger.info("=== SSE STREAM REQUEST for task: %s ===", task_id)

    if task_id not in tasks:
        logger.warning("Unknown task_id requested: %s", task_id)
        raise HTTPException(status_code=404, detail="Task not found.")

    task = tasks[task_id]
    normalized_key = normalize_groq_key(groq_key or os.environ.get("GROQ_API_KEY", ""))
    if normalized_key:
        task["groq_key"] = normalized_key
        logger.info("Groq API key received for task %s", task_id)
    else:
        logger.warning("No Groq API key found for task %s — analysis will fail", task_id)

    async def event_generator():
        logger.info("[%s] Starting background analysis task", task_id)
        analysis = asyncio.create_task(_run_analysis(task_id))

        events_sent = 0
        try:
            while True:
                if await request.is_disconnected():
                    logger.warning("[%s] Client disconnected — cancelling analysis", task_id)
                    analysis.cancel()
                    break

                try:
                    event = await asyncio.wait_for(task["queue"].get(), timeout=2.0)
                    if event is None:   # Sentinel → done
                        logger.info("[%s] Sentinel received — stream complete. Events sent: %d", task_id, events_sent)
                        break
                    events_sent += 1
                    yield event
                except asyncio.TimeoutError:
                    yield {"event": "heartbeat", "data": "ping"}

        except asyncio.CancelledError:
            logger.info("[%s] event_generator cancelled", task_id)
            analysis.cancel()

    return EventSourceResponse(event_generator())


@app.get("/api/report/{task_id}")
async def get_report(task_id: str):
    """Return the completed JSON report for a task."""
    logger.info("Report requested for task: %s", task_id)
    report = _load_report_for_task(task_id)
    if report is None:
        logger.info("Report not ready yet for task: %s", task_id)
        raise HTTPException(status_code=202, detail="Analysis still in progress.")
    logger.info("Returning completed report for task: %s", task_id)
    return report


@app.get("/api/download/{task_id}")
async def download_report(task_id: str):
    """Return the final analysis as a downloadable PDF report."""
    logger.info("Download requested for task: %s", task_id)
    report = _load_report_for_task(task_id)
    if report is None:
        raise HTTPException(status_code=202, detail="Analysis still in progress.")

    pdf_bytes = build_download_pdf(report)
    filename = f"{task_id}_ipo_report.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _load_report_for_task(task_id: str):
    """Load report from memory or from the saved JSON file on disk."""
    if task_id in tasks:
        report = tasks[task_id].get("report")
        if report is not None:
            return report

    report_path = UPLOAD_DIR / f"{task_id}_report.json"
    if report_path.exists():
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Could not decode report JSON at %s", report_path)
    return None


def build_download_report(report: dict) -> str:
    """Create a plain-text report suitable for download."""
    verdict = report.get("verdict", {}) if isinstance(report, dict) else {}
    if not isinstance(verdict, dict):
        verdict = {}

    lines = [
        "# IPO Analysis Report",
        "",
        "## Verdict",
        f"- Verdict: {verdict.get('verdict', 'CAUTION')}",
        f"- Overall Score: {verdict.get('overall_score', 'N/A')}/10",
        f"- Summary: {verdict.get('verdict_summary', '')}",
        f"- One-liner: {verdict.get('one_liner', '')}",
        "",
    ]

    strengths = verdict.get("top_strengths") or []
    weaknesses = verdict.get("top_weaknesses") or []
    monitorables = verdict.get("key_monitorables") or []

    if strengths:
        lines.append("## Top Strengths")
        for item in strengths:
            lines.append(f"- {item}")
        lines.append("")

    if weaknesses:
        lines.append("## Key Weaknesses")
        for item in weaknesses:
            lines.append(f"- {item}")
        lines.append("")

    if monitorables:
        lines.append("## Post-listing Monitorables")
        for item in monitorables:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Section Analysis")
    for key, section in report.items():
        if key == "verdict" or not isinstance(section, dict):
            continue
        label = section.get("label") or key.replace("_", " ").title()
        score = section.get("score", "N/A")
        summary = section.get("summary", "")
        positives = section.get("positives") or []
        negatives = section.get("negatives") or []
        red_flags = section.get("red_flags") or []

        lines.extend([
            f"### {label}",
            f"- Score: {score}/10",
            f"- Summary: {summary}",
        ])
        if positives:
            lines.append("- Positives:")
            for item in positives:
                lines.append(f"  - {item}")
        if negatives:
            lines.append("- Concerns:")
            for item in negatives:
                lines.append(f"  - {item}")
        if red_flags:
            lines.append("- Red Flags:")
            for item in red_flags:
                lines.append(f"  - {item}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_download_pdf(report: dict) -> bytes:
    """Create a PDF version of the report for download."""
    story = []
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]

    story.append(Paragraph("IPO Analysis Report", title_style))
    story.append(Spacer(1, 12))

    verdict = report.get("verdict", {}) if isinstance(report, dict) else {}
    if not isinstance(verdict, dict):
        verdict = {}

    story.append(Paragraph(f"Verdict: {verdict.get('verdict', 'CAUTION')}", heading_style))
    story.append(Paragraph(f"Overall Score: {verdict.get('overall_score', 'N/A')}/10", body_style))
    story.append(Paragraph(f"Summary: {verdict.get('verdict_summary', '')}", body_style))
    story.append(Paragraph(f"One-liner: {verdict.get('one_liner', '')}", body_style))
    story.append(Spacer(1, 12))

    strengths = verdict.get("top_strengths") or []
    weaknesses = verdict.get("top_weaknesses") or []
    monitorables = verdict.get("key_monitorables") or []

    if strengths:
        story.append(Paragraph("Top Strengths", heading_style))
        story.append(ListFlowable([Paragraph(item, body_style) for item in strengths], bulletType="bullet", start='bullet'))
        story.append(Spacer(1, 8))

    if weaknesses:
        story.append(Paragraph("Key Weaknesses", heading_style))
        story.append(ListFlowable([Paragraph(item, body_style) for item in weaknesses], bulletType="bullet", start='bullet'))
        story.append(Spacer(1, 8))

    if monitorables:
        story.append(Paragraph("Post-listing Monitorables", heading_style))
        story.append(ListFlowable([Paragraph(item, body_style) for item in monitorables], bulletType="bullet", start='bullet'))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Section Analysis", heading_style))
    for key, section in report.items():
        if key == "verdict" or not isinstance(section, dict):
            continue
        label = section.get("label") or key.replace("_", " ").title()
        story.append(Paragraph(label, styles["Heading3"]))
        story.append(Paragraph(f"Score: {section.get('score', 'N/A')}/10", body_style))
        story.append(Paragraph(f"Summary: {section.get('summary', '')}", body_style))
        story.append(Spacer(1, 6))

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title="IPO Analysis Report")
    doc.build(story)
    return buffer.getvalue()


# ── Background analysis pipeline ───────────────────────────────────────────────

async def _run_analysis(task_id: str):
    from .parser import RHPParser
    from .analyzer import IPOAnalyzer

    task = tasks[task_id]
    queue: asyncio.Queue = task["queue"]
    timings = task["timings"]

    async def put(event: str, data: dict):
        await queue.put({"event": event, "data": json.dumps(data)})

    pipeline_start = time.time()
    logger.info("[%s] ── PIPELINE START ──────────────────────────", task_id)

    try:
        # ── Step 1: PDF Parsing ────────────────────────────────────────────
        task["status"] = "parsing"
        logger.info("[%s] STEP 1/3 → Docling PDF parsing started", task_id)
        logger.info("[%s] PDF file: %s (%.2f MB)", task_id,
                    task["file_path"], task["size_bytes"] / 1_048_576)

        await put("progress", {
            "step": "parsing",
            "message": "🔍 Parsing PDF with Docling… This takes 1–5 min for a 300-page RHP. Hang tight!",
        })

        parse_start = time.time()
        parser = RHPParser(task["file_path"])
        sections = await asyncio.to_thread(parser.extract_sections)
        parse_elapsed = time.time() - parse_start
        timings["parse_done"] = time.time()

        # Save Docling markdown output for inspection
        md_path = Path(task["parsed_md_path"])
        if hasattr(parser, "_last_markdown") and parser._last_markdown:
            md_path.write_text(parser._last_markdown, encoding="utf-8")
            logger.info("[%s] Docling markdown saved → %s (%d chars)", task_id,
                        md_path, len(parser._last_markdown))
        else:
            logger.info("[%s] Docling markdown not available to save (no _last_markdown attr)", task_id)

        company = sections.get("company_name", "Unknown Company")
        pages = sections.get("total_pages", "?")
        found_keys = [k for k in sections if k not in ("total_pages", "company_name") and sections[k]]
        section_sizes = {k: len(sections[k]) for k in found_keys}

        logger.info("[%s] STEP 1/3 DONE in %.1fs", task_id, parse_elapsed)
        logger.info("[%s] Company: %s | Pages: %s | Sections found: %d",
                    task_id, company, pages, len(found_keys))
        for sec, size in section_sizes.items():
            logger.info("[%s]   %-20s → %s chars", task_id, sec, f"{size:,}")

        await put("progress", {
            "step": "parsed",
            "message": f"✅ Parsed {pages} pages of {company}. Found {len(found_keys)} sections.",
            "company_name": company,
            "total_pages": pages,
            "sections_found": found_keys,
            "section_sizes": section_sizes,
            "parse_seconds": round(parse_elapsed, 1),
        })

        # ── Step 2: Groq LLM Analysis ──────────────────────────────────────
        task["status"] = "analyzing"
        logger.info("[%s] STEP 2/3 → Groq LLM analysis started (7 sections + verdict)", task_id)

        groq_key = normalize_groq_key(task.get("groq_key") or os.environ.get("GROQ_API_KEY", ""))
        analyzer = IPOAnalyzer(sections, groq_key=groq_key)

        analysis_start = time.time()
        async for event in analyzer.analyze_stream():
            await queue.put(event)
        analysis_elapsed = time.time() - analysis_start
        timings["analysis_done"] = time.time()

        logger.info("[%s] STEP 2/3 DONE in %.1fs", task_id, analysis_elapsed)

        # ── Step 3: Save Report ────────────────────────────────────────────
        logger.info("[%s] STEP 3/3 → Saving report to disk", task_id)
        task["report"] = analyzer.all_analyses
        task["status"] = "done"

        report_path = UPLOAD_DIR / f"{task_id}_report.json"
        report_path.write_text(json.dumps(analyzer.all_analyses, indent=2, ensure_ascii=False))
        logger.info("[%s] Report saved → %s", task_id, report_path)

        total_elapsed = time.time() - pipeline_start
        logger.info("[%s] ── PIPELINE COMPLETE in %.1fs ──────────────", task_id, total_elapsed)
        logger.info("[%s] Timings: parse=%.1fs | analysis=%.1fs | total=%.1fs",
                    task_id,
                    parse_elapsed,
                    analysis_elapsed,
                    total_elapsed)

        await put("done", {
            "message": "Analysis complete.",
            "total_seconds": round(total_elapsed, 1),
        })

    except Exception as exc:
        logger.exception("[%s] !! PIPELINE FAILED: %s", task_id, exc)
        await put("error", {"message": str(exc)})
        task["status"] = "error"

    finally:
        await queue.put(None)  # Sentinel
        logger.info("[%s] Queue sentinel sent — SSE stream will close", task_id)
