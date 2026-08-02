from app.main import build_download_report
from app.analyzer import normalize_groq_key


def test_build_download_report_includes_verdict_and_sections():
    report = {
        "verdict": {
            "verdict": "INVEST",
            "overall_score": 8.4,
            "verdict_summary": "Strong fundamentals and credible use of proceeds.",
            "one_liner": "This IPO looks attractive for long-term investors.",
        },
        "objects_of_issue": {
            "label": "Objects of the Issue",
            "score": 7,
            "summary": "The company plans to deploy funds into capex and growth.",
        },
    }

    text = build_download_report(report)

    assert "# IPO Analysis Report" in text
    assert "INVEST" in text
    assert "Objects of the Issue" in text
    assert "Strong fundamentals and credible use of proceeds." in text


def test_normalize_groq_key_strips_whitespace():
    assert normalize_groq_key("  abc123  ") == "abc123"
    assert normalize_groq_key(None) == ""
