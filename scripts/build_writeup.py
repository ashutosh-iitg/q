"""Build the standalone HTML write-up and its PDF from one list of pages.

Run with: uv run --no-project --with reportlab --with pillow python scripts/build_writeup.py
The PDF libraries are export-only dependencies, not dependencies of Q.
"""

from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "architecture"
PDF = ROOT / "output" / "pdf" / "q-architecture-writeup.pdf"

TITLE = "Q: a family health agent"
DEK = "One shared conversation, many patients, and a safe next step on every turn."
EYEBROW = "Architecture write-up · 30 October 2026"

# A page is (eyebrow, heading, blocks). A block is a tuple whose first item names its kind.
PAGES = [
    (EYEBROW, "1 · The problem", [
        ("p", "A family shares one account, but the person typing is often not the patient. A parent reports a "
              "child's fever, switches to a grandparent's blood pressure, and returns the next day with \"she's "
              "better\". Q must attach every message to the right person and the right illness episode, and carry "
              "that context forward so nobody repeats themselves."),
        ("p", "The goal is the shortest safe path to the right next step: a clear answer, focused questions asked "
              "one at a time, home care within a supported protocol, or a doctor or emergency services. Q is not a "
              "diagnosis engine, and familiar history is never a reason to dismiss a new symptom."),
        ("h3", "Scope"),
        ("p", "English text; general questions; symptom intake; follow-ups across sessions; fixed emergency and "
              "doctor replies; a structured doctor intake note. One family member per message. Knowledge base, lab "
              "results and telemedicine plug in through interfaces that have no implementation yet. No clinician "
              "is contacted."),
        ("h3", "Models interpret, code decides"),
        ("p", "Models handle what language makes uncertain: who a message is about, which episode it continues, "
              "what it reports, and how to word each question and reply. Code holds authority: identity and "
              "ownership checks, which finding to ask about next, the outcome, fixed escalation text, durable "
              "writes, and what a doctor note may claim. Every model output is typed and checked before it is used."),
        ("dl", [
            ("Know", "The selected member's history and the triage protocol for the complaint."),
            ("Act", "One focused question, an answer, home monitoring, or the fixed escalation reply."),
            ("Remember", "Episodes across sessions; each finding with its quote, source and status."),
            ("Coordinate", "LangGraph: explicit transitions, bounded tool loops, traced outcomes."),
        ]),
    ]),
    ("Architecture", "2 · How one turn flows", [
        ("caption", "The outer loop advances each time the user replies. Inside it, each agent looks things up "
                    "before submitting a typed result that code validates."),
        ("figure",),
    ]),
    ("Design", "3 · Key decisions, and why", [
        ("table", ("Decision", "What I built, and why", "What would change it"), [
            ("Member before episode", "Resolve who the message is about, then pass that member to the episode "
             "agent, whose tools only see that member's episodes.",
             "Merge the two agents if identity accuracy holds while calls fall."),
            ("The protocol guides, the agent asks", "Protocols list findings and their urgency; code picks the next "
             "finding, most urgent first; the intake agent words one question from working memory. Verbatim "
             "protocol text had asked a self-reporting user \"Are they struggling to breathe?\".",
             "A model check that the wording still asks the chosen finding, if evaluation shows drift."),
            ("Disposition lives in code", "An outcome counts only when it is sufficient: a positive answer at its "
             "level and every more urgent finding ruled out. Emergency and doctor text is fixed.",
             "Versioned clinical rules, tested for missed and unnecessary escalation separately."),
            ("Findings carry time and source", "Each answer is present, absent or resolved, with the user's quote, "
             "the question Q asked and the session. Other symptoms of the illness are stored as active or resolved, "
             "so a fever that settled is never treated as current.",
             "A typed observations table when trends over readings are needed."),
            ("The note is verified, not trusted", "Each history sentence must quote a user message. Q's words and bare "
             "replies are not evidence. Denials and the reporter come from code, and every item says which "
             "conversation reported it and when.",
             "A narrow judge model that checks each sentence means what its quote says."),
            ("Small harness, durable turns", "One shared tool loop: three lookup rounds, then a forced answer. One "
             "atomic PostgreSQL save per turn with an idempotent turn ID, so a retry replays without model calls.",
             "A heavier harness only for long investigations that need planning."),
        ]),
        ("h3", "Memory model"),
        ("p", "PostgreSQL holds four layers. Working memory: session state and messages, written every turn. "
              "Episodes: span sessions and record reports, never diagnoses, with findings and related symptoms. "
              "Member facts: quoted, moving from provisional to consolidated to verified, where only seed or "
              "clinician sources verify. Summaries: one checked call per closed session. Identity and episode "
              "lookups are exact, so there is no vector store."),
    ]),
    ("Evaluation", "4 · How Q is evaluated", [
        ("p", "Each failure should be caught by the cheapest layer that can catch it, and every failure seen live "
              "becomes a test in a cheaper layer."),
        ("labelled", [
            ("Rules, no model", "Policy order, triage sufficiency, protocol validation, quote matching and the note's "
             "evidence rules."),
            ("Graph with a scripted model", "The real LangGraph and API on PostgreSQL, with a scripted output per agent "
             "role: routing, persistence, replay, scoped tools and every fallback. 194 tests run in about 3 seconds."),
            ("Live scenarios", "The real model on synthetic families, saved as transcripts and read together with "
             "their traces and stored state."),
            ("Hard failures", "Writes to the wrong patient, missed escalation, and a note claiming something "
             "unreported. They are counted on their own, never averaged against fluent replies. Missed and "
             "unnecessary escalations are tracked separately, so escalating everything cannot look good."),
        ]),
        ("h3", "Failures found, and what each one changed"),
        ("table", ("What we saw", "Root cause", "Fix, and the guard that keeps it fixed"), [
            ("A self-report was asked \"Are they struggling to breathe?\"; \"Who is they?\" got the same question.",
             "Protocol text sent verbatim. The clarification was routed to the general answer.",
             "The agent words questions. A clarification intent keeps the intake going. Tests cover the payload "
             "and the routing."),
            ("\"With the cough and fever you mentioned\", after the fever had gone.",
             "Findings had no time. \"I had fever initially\" was stored as present; \"which subsided\" was lost.",
             "Resolved answers and related symptoms with status. Below urgent levels, triage treats resolved as "
             "history."),
            ("A note said \"denies blue lips\" for a chat that never asked.",
             "A real answer from an earlier chat was reused silently. Quote checks accepted Q's words and a bare \"No\".",
             "Each answer records where it came from. Emergency answers are asked again in a new chat. The note "
             "guardrail and code-written denials."),
            ("A stray \"No\" after a handoff raised the urgency.",
             "An already escalated case falls back to the default urgency.",
             "Open: store the urgency on the episode."),
        ]),
        ("h3", "Limits"),
        ("p", "These are engineering checks, not clinical validation: protocols are synthetic placeholders, and code "
              "can check that a quote exists, not that a sentence means what it says. Next: repeated paraphrase "
              "trials, a narrow judge for meaning, and clinician review of protocols and notes."),
    ]),
    ("Observability", "5 · Seeing why Q did what it did", [
        ("labelled", [
            ("One trace per turn", "Each message is a LangSmith trace tagged with household, session and turn; the "
             "session groups the turns into a thread. Each model call is named after its agent and step, tool calls "
             "are nested with arguments, results and errors, and the trace's output is the outcome: decision, "
             "reason, urgency, member, episode and protocol."),
            ("Reasons, not guesses", "Every decision carries the rule that produced it (protocol, "
             "clarification_requested, already_escalated), so a trace answers why without rereading prompts."),
            ("State you can read", "Findings keep their quote, Q's question and the session; episodes keep related "
             "symptoms; notes keep reported_in and the statements the guardrail dropped. In each investigation above, "
             "the stored state found the cause: a fever stored as present with a truncated quote, and message "
             "session IDs that showed a carried answer."),
            ("Privacy", "Traces include conversations, which are synthetic here. Input and output hiding is one "
             "setting for real data."),
        ]),
        ("h3", "What to measure next"),
        ("table", ("Area", "Signals", "Why it matters"), [
            ("Safety", "Missed and unnecessary escalations; wrong-member writes", "The failures that must not be "
             "averaged away"),
            ("Faithfulness", "Dropped note statements; agent wording that fell back to the protocol",
             "A rising rate means a prompt or model regressed"),
            ("Conversation", "Clarifications, repeated questions, turns to an outcome", "Effort for a worried family"),
            ("Cost", "Model calls per turn, tokens, p50 and p95 latency", "Model calls dominate both cost and delay"),
        ]),
    ]),
    ("Roadmap", "6 · Where Q goes next", [
        ("labelled", [
            ("Make safety independent", "A red-flag screen, backed by a keyword check, that replies before identity "
             "is resolved and even when the model API fails. Licensed, clinician-owned protocols."),
            ("A care companion", "Proactive check-backs, a profile picker, several members in one message, care plans "
             "that clinicians write and Q follows, remembered preferences, localized and accessible intake."),
            ("Connect to real care", "The knowledge-base, lab and telemedicine hooks already in the graph. Delivery "
             "through an outbox with an idempotency key. A clinician view that receives the case and, once accepted, "
             "opens or updates the episode."),
            ("Scale", "300k conversations a day is about 28 turns a second on average. A database read is under 1% of "
             "a turn, so model calls set cost and latency. So: stateless workers, model concurrency limits per "
             "household, routing models by measured gain, Redis only as a write-through cache, a durable job queue, "
             "and ordered turns per session."),
        ]),
        ("closing", "As Q grows, models get more to read, never more authority over the outcome or over what a note "
                    "may claim."),
    ]),
    ("Try it", "7 · Running the demo", [
        ("p", "Python 3.13+, uv, Docker, and an OpenAI API key for a model that supports structured output."),
        ("steps", [
            ("Configure. Set POSTGRES_PASSWORD, a matching DATABASE_URL, OPENAI_API_KEY and Q_MODEL_MINI.",
             "cp .env.example .env"),
            ("Install dependencies and start PostgreSQL.", "uv sync\ndocker compose up -d"),
            ("Start the API. It creates the schema and seeds 10 synthetic families.",
             "uv run --env-file .env uvicorn api.main:app --reload"),
            ("In a second terminal, start the chat app.", "uv run --env-file .env streamlit run streamlit_app.py"),
            ("Open http://localhost:8501, choose a family and the profile you are writing as, and chat. New chat "
             "closes the session into family memory. The sidebar exports the chat as Markdown or JSON; doctor "
             "replies show the intake note with where each item was reported.", None),
        ]),
        ("h3", "Checks"),
        ("bullets", [
            ("Run the tests (database tests skip without DATABASE_URL).", "uv run --env-file .env pytest -q"),
            ("Replay the scenarios with the real model and save transcripts.",
             "uv run --env-file .env python -m scripts.run_scenarios"),
            ("Optional: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env to trace every turn.", None),
        ]),
    ]),
]

CSS = """
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#edf2f5;color:#182d3d;font:16px/1.6 system-ui,sans-serif}
main{max-width:960px;margin:auto;padding:30px 20px}.sheet{background:#fff;padding:42px 48px;margin:0 0 24px}
.eyebrow{color:#35628a;font-size:12px;letter-spacing:.14em;text-transform:uppercase;font-weight:700}
h1{font-size:40px;line-height:1.12;margin:12px 0 10px}h2{font-size:27px;line-height:1.2;margin:10px 0 20px}
h3{font-size:18px;margin:24px 0 8px}p{margin:0 0 14px}.dek{font-size:19px;color:#426077;margin-bottom:28px}
.caption{font-size:13px;color:#526575}.closing{border-left:3px solid #35628a;padding-left:16px;font-weight:600}
dl{display:grid;grid-template-columns:150px 1fr;gap:8px 16px}dt{font-weight:700}dd{margin:0}
table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.45;margin-bottom:8px}
th,td{padding:9px 9px;vertical-align:top;text-align:left;border-bottom:1px solid #dce4ea}thead th{color:#35628a}
tbody th{width:22%}ul,ol{padding-left:22px}li{margin-bottom:10px}
pre{background:#f2f6f9;border:1px solid #dce4ea;border-radius:6px;padding:10px 12px;margin:6px 0 0;font-size:13px;white-space:pre-wrap;overflow-wrap:anywhere}
.diagram svg{display:block;width:100%;height:auto}.diagram{margin:0}
@media(max-width:600px){main{padding:12px}.sheet{padding:24px 16px}h1{font-size:31px}h2{font-size:23px}dl{grid-template-columns:1fr;gap:0}dd{margin:0 0 12px}
table,thead,tbody,tr,th,td{display:block;width:100%}thead{display:none}tbody th{width:100%;padding-bottom:0;border:0}td{padding-top:5px}}
@page{size:A4;margin:15mm}@media print{body{background:white;font-size:10pt;line-height:1.45}main{max-width:none;padding:0}
.sheet{padding:0;margin:0;break-after:page}.sheet:last-child{break-after:auto}h1{font-size:29pt}h2{font-size:20pt}h3{font-size:12pt;margin-top:14px}
.dek{font-size:13pt}table{font-size:8.5pt}th,td{padding:6px}.diagram svg{max-height:235mm}tr,li{break-inside:avoid}}
"""


def html_block(block: tuple, svg: str) -> str:
    kind, *args = block
    if kind == "p":
        return f"<p>{escape(args[0])}</p>"
    if kind == "h3":
        return f"<h3>{escape(args[0])}</h3>"
    if kind == "caption":
        return f"<p class='caption'>{escape(args[0])}</p>"
    if kind == "closing":
        return f"<p class='closing'>{escape(args[0])}</p>"
    if kind == "figure":
        return f"<figure class='diagram'>{svg}</figure>"
    if kind in ("dl", "labelled"):
        return "<dl>" + "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in args[0]) + "</dl>"
    if kind == "table":
        headers, rows = args
        head = "".join(f"<th>{escape(h)}</th>" for h in headers)
        body = "".join("<tr><th scope='row'>" + escape(r[0]) + "</th>" + "".join(f"<td>{escape(c)}</td>" for c in r[1:])
                       + "</tr>" for r in rows)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    tag = "ol" if kind == "steps" else "ul"
    items = "".join(f"<li>{escape(text)}" + (f"<pre>{escape(code)}</pre>" if code else "") + "</li>"
                    for text, code in args[0])
    return f"<{tag}>{items}</{tag}>"


def build_html() -> None:
    svg = (OUT / "q-agent-loop.svg").read_text().replace(
        'width="1000" height="1360"', 'role="img" aria-label="Q agent workflow and bounded lookup loop"')
    sheets = []
    for number, (eyebrow, heading, blocks) in enumerate(PAGES):
        hero = f"<h1>{escape(TITLE)}</h1><p class='dek'>{escape(DEK)}</p>" if number == 0 else ""
        body = "".join(html_block(b, svg) for b in blocks)
        sheets.append(f'<section class="sheet"><div class="eyebrow">{escape(eyebrow)}</div>{hero}'
                      f"<h2>{escape(heading)}</h2>{body}</section>")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; base-uri 'none'; form-action 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'none'">
<title>Q Architecture Write-up</title>
<style>{CSS}</style></head><body><main>
{''.join(sheets)}
</main></body></html>"""
    (OUT / "q-agent-loop.html").write_text(html)


def build_pdf() -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                    Table, TableStyle)

    PDF.parent.mkdir(parents=True, exist_ok=True)
    ink, muted, accent, rule, wash = map(colors.HexColor, ("#182d3d", "#526575", "#35628a", "#dce4ea", "#f2f6f9"))
    copy = ParagraphStyle("Copy", fontName="Helvetica", fontSize=9.8, leading=13.8, textColor=ink, spaceAfter=8)
    styles = {
        "Copy": copy,
        "Small": ParagraphStyle("Small", parent=copy, fontSize=8.6, leading=12, textColor=muted),
        "Cell": ParagraphStyle("Cell", parent=copy, fontSize=8.3, leading=11.2, spaceAfter=0),
        "Hero": ParagraphStyle("Hero", parent=copy, fontName="Helvetica-Bold", fontSize=29, leading=33, spaceAfter=8),
        "Dek": ParagraphStyle("Dek", parent=copy, fontSize=13, leading=18, textColor=accent, spaceAfter=16),
        "H": ParagraphStyle("H", parent=copy, fontName="Helvetica-Bold", fontSize=19, leading=23, spaceAfter=10),
        "Sub": ParagraphStyle("Sub", parent=copy, fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                              spaceBefore=8, spaceAfter=5, keepWithNext=True),
        "Eyebrow": ParagraphStyle("Eyebrow", parent=copy, fontName="Helvetica-Bold", fontSize=8, textColor=accent),
        "Closing": ParagraphStyle("Closing", parent=copy, fontName="Helvetica-Bold", spaceBefore=6),
        "Code": ParagraphStyle("Code", fontName="Courier", fontSize=8.6, leading=11.2, textColor=ink, leftIndent=18,
                               spaceBefore=3, spaceAfter=10),
    }

    def para(text: str, style: str = "Copy") -> Paragraph:
        return Paragraph(escape(text), styles[style])

    def table(headers: tuple, rows: list) -> Table:
        widths = [100, 210, 195] if headers[0] == "Decision" else [120, 185, 200]
        data = [[para(h, "Cell") for h in headers]]
        data += [[Paragraph(f"<b>{escape(r[0])}</b>", styles["Cell"]), *(para(c, "Cell") for c in r[1:])] for r in rows]
        result = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
        result.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), wash),
                                    ("LINEBELOW", (0, 0), (-1, -1), .5, rule),
                                    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                                    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        return result

    def items(entries: list, numbered: bool) -> list:
        flow = []
        for number, (text, code) in enumerate(entries, 1):
            block = [Paragraph(f"{f'{number}.' if numbered else '•'}&nbsp; {escape(text)}", copy)]
            if code:
                block.append(Preformatted(code, styles["Code"]))
            flow.append(KeepTogether(block))
        return flow

    def pdf_block(block: tuple) -> list:
        kind, *args = block
        if kind == "p":
            return [para(args[0])]
        if kind == "h3":
            return [para(args[0], "Sub")]
        if kind == "caption":
            return [para(args[0], "Small")]
        if kind == "closing":
            return [para(args[0], "Closing")]
        if kind == "figure":
            return [Image(str(OUT / "q-agent-loop.png"), width=470, height=639.2)]
        if kind in ("dl", "labelled"):
            return [Paragraph(f"<b>{escape(k)}.</b> {escape(v)}", copy) for k, v in args[0]]
        if kind == "table":
            return [table(*args)]
        return items(args[0], numbered=kind == "steps")

    story = []
    for number, (eyebrow, heading, blocks) in enumerate(PAGES):
        if number:
            story.append(PageBreak())
        story.append(para(eyebrow.upper(), "Eyebrow"))
        if number == 0:
            story += [para(TITLE, "Hero"), para(DEK, "Dek")]
        story.append(para(heading, "Sub" if number == 0 else "H"))
        for block in blocks:
            story += pdf_block(block)

    def frame(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(muted)
        canvas.drawString(45, 25, TITLE)
        canvas.drawRightString(A4[0] - 45, 25, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=45, leftMargin=45, topMargin=38, bottomMargin=42,
                            title="Q Architecture Write-up", author="Ashutosh", pageCompression=1)
    doc.build(story, onFirstPage=frame, onLaterPages=frame)


if __name__ == "__main__":
    build_html()
    build_pdf()
    print("Updated output/architecture/q-agent-loop.html")
    print("Created output/pdf/q-architecture-writeup.pdf")
