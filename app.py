"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph

load_dotenv()


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HITL PR Review Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for modern, premium styling ────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hero header */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin: 0;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-langgraph { background: #3b82f620; color: #60a5fa; border: 1px solid #3b82f640; }
    .badge-sqlite    { background: #10b98120; color: #34d399; border: 1px solid #10b98140; }
    .badge-hitl      { background: #f59e0b20; color: #fbbf24; border: 1px solid #f59e0b40; }

    /* Severity tags */
    .severity-pill {
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .sev-blocker    { background: #ef444422; color: #f87171; border: 1px solid #ef444455; }
    .sev-issue      { background: #f9731622; color: #fb923c; border: 1px solid #f9731655; }
    .sev-suggestion { background: #3b82f622; color: #60a5fa; border: 1px solid #3b82f655; }
    .sev-nit        { background: #64748b22; color: #94a3b8; border: 1px solid #64748b55; }

    /* Risk tag */
    .risk-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #ef444415;
        border: 1px solid #ef444440;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        margin: 3px;
    }

    /* Button overrides */
    div[data-testid="stButton"] > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Hero Banner ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">
            <span>🛡️ HITL Pull Request Review Agent</span>
            <span class="badge-pill badge-langgraph">LangGraph v1.2</span>
            <span class="badge-pill badge-hitl">Human-in-the-Loop</span>
            <span class="badge-pill badge-sqlite">SQLite Audit</span>
        </div>
        <p class="hero-subtitle">
            Autonomous PR triage with confidence routing: <b>Auto-Approve (&gt;72%)</b>, 
            <b>Human Approval (58–72%)</b>, and <b>Escalation Clarifications (&lt;58%)</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── Sidebar — recent sessions & stats ─────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Audit Timeline")
    st.caption("Live session history from SQLite `./hitl_audit.db`")

    async def fetch_recent_sessions():
        async with db_conn() as conn:
            async with conn.execute(
                """
                SELECT thread_id,
                       pr_url,
                       MIN(timestamp)        AS started,
                       MAX(timestamp)        AS last_event,
                       MAX(risk_level)       AS worst_risk,
                       COUNT(*)              AS events
                  FROM audit_events
                 GROUP BY thread_id, pr_url
                 ORDER BY MAX(timestamp) DESC
                 LIMIT 20
                """
            ) as cur:
                return await cur.fetchall()

    try:
        sessions = asyncio.run(fetch_recent_sessions())
        if sessions:
            st.metric("Total Recorded Sessions", len(sessions))
            st.divider()
            for s in sessions:
                tid = s["thread_id"]
                short_tid = tid[:8]
                pr_num = s["pr_url"].rstrip("/").split("/")[-1]
                risk = s["worst_risk"]
                badge = "🟢 Low" if risk == "low" else ("🟡 Med" if risk == "med" else "🔴 High")
                
                with st.container():
                    col_btn, col_info = st.columns([3, 2])
                    if col_btn.button(f"PR #{pr_num} ({short_tid})", key=f"session_{tid}", use_container_width=True):
                        st.session_state.thread_id = tid
                        st.session_state.pr_url = s["pr_url"]
                        st.session_state.interrupt_payload = None
                        st.session_state.final = None
                        st.rerun()
                    col_info.caption(f"{badge}\n`{s['events']} evts`")
        else:
            st.info("No audit sessions yet. Run a review to populate!")
    except Exception as e:
        st.caption(f"Waiting for database connection: {e}")

    st.divider()
    st.caption("Developed for Day 27 HITL Lab · VinUni AI")


# ─── Top Form: Start a new review & Quick Demos ────────────────────────────
st.markdown("##### 🚀 Run Code Review")
col_input, col_submit = st.columns([5, 1])

with col_input:
    pr_url = st.text_input(
        "GitHub Pull Request URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
        label_visibility="collapsed",
    )

with col_submit:
    submitted = st.button("🔍 Run Review", type="primary", use_container_width=True)

# Quick Demo helper buttons
col_d1, col_d2, col_d3 = st.columns([2, 2, 3])
if col_d1.button("📋 Load Demo PR #1 (Approval)", use_container_width=True):
    st.session_state.pr_url = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"
    st.rerun()

if col_d2.button("⚠️ Load Demo PR #2 (Escalation)", use_container_width=True):
    st.session_state.pr_url = "https://github.com/VinUni-AI20k/PR-Demo/pull/2"
    st.rerun()

st.divider()


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    
    st.markdown(
        f"""
        <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: #fbbf24;">✋ Human Approval Required</h3>
                <span style="font-size: 20px; font-weight: 800; color: #fbbf24;">Confidence: {conf:.0%}</span>
            </div>
            <p style="margin: 8px 0 0 0; color: #cbd5e1; font-size: 14px;">{payload['confidence_reasoning']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(conf, text=f"Model Confidence: {conf:.0%}")

    st.markdown("#### 📝 Summary of Changes")
    st.info(payload["summary"])

    comments = payload.get("comments", [])
    if comments:
        st.markdown(f"#### 💬 Automated Findings ({len(comments)})")
        for c in comments:
            sev = c.get("severity", "suggestion")
            sev_class = f"sev-{sev}"
            line_str = f":{c['line']}" if c.get("line") else ""
            st.markdown(
                f"""
                <div style="padding: 10px 14px; background: rgba(255,255,255,0.02); border-left: 3px solid #64748b; margin-bottom: 8px; border-radius: 4px;">
                    <span class="severity-pill {sev_class}">{sev}</span>
                    <code>{c['file']}{line_str}</code>
                    <div style="margin-top: 6px; color: #e2e8f0; font-size: 14px;">{c['body']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("🔍 View Raw Diff Preview", expanded=False):
        st.code(payload.get("diff_preview", ""), language="diff")

    st.markdown("#### 🎯 Reviewer Decision")
    feedback = st.text_input(
        "Optional Reviewer Notes / Feedback:",
        key="approval_feedback",
        placeholder="e.g. Looks good to merge after squashing commits...",
    )

    col1, col2, col3 = st.columns(3)
    if col1.button("✅ Approve & Post Review", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("❌ Reject Review", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("✏️ Request Changes / Edit", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    
    st.markdown(
        f"""
        <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: #f87171;">🚨 Strong Escalation Triggered</h3>
                <span style="font-size: 20px; font-weight: 800; color: #f87171;">Confidence: {conf:.0%}</span>
            </div>
            <p style="margin: 8px 0 0 0; color: #cbd5e1; font-size: 14px;">{payload['confidence_reasoning']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(conf, text=f"Model Confidence: {conf:.0%}")

    risks = payload.get("risk_factors", [])
    if risks:
        st.markdown("#### ⚠️ Highlighted Risk Factors")
        risk_html = "".join([f'<span class="risk-tag">⚠️ {r}</span>' for r in risks])
        st.markdown(f'<div style="margin-bottom: 16px;">{risk_html}</div>', unsafe_allow_html=True)

    st.markdown("#### 📝 PR Summary")
    st.warning(payload["summary"])

    st.markdown("#### ❓ Clarifying Questions for Reviewer")
    st.caption("The agent needs human answers to these questions before it can formulate an accurate final review:")

    with st.form("escalation_form"):
        answers: dict[str, str] = {}
        for idx, q in enumerate(payload.get("questions", [])):
            st.markdown(f"**Q{idx+1}:** {q}")
            answers[q] = st.text_input(
                f"Your Answer for Q{idx+1}",
                key=f"q_{idx}",
                placeholder="Enter context, mitigation, or design intent...",
                label_visibility="collapsed",
            )
            st.write("")
        
        submitted_answers = st.form_submit_button(
            "🚀 Submit Answers & Re-Synthesize Review",
            type="primary",
            use_container_width=True,
        )
        if submitted_answers:
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR & analyzing diff with Senior Reviewer LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming graph & synthesizing review..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    
    st.markdown("### 🏁 Final Execution Result")
    
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"🎉 **Action Completed: `{action}`** — Comment successfully posted to GitHub!")
        
        col_res1, col_res2 = st.columns([1, 3])
        if "analysis" in final:
            a = final["analysis"]
            col_res1.metric("Final Confidence", f"{a.confidence:.0%}")
            col_res2.markdown(f"**Review Summary:**\n{a.summary}")
            
        st.markdown(f"🔗 [**View Your Live Comment on GitHub**]({st.session_state.pr_url})")
    elif action == "rejected":
        st.warning("⚠️ **Review Rejected by Human** — No comment was posted to the PR.")
    else:
        st.info(f"**Final Status:** `{action}`")
        
    st.caption(
        f"Session Thread ID: `{st.session_state.thread_id}` · "
        f"Audit Replay Command: `uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )
