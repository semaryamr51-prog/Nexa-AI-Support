"""Four specialist agents, routed by the LSTM with grounded tools and history.

Agents are deliberately deterministic in offline mode. When Groq/Ollama is
configured, each specialist can augment its grounded result through an LLM.
"""
from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Optional

from src.classify import classify
from src.knowledge import Passage, search
from src.llm import generate_answer
from src.mock_backends import mock_billing_db, mock_calendar

CONFIDENCE_FALLBACK_THRESHOLD = 0.45
VALID_INTENTS = {"billing", "technical", "booking", "general"}


def _invoice_id(message: str) -> str | None:
    match = re.search(r"\bINV[-\s]?(\d{4,8})\b", message, flags=re.I)
    return f"INV-{match.group(1)}" if match else None


def _date_phrase(message: str) -> str | None:
    lower = message.lower()
    patterns = [r"\bnext\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\bthis\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today)\b",
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}\b",
        r"\b\d{4}-\d{2}-\d{2}\b"]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            return message[match.start():match.end()]
    return None


def _summarize_passage(p: Passage) -> str:
    text = " ".join(p.text.split())
    # Keep the essential policy sentences without dumping entire documents.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return " ".join(sentences[:3])[:650]




def _retrieve(query: str, intent: str, limit: int = 2) -> list[Passage]:
    """Use semantic retrieval when explicitly enabled; otherwise local section search."""
    if os.getenv("RAG_BACKEND", "lexical").lower() != "semantic":
        return search(query, intent=intent, limit=limit)
    from src.rag import get_retriever
    docs = get_retriever(k=8).invoke(query)
    passages = []
    wanted = 'booking_policy.md' if intent == 'booking' else f'{intent}_faq.md'
    for doc in docs:
        source = Path(str(doc.metadata.get('source', ''))).name
        if intent != 'general' and source != wanted:
            continue
        content = doc.page_content.strip()
        heading = re.search(r"(?m)^## (.+)$", content)
        title = heading.group(1).strip() if heading else 'Relevant documentation'
        if content:
            passages.append(Passage(title, content, source))
        if len(passages) == limit:
            break
    return passages or search(query, intent=intent, limit=limit)


def _specialist_response(intent: str, user_message: str):
    """Tool execution is explicit, auditable, and independent of model output."""
    calls: list[str] = []
    sources: list[str] = []
    details = []
    snippets = []
    invoice = _invoice_id(user_message) if intent == "billing" else None
    if invoice:
        calls.append("check_invoice")
        details.append(mock_billing_db.get_invoice(invoice))
    if intent == "booking":
        date = _date_phrase(user_message)
        if date:
            calls.append("check_availability")
            details.append(mock_calendar.check(date))
    passages = _retrieve(user_message, intent=intent, limit=2)
    calls.append("search_docs")
    for p in passages:
        sources.append(f"{p.source} · {p.title}")
        snippets.append(_summarize_passage(p))
    if details:
        lead = "\n\n".join(details)
    elif snippets:
        lead = {"billing": "Here's what the billing policy says:",
                "technical": "Here's the recommended troubleshooting guidance:",
                "booking": "Here's the session policy:",
                "general": "Here's what the knowledge base says:"}[intent]
    else:
        lead = {"billing": "I can explain the payment policy. Please share a demo invoice ID (for example, INV-2091) if you want a sample lookup.",
                "technical": "Please describe the error, device, and what you tried. I couldn't find a specific troubleshooting article for that request.",
                "booking": "Which day would you like to check? Availability here is simulated and I can't create bookings.",
                "general": "I don't have a verified answer to that in the demo knowledge base. Please contact the platform support team for details."}[intent]
    if snippets:
        lead += "\n\n" + "\n\n".join(f"**{p.title}** — {snippet}" for p, snippet in zip(passages, snippets))
    if sources:
        lead += "\n\n*Sources: " + "; ".join(sources) + "*"
    context = "\n\n".join(details + [f"Source: {p.source} | {p.title}\n{p.text}" for p in passages])
    return lead, context, calls, sources


def route_and_run(user_message: str, history: Optional[list] = None, max_history_turns: int = 6) -> dict:
    """Classify -> route -> specialist tools -> optional LLM -> structured trace."""
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("Please enter a message.")
    intent, confidence = classify(user_message[:4000])
    agent = intent if intent in VALID_INTENTS and confidence >= CONFIDENCE_FALLBACK_THRESHOLD else "general"
    offline_reply, context, calls, sources = _specialist_response(agent, user_message[:4000])
    bounded = (history or [])[-max_history_turns * 2:]
    reply, provider = generate_answer(agent, user_message[:4000], bounded, context, offline_reply)
    return {"reply": reply, "intent": intent, "confidence": confidence,
            "agent": agent, "tool_calls": calls, "sources": sources,
            "provider": provider}
