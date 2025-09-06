import re
from typing import Dict, Any
import nltk

# Ensure VADER lexicon is available at runtime
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except:
    nltk.download('vader_lexicon')
    from nltk.sentiment import SentimentIntensityAnalyzer

SIA = SentimentIntensityAnalyzer()

FILTER_TERMS = {"support", "query", "request", "help"}
URGENT_MARKERS = {
    "urgent", "immediate", "immediately", "asap", "critical", "blocked",
    "cannot access", "can’t access", "unable to login", "system down"
}
CATEGORY_RULES = {
    "Billing Issue": {"billing", "invoice", "payment", "pricing", "charge"},
    "Account Issue": {"account", "login", "password", "reset", "verify"},
    "Technical/Integration": {"api", "integration", "sdk", "token", "webhook"},
}

EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[-\s]?)?\d{10}")

def normalize(text: str) -> str:
    return (text or "").strip()

def is_filtered(subject: str) -> bool:
    s = subject.lower()
    return any(term in s for term in FILTER_TERMS)

def sentiment_label(text: str) -> str:
    scores = SIA.polarity_scores(text or "")
    comp = scores["compound"]
    if comp >= 0.2:
        return "Positive"
    if comp <= -0.2:
        return "Negative"
    return "Neutral"

def priority_label(subject: str, body: str) -> str:
    blob = f"{subject} {body}".lower()
    return "Urgent" if any(k in blob for k in URGENT_MARKERS) else "Not urgent"

def category_label(subject: str, body: str) -> str:
    blob = f"{subject} {body}".lower()
    for cat, kws in CATEGORY_RULES.items():
        if any(k in blob for k in kws):
            return cat
    return "General Query"

def extract_info(text: str) -> Dict[str, Any]:
    emails = EMAIL_PATTERN.findall(text or "")
    phones = PHONE_PATTERN.findall(text or "")
    # Basic “requirement/ask” extraction: pull sentences with verbs like need/want/cannot
    asks = []
    for sent in re.split(r'(?<=[.!?])\s+', text or ""):
        ls = sent.lower()
        if any(k in ls for k in ["need", "want", "require", "cannot", "can't", "unable", "error", "issue"]):
            asks.append(sent.strip())
    return {
        "emails_found": list(set(emails)),
        "phones_found": ["".join(p) if isinstance(p, tuple) else p for p in phones],
        "possible_requirements": asks[:3]  # keep it short
    }

# --- LLM Reply (optional) ---

def generate_reply(subject: str, body: str, category: str, sentiment: str, priority: str,
                   product_hint: str | None = None,
                   use_llm: bool = False, openai_client=None) -> str:
    """
    If use_llm=True and openai_client is provided, calls GPT.
    Otherwise returns a high-quality templated reply with empathy & context.
    """
    # Tone control based on sentiment
    opening = {
        "Positive": "Thank you for reaching out and for the clear details.",
        "Neutral": "Thank you for contacting our support team.",
        "Negative": "I’m sorry for the trouble you’re facing, and I appreciate your patience."
    }[sentiment]

    if not use_llm or openai_client is None:
        # Template with category-aware content
        category_note = {
            "Billing Issue": "I understand you’re experiencing a billing/pricing issue. I’ll help clarify charges and ensure your account reflects the correct amount.",
            "Account Issue": "It looks like there’s an account access issue. I’ll guide you through restoring access securely.",
            "Technical/Integration": "I see you have a technical integration question. I’ll share the required steps and references.",
            "General Query": "I’ll address your query with the necessary details below."
        }[category]

        product_line = f"\n• Referenced product: **{product_hint}**" if product_hint else ""

        next_steps = {
            "Billing Issue": [
                "Share the last 4 digits of the invoice ID (no sensitive info).",
                "Confirm the billing period and the expected amount.",
                "We’ll review and correct any discrepancy."
            ],
            "Account Issue": [
                "Use the ‘Forgot Password’ link and follow the instructions.",
                "If still blocked, reply with the email/username used for the account (no passwords).",
                "We’ll verify and restore access."
            ],
            "Technical/Integration": [
                "Confirm the API endpoint and the HTTP response you’re seeing.",
                "Share a minimal log snippet (without secrets).",
                "We’ll provide a working example or patch quickly."
            ],
            "General Query": [
                "I’ve summarized key details below.",
                "If I missed anything, reply with specifics and I’ll refine the answer.",
                "Happy to help further."
            ],
        }[category]

        urgent_line = "We’re prioritizing this and will respond first." if priority == "Urgent" else "This will be handled promptly."

        bullet_next = "\n".join([f"  - {s}" for s in next_steps])

        return (
f"""Subject: Re: {subject}

Hi,

{opening} {category_note}{product_line}

**What I’ll do next**
- {urgent_line}
- Create a ticket and track progress until resolution.

**Action items (to accelerate resolution)**
{bullet_next}

If there’s anything else we should consider (browser, environment, steps tried), please reply and I’ll tailor the fix.

Best regards,
Support Team
"""
        )

    # LLM path (optional)
    prompt = f"""
You are a professional, friendly support agent. Write a concise, empathetic reply.
Subject: {subject}
Category: {category}
Sentiment: {sentiment}
Priority: {priority}
Body:
{body}

If product is mentioned, reference it. Keep it polite and action-oriented.
"""
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": prompt}],
        temperature=0.4,
        max_tokens=260,
    )
    return completion.choices[0].message.content.strip()
