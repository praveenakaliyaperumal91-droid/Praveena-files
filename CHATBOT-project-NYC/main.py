import os
import time
from flask import Request, jsonify, make_response
import functions_framework
from markdown import markdown as md_to_html
from selectolax.parser import HTMLParser
from systemprompt import systemprompt
from vertexai import init, rag
from anthropic import AnthropicVertex
from google.cloud import logging as cloud_logging  # ✅ REQUIRED

# ---------------- Config ----------------
PROJECT_ID = os.getenv("GCP_PROJECT", "christinevalmy")
REGION = os.environ.get("FUNCTION_REGION", "us-east5")
MODEL = os.environ.get("MODEL", "claude-opus-4-1@20250805")
CORPUS_RESOURCE = os.environ.get(
    "RAG_CORPUS",
    "projects/christinevalmy/locations/us-central1/ragCorpora/1152921504606846976"
)

# ---------------- Init ----------------
init(project=PROJECT_ID, location=REGION)
anthropic_client = AnthropicVertex(region=REGION, project_id=PROJECT_ID)

# Cloud Logging
logging_client = cloud_logging.Client()
logger = logging_client.logger("cv-claude-opus")

# ---------------- Helpers ----------------
def generate_prompt(user_input: str) -> str:
    return systemprompt + "\nUser: " + user_input

def retrieve_from_rag(query_text: str):
    return rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=CORPUS_RESOURCE)],
        text=query_text,
        rag_retrieval_config=rag.RagRetrievalConfig(top_k=5)
    )

def md_to_plaintext(md: str) -> str:
    html = md_to_html(md)
    tree = HTMLParser(html)
    if tree.body:
        txt = tree.body.text(separator="\n")
    else:
        txt = tree.root.text(separator="\n")
    return txt.replace("\\u2019", "'").replace("\\u2014", "—").strip()

# ---------------- HTTP Entry ----------------
@functions_framework.http
def app(request: Request):
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get("user_id", "unknown")
        thread_id = data.get("thread_id", "unknown")

        # 1) Cap history (reduce memory/token use)
        history = (data.get("history") or [])[-50:]

        # Accept aliases for the query
        user_query = (
            data.get("query")
            or data.get("message")
            or data.get("text")
            or ""
        ).strip()

        logger.log_struct({
            "event": "user_message",
            "user_id": user_id,
            "thread_id": thread_id,
            "message": user_query,
            "role": "user"
        }, severity="INFO")

        # 2) RAG retrieval (cap to top-3 short snippets)
        try:
            ctx = retrieve_from_rag(user_query)
            snippets = []
            matches = getattr(ctx, "matches", []) or []
            for m in matches[:3]:
                t = getattr(m, "content", None) or getattr(m, "text", None) or ""
                if isinstance(t, str) and t.strip():
                    snippets.append(t.strip()[:1200])
            context_str = "\n\n---\n".join(snippets)
        except Exception as e:
            logger.log_struct({"event": "rag_error", "detail": str(e)}, severity="WARNING")
            context_str = ""

        # 3) Build prompt
        base_prompt = generate_prompt(
            f"Retrieved context:\n{context_str}\n\nUser question: {user_query}"
        )

        # 4) Messages (expecting Anthropic-style content arrays in history)
        messages = []
        for m in history:
            role = (m.get("role") or "").strip().lower()
            content = m.get("content")
            if role in {"user", "assistant"} and isinstance(content, list) and content:
                messages.append({"role": role, "content": content})

        # Append current user prompt as proper content array
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": base_prompt}]
        })

        # 5) Model call
        start = time.time()
        resp = anthropic_client.messages.create(
            model=MODEL,
            messages=messages,
            max_tokens=1000,
        )
        latency = round(time.time() - start, 2)

        # 6) Post-process
        answer_md = "\n\n".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "text", None)
        )
        answer_text = md_to_plaintext(answer_md)

        # Log assistant reply
        logger.log_struct({
            "event": "assistant_reply",
            "user_id": user_id,
            "thread_id": thread_id,
            "message": answer_text,
            "role": "assistant",
            "model": MODEL,
            "latency_sec": latency
        }, severity="INFO")

        return make_response(jsonify(
            response=answer_text,
            model=MODEL,
            latency_sec=latency,
            rag_corpus=CORPUS_RESOURCE
        ), 200)

    except Exception as e:
        logger.log_struct({
            "event": "error",
            "error_type": type(e).__name__,
            "detail": str(e)
        }, severity="ERROR")
        return make_response(jsonify(error=type(e).__name__, detail=str(e)), 500)
