"""Flask server helpers for OpenHospital doctor agents."""

from __future__ import annotations

import asyncio
import html
import inspect
import re
from pathlib import Path
from typing import Any, Callable, Dict

from flask import Flask, Response, jsonify, request

RunHandler = Callable[..., Any]


def create_agent_server(
    *,
    test_handler: RunHandler,
    title: str = "OpenHospital Agent",
) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> Any:
        return _readme_response(title=title)

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.post("/test")
    def test() -> Any:
        return _run_endpoint(test_handler)

    return app


def serve_agent(
    *,
    test_handler: RunHandler,
    host: str = "0.0.0.0",
    port: int = 7860,
    title: str = "OpenHospital Agent",
) -> None:
    app = create_agent_server(
        test_handler=test_handler,
        title=title,
    )
    app.run(host=host, port=int(port))


def _run_endpoint(handler: RunHandler) -> Any:
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"success": False, "error": "Request body must be a JSON object."}), 400

    service_token = _request_service_token(payload)
    if not service_token:
        if not _local_test_allowed():
            return jsonify({
                "success": False,
                "error": "contestServiceToken is required for remote /test calls.",
            }), 401
        payload = dict(payload)
        payload["local_test"] = True
    else:
        payload = dict(payload)
        payload["service_token"] = service_token

    service_base_url = _request_service_base_url(payload)
    if service_base_url:
        payload["service_base_url"] = service_base_url
    team_id = _request_team_id(payload)
    if team_id:
        payload["team_id"] = team_id

    try:
        result = _call_handler(handler, payload=payload)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500
    response = {"success": True, "result": result}
    if isinstance(result, dict) and "final_result" in result:
        response["final_result"] = result["final_result"]
    return jsonify(response)


def _call_handler(handler: RunHandler, *, payload: Dict[str, Any]) -> Any:
    try:
        value = handler(payload=payload)
    except TypeError:
        value = handler()
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def _readme_response(*, title: str) -> Any:
    readme_path = Path.cwd() / "README.md"
    if not readme_path.is_file():
        return jsonify(
            {
                "name": title,
                "status": "ok",
                "endpoints": {
                    "health": "GET /health",
                    "test": "POST /test",
                },
            }
        )

    readme_content = readme_path.read_text(encoding="utf-8")
    readme_content = _remove_markdown_front_matter(readme_content)
    html_content = _markdown_to_html(readme_content)

    return Response(
        f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.8.1/github-markdown.min.css">
    <title>{html.escape(title)}</title>
    <style>
        .markdown-body {{
            box-sizing: border-box;
            min-width: 200px;
            max-width: 920px;
            margin: 0 auto;
            padding: 40px;
        }}
        @media (max-width: 767px) {{
            .markdown-body {{
                padding: 24px;
            }}
        }}
    </style>
</head>
<body>
    <article class="markdown-body">
        {html_content}
    </article>
</body>
</html>
""",
        mimetype="text/html",
    )


def _remove_markdown_front_matter(text: str) -> str:
    return re.sub(r"\A---\s*\n.*?\n---\s*\n?", "", text, flags=re.DOTALL)


def _markdown_to_html(text: str) -> str:
    try:
        import markdown2
    except ImportError:
        return f"<pre>{html.escape(text)}</pre>"
    return markdown2.markdown(text, extras=["fenced-code-blocks", "tables"])


def _request_service_token(payload: Dict[str, Any]) -> str:
    value = request.headers.get("X-Hospital-Service-Token")
    if value is None:
        value = payload.get("service_token") or payload.get("contestServiceToken")
    text = str(value or "").strip()
    if text.lower().startswith("bearer "):
        return text.split(None, 1)[1].strip()
    return text


def _request_service_base_url(payload: Dict[str, Any]) -> str:
    value = (
        payload.get("service_base_url")
        or payload.get("contestServiceBaseUrl")
        or request.headers.get("X-Hospital-Service-Base-Url")
    )
    return str(value or "").strip()


def _request_team_id(payload: Dict[str, Any]) -> str:
    value = (
        payload.get("team_id")
        or payload.get("account")
        or request.headers.get("X-Hospital-Team-Id")
    )
    return str(value or "").strip()


def _local_test_allowed() -> bool:
    forwarded_for = str(request.headers.get("X-Forwarded-For") or "").strip()
    if forwarded_for:
        return False
    remote_addr = str(request.remote_addr or "").strip()
    return remote_addr in {"127.0.0.1", "::1", "localhost"}
