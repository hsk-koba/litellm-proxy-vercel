import os
import json
import time
from typing import Any

import requests
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

# LiteLLM Proxy はコンテナ内で別プロセスとして起動する。外部には Flask
# (Anthropic Messages 互換 API) だけを公開し、モデル選択とフォールバックは
# litellm_config.yml 内の Proxy に一任する。
LITELLM_PROXY_URL = os.environ.get("LITELLM_PROXY_URL", "http://127.0.0.1:4000").rstrip("/")
LITELLM_PROXY_TIMEOUT = int(os.environ.get("LITELLM_PROXY_TIMEOUT", "120"))

def check_auth():
    master_key = os.environ.get("LITELLM_MASTER_KEY")
    if not master_key:
        return True
    auth_header = request.headers.get("x-api-key") or request.headers.get("Authorization", "")
    auth_header = auth_header.replace("Bearer ", "").strip()
    return auth_header == master_key

def clean_content(content: Any) -> str:
    """Tool Call を維持しつつ content を OpenAI/NVIDIA 互換形式にサニタイズ"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                # 通常テキスト
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                # Tool Use (ツール呼び出し)
                elif item.get("type") == "tool_use":
                    text_parts.append(f"[Tool Call: {item.get('name')} {json.dumps(item.get('input', {}))}]")
                # Tool Result (ツール実行結果)
                elif item.get("type") == "tool_result":
                    text_parts.append(f"[Tool Result: {item.get('content')}]")
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)
    return str(content)


def select_model(requested_model: str) -> str:
    """Claude 名を Proxy に定義したモデルエイリアスへ対応付ける。"""
    requested_model = requested_model.lower()
    if "opus" in requested_model:
        return "opus"
    if "haiku" in requested_model:
        return "haiku"
    if "nemotron-super" in requested_model:
        return "nemotron-super"
    return "sonnet"


def proxy_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    master_key = os.environ.get("LITELLM_MASTER_KEY")
    if master_key:
        headers["Authorization"] = f"Bearer {master_key}"
    return headers


def proxy_completion(payload: dict[str, Any], stream: bool):
    """Proxy の OpenAI 互換 chat-completions エンドポイントを呼び出す。"""
    return requests.post(
        f"{LITELLM_PROXY_URL}/v1/chat/completions",
        headers=proxy_headers(),
        json=payload,
        stream=stream,
        timeout=LITELLM_PROXY_TIMEOUT,
    )

def handle_messages_request(data):
    requested_model = data.get("model", "")
    target_model = select_model(requested_model)

    raw_messages = data.get("messages", [])
    system_prompt = data.get("system")
    stream = data.get("stream", False)

    formatted_messages = []
    if system_prompt:
        formatted_messages.append({"role": "system", "content": clean_content(system_prompt)})

    for msg in raw_messages:
        role = msg.get("role", "user")
        content = clean_content(msg.get("content", ""))
        formatted_messages.append({"role": role, "content": content})

    proxy_payload = {
        "model": target_model,
        "messages": formatted_messages,
        "stream": stream,
    }
    # Proxy 側の drop_params=true により、Anthropic 固有の未対応パラメータは
    # 安全に除外される。NVIDIA 側で利用可能な値だけを渡す。
    for key in ("max_tokens", "temperature", "top_p"):
        if key in data:
            proxy_payload[key] = data[key]
    if "stop_sequences" in data:
        proxy_payload["stop"] = data["stop_sequences"]

    response = proxy_completion(proxy_payload, stream=stream)
    if not response.ok:
        try:
            error = response.json()
        except ValueError:
            error = response.text
        return jsonify({"error": error}), response.status_code

    if stream:
        def generate():
            msg_start = {
                "type": "message_start",
                "message": {
                    "id": f"msg_{int(time.time())}",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": requested_model or "claude-3-5-sonnet-20241022",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1}
                }
            }
            yield f"event: message_start\ndata: {json.dumps(msg_start)}\n\n"
            
            block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            }
            yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                delta_text = ""
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    delta_text = delta.get("content") or delta.get("reasoning_content") or ""
                except (IndexError, KeyError, TypeError, ValueError):
                    delta_text = ""

                if delta_text:
                    delta_event = {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": delta_text}
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            
            msg_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 100}
            }
            yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # 一括レスポンス
    try:
        completion = response.json()
        content_text = completion["choices"][0]["message"].get("content") or ""
        usage = completion.get("usage", {})
    except (IndexError, KeyError, TypeError, ValueError):
        content_text = ""
        usage = {}

    anthropic_response = {
        "id": f"msg_{int(time.time())}",
        "type": "message",
        "role": "assistant",
        "model": requested_model or "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": content_text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0)
        }
    }
    return jsonify(anthropic_response), 200

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    try:
        response = requests.get(
            f"{LITELLM_PROXY_URL}/health",
            headers=proxy_headers(),
            timeout=3,
        )
    except requests.RequestException as exc:
        return jsonify({"status": "unhealthy", "proxy": str(exc)}), 503

    if not response.ok:
        return jsonify({"status": "unhealthy", "proxy_status": response.status_code}), 503
    return jsonify({"status": "healthy"}), 200

@app.route("/v1/messages", methods=["POST"])
def claude_messages():
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        return handle_messages_request(request.get_json() or {})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/<path:path>", methods=["GET", "POST"])
def catch_all(path):
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    if request.method == "POST":
        try:
            return handle_messages_request(request.get_json() or {})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"status": "healthy", "path_received": path}), 200

if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", "8000")), debug=False)
