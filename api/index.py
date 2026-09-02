import os
import json
import time
from flask import Flask, request, jsonify, Response
from litellm import Router

app = Flask(__name__)

# --- 1. Router のモデルリスト定義 ---
model_list = [
    {
        "model_name": "opus",
        "litellm_params": {
            "model": "nvidia_nim/moonshotai/kimi-k3",
            "api_base": "https://integrate.api.nvidia.com/v1",
            "api_key": os.environ.get("NVIDIA_API_KEY"),
            "extra_body": {"chat_template_kwargs": {"thinking": False}}
        },
        "model_info": {"allowed_fails": 1, "cooldown_time": 60}
    },
    {
        "model_name": "sonnet",
        "litellm_params": {
            "model": "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b",
            "api_base": "https://integrate.api.nvidia.com/v1",
            "api_key": os.environ.get("NVIDIA_API_KEY"),
            "extra_body": {"chat_template_kwargs": {"thinking": False}}
        },
        "model_info": {"allowed_fails": 1, "cooldown_time": 60}
    },
    {
        "model_name": "nemotron-super",
        "litellm_params": {
            "model": "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
            "api_base": "https://integrate.api.nvidia.com/v1",
            "api_key": os.environ.get("NVIDIA_API_KEY"),
            "extra_body": {"chat_template_kwargs": {"thinking": False}}
        },
        "model_info": {"allowed_fails": 1, "cooldown_time": 60}
    },
    {
        "model_name": "haiku",
        "litellm_params": {
            "model": "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
            "api_base": "https://integrate.api.nvidia.com/v1",
            "api_key": os.environ.get("NVIDIA_API_KEY"),
        },
        "model_info": {"allowed_fails": 1, "cooldown_time": 60}
    }
]

fallbacks = [
    {"opus": ["sonnet", "nemotron-super", "haiku"]},
    {"sonnet": ["nemotron-super", "haiku"]},
    {"nemotron-super": ["haiku"]}
]

router = Router(model_list=model_list, fallbacks=fallbacks, num_retries=0, timeout=120)

def check_auth():
    master_key = os.environ.get("LITELLM_MASTER_KEY")
    if not master_key:
        return True
    auth_header = request.headers.get("x-api-key") or request.headers.get("Authorization", "")
    auth_header = auth_header.replace("Bearer ", "").strip()
    return auth_header == master_key

def clean_content(content):
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

def handle_messages_request(data):
    requested_model = data.get("model", "").lower()
    if "opus" in requested_model:
        target_model = "opus"
    elif "haiku" in requested_model:
        target_model = "haiku"
    else:
        target_model = "sonnet"

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

    response = router.completion(
        model=target_model,
        messages=formatted_messages,
        stream=stream,
        drop_params=True
    )

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

            for chunk in response:
                delta_text = ""
                try:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    if hasattr(delta, "content") and delta.content:
                        delta_text = delta.content
                    elif hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        delta_text = delta.reasoning_content
                    elif isinstance(delta, dict):
                        delta_text = delta.get("content") or delta.get("reasoning_content") or ""
                except Exception:
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
        content_text = response.choices[0].message.content or ""
    except Exception:
        content_text = ""

    anthropic_response = {
        "id": f"msg_{int(time.time())}",
        "type": "message",
        "role": "assistant",
        "model": requested_model or "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": content_text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": getattr(response, "usage", {}).get("prompt_tokens", 0) if hasattr(response, "usage") else 0,
            "output_tokens": getattr(response, "usage", {}).get("completion_tokens", 0) if hasattr(response, "usage") else 0
        }
    }
    return jsonify(anthropic_response), 200

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
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
    app.run(port=8000, debug=True)