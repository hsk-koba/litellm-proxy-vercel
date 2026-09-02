import os
from flask import Flask, request, jsonify, Response
import json
import time
from litellm import Router

app = Flask(__name__)

# --- 1. Router のモデルリスト定義 ---
model_list = [
    # Opus級: NVIDIA NIM Kimi K3
    {
        "model_name": "opus",
        "litellm_params": {
            "model": "nvidia_nim/moonshotai/kimi-k3",
            "api_base": "https://integrate.api.nvidia.com/v1",
            "api_key": os.environ.get("NVIDIA_API_KEY"),
            "reasoning_effort": "high",
        },
        "model_info": {
            "allowed_fails": 1,
            "cooldown_time": 60
        }
    },
    # Sonnet級: NVIDIA NIM Nemotron Ultra
    {
        "model_name": "sonnet",
        "litellm_params": {
            "model": "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b",
            "api_base": "https://integrate.api.nvidia.com/v1",
            "api_key": os.environ.get("NVIDIA_API_KEY"),
        },
        "model_info": {
            "allowed_fails": 1,
            "cooldown_time": 60
        }
    },
    # Sonnet障害時の中間レーン
    {
        "model_name": "nemotron-super",
        "litellm_params": {
            "model": "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
            "api_base": "https://integrate.api.nvidia.com/v1",
            "api_key": os.environ.get("NVIDIA_API_KEY"),
        },
        "model_info": {
            "allowed_fails": 1,
            "cooldown_time": 60
        }
    },
    # Haiku級
    {
        "model_name": "haiku",
        "litellm_params": {
            "model": "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
            "api_base": "https://integrate.api.nvidia.com/v1",
            "api_key": os.environ.get("NVIDIA_API_KEY"),
        },
        "model_info": {
            "allowed_fails": 1,
            "cooldown_time": 60
        }
    }
]

# --- 2. フォールバック設定 ---
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
    """Claude 形式の複雑な content (list/dict) を OpenAI 互換の文字列に変換"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
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

    # バックエンドモデルの呼び出し
    response = router.completion(
        model=target_model,
        messages=formatted_messages,
        stream=stream,
        drop_params=True
    )

    # -----------------------------------------------------------
    # 1. ストリーミング応答 (Claude Code SSE 形式へ変換)
    # -----------------------------------------------------------
    if stream:
        def generate():
            # 必須: message_start イベント
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
            
            # 必須: content_block_start イベント
            block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            }
            yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"

            # チャンクごとのテキスト出力
            for chunk in response:
                try:
                    delta_text = chunk.choices[0].delta.content or ""
                except Exception:
                    delta_text = ""

                if delta_text:
                    delta_event = {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": delta_text}
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

            # 必須: content_block_stop & message_stop イベント
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            
            msg_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 100}
            }
            yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    # -----------------------------------------------------------
    # 2. 一括レスポンス応答 (Anthropic API 形式へ変換)
    # -----------------------------------------------------------
    try:
        content_text = response.choices[0].message.content or ""
    except Exception:
        content_text = ""

    anthropic_response = {
        "id": f"msg_{int(time.time())}",
        "type": "message",
        "role": "assistant",
        "model": requested_model or "claude-3-5-sonnet-20241022",
        "content": [
            {
                "type": "text",
                "text": content_text
            }
        ],
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
# WSGI エントリポイント
if __name__ == "__main__":
    app.run(port=8000, debug=True)