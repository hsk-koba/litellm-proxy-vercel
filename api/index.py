import os
from flask import Flask, request, jsonify, Response
import json
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
    
    # System プロンプトの追加
    if system_prompt:
        formatted_messages.append({
            "role": "system",
            "content": clean_content(system_prompt)
        })

    # 各メッセージのフォーマット整形
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
            for chunk in response:
                chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                yield f"data: {json.dumps(chunk_dict)}\n\n"
            yield "data: [DONE]\n\n"

        return Response(generate(), mimetype="text/event-stream")

    res_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    return jsonify(res_dict), 200


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