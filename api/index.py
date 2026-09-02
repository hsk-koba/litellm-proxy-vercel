import os
from flask import Flask, request, jsonify, Response
import json:wq
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
    
    # x-api-key (Anthropic形式) または Authorization (OpenAI形式) を両方チェック
    auth_header = request.headers.get("x-api-key") or request.headers.get("Authorization", "")
    auth_header = auth_header.replace("Bearer ", "").strip()
    return auth_header == master_key

# --- エンドポイント設定 ---

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "running", "message": "LiteLLM Router for Claude Code"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

# 1. Claude Code が呼び出す Anthropic Messages API (/v1/messages)
@app.route("/v1/messages", methods=["POST"])
def claude_messages():
    if not check_auth():
        return jsonify({"error": {"type": "authentication_error", "message": "Invalid API Key"}}), 401

    data = request.get_json() or {}
    
    # Claude Code 側からのモデル名要求（例: claude-3-5-sonnet...）をプロキシ内のエイリアスに変換
    requested_model = data.get("model", "").lower()
    if "opus" in requested_model:
        target_model = "opus"
    elif "haiku" in requested_model:
        target_model = "haiku"
    else:
        target_model = "sonnet"  # デフォルトは sonnet にルーティング

    messages = data.get("messages", [])
    system_prompt = data.get("system")
    stream = data.get("stream", False)

    # system プロンプトが存在する場合は messages の先頭に追加
    formatted_messages = []
    if system_prompt:
        if isinstance(system_prompt, list):
            system_content = "\n".join([s.get("text", "") for s in system_prompt if isinstance(s, dict)])
        else:
            system_content = str(system_prompt)
        formatted_messages.append({"role": "system", "content": system_content})

    formatted_messages.extend(messages)

    try:
        response = router.completion(
            model=target_model,
            messages=formatted_messages,
            stream=stream,
            drop_params=True
        )

        # ストリーミング (SSE) 応答
        if stream:
            def generate():
                for chunk in response:
                    chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                    yield f"data: {json.dumps(chunk_dict)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(
                generate(),
                mimetype="text/event-stream",
                headers={"X-Accel-Buffering": "no"}
            )

        # 通常一括応答
        res_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        return jsonify(res_dict), 200

    except Exception as e:
        return jsonify({"error": {"type": "api_error", "message": str(e)}}), 500


# 2. OpenAI 互換クライアント用 (/v1/chat/completions)
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    model = data.get("model", "opus")
    messages = data.get("messages", [])
    stream = data.get("stream", False)

    try:
        response = router.completion(
            model=model,
            messages=messages,
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

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WSGI エントリポイント
if __name__ == "__main__":
    app.run(port=8000, debug=True)