import os
from flask import Flask, request, jsonify
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

# ルーター初期化
router = Router(
    model_list=model_list,
    fallbacks=fallbacks,
    num_retries=0,
    timeout=120
)

# --- 認証ミドルウェア関数 ---
def check_auth():
    master_key = os.environ.get("LITELLM_MASTER_KEY")
    if not master_key:
        return True  # マスターキー未設定の場合は通過させる（テスト用）

    auth_header = request.headers.get("Authorization", "")
    expected_header = f"Bearer {master_key}"
    
    if auth_header != expected_header:
        return False
    return True


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    model = data.get("model", "opus")
    messages = data.get("messages", [])
    stream = data.get("stream", False)

    if not messages:
        return jsonify({"error": "messages is required"}), 400

    try:
        response = router.completion(
            model=model,
            messages=messages,
            stream=stream,
            drop_params=True
        )

        # --------------------------------------------------
        # 1. ストリーミング処理 (stream=True)
        # --------------------------------------------------
        if stream:
            def generate():
                for chunk in response:
                    # LiteLLM の Chunk オブジェクトを dict 化
                    if hasattr(chunk, "model_dump"):
                        chunk_dict = chunk.model_dump()
                    else:
                        chunk_dict = dict(chunk)
                    
                    # SSE 形式 (data: {JSON}\n\n) で出力
                    yield f"data: {json.dumps(chunk_dict)}\n\n"
                
                # OpenAI 互換の終了シグナル
                yield "data: [DONE]\n\n"

            return Response(
                generate(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # バッファリングを無効化
                }
            )

        # --------------------------------------------------
        # 2. 一括レスポンス処理 (stream=False)
        # --------------------------------------------------
        if hasattr(response, "model_dump"):
            res_dict = response.model_dump()
        else:
            res_dict = dict(response)

        return jsonify(res_dict), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WSGI エントリポイント
if __name__ == "__main__":
    app.run(port=8000, debug=True)