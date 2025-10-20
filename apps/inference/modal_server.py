"""
Modal deployment for Neuronpedia Inference Server.

Example usage:
    modal deploy apps/inference/modal_server.py

    # Or run ephemerally:
    modal run apps/inference/modal_server.py
"""

import modal

CUDA_VERSION = "12.8.1"
FLAVOR = "devel"
OPERATING_SYS = "ubuntu24.04"
TAG = f"{CUDA_VERSION}-{FLAVOR}-{OPERATING_SYS}"

HF_CACHE_PATH = "/cache"

inference_image = (
    modal.Image.from_registry(f"nvidia/cuda:{TAG}", add_python="3.10")
    .entrypoint([])
    .apt_install("git", "gcc", "g++", "make")
    .pip_install(
        "python-dotenv>=1.0.1",
        "numpy>=1.24",
        "psutil>=5.9.8",
        "torch>=1.10",
        "transformers>=4.38.1",
        "einops>=0.7.0",
        "huggingface-hub>=0.34.0",
        "pandas>=2.2.2",
        "sae-lens>=6.12.1",
        "fastapi>=0.115.6",
        "uvicorn>=0.34.0",
        "sentry-sdk[fastapi]>=2.20.0",
        "nnsight>=0.4.3",
        "hf-transfer>=0.1.9",
    )
    .run_commands(
        "pip install git+https://github.com/hijohnnylin/TransformerLens.git@temp_branch_version"
    )
    .add_local_dir(
        local_path="../../packages/python/neuronpedia-inference-client",
        remote_path="/root/neuronpedia-inference-client",
        copy=True,
    )
    .run_commands(
        "pip install -e /root/neuronpedia-inference-client"
    )
    .env({
        "HF_HUB_CACHE": HF_CACHE_PATH,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .add_local_dir(
        local_path="neuronpedia_inference",
        remote_path="/root/neuronpedia_inference",
    )
)

app = modal.App("neuronpedia-inference", image=inference_image)

hf_cache_volume = modal.Volume.from_name("hf-cache-neuronpedia", create_if_missing=True)

@app.function(
    gpu="A10",
    volumes={HF_CACHE_PATH: hf_cache_volume},
    secrets=[modal.Secret.from_name("neuronpedia-inference")],
    scaledown_window=300,
    timeout=3600,
)
@modal.web_server(5002, startup_timeout=300)
def serve():
    """
    Serve the FastAPI inference server via uvicorn.

    Required secrets in 'neuronpedia-inference':
    - HF_TOKEN: HuggingFace token for downloading gated models (optional)
    - SECRET: API secret key for X-SECRET-KEY auth header (optional)
    - SENTRY_DSN: Sentry error tracking DSN (optional)
    """
    import os
    import subprocess

    os.environ["MODEL_ID"] = os.getenv("MODEL_ID", "gpt2-small")
    os.environ["SAE_SETS"] = os.getenv("SAE_SETS", '["res-jb"]')
    os.environ["MODEL_DTYPE"] = os.getenv("MODEL_DTYPE", "float32")
    os.environ["SAE_DTYPE"] = os.getenv("SAE_DTYPE", "float32")
    os.environ["TOKEN_LIMIT"] = os.getenv("TOKEN_LIMIT", "200")
    os.environ["MAX_LOADED_SAES"] = os.getenv("MAX_LOADED_SAES", "500")
    os.environ["INCLUDE_SAE"] = os.getenv("INCLUDE_SAE", "[]")
    os.environ["EXCLUDE_SAE"] = os.getenv("EXCLUDE_SAE", "[]")
    os.environ["MODEL_FROM_PRETRAINED_KWARGS"] = os.getenv("MODEL_FROM_PRETRAINED_KWARGS", "{}")

    subprocess.Popen([
        "uvicorn",
        "neuronpedia_inference.server:app",
        "--host", "0.0.0.0",
        "--port", "5002",
    ])


@app.function(
    gpu="A10",
    volumes={HF_CACHE_PATH: hf_cache_volume},
    secrets=[modal.Secret.from_name("neuronpedia-inference")],
    scaledown_window=300,
    timeout=3600,
)
def run_inference_with_config(
    model_id: str = "gpt2-small",
    sae_sets: list[str] | None = None,
    model_dtype: str = "float32",
    sae_dtype: str = "float32",
    token_limit: int = 200,
    max_loaded_saes: int = 500,
    override_model_id: str | None = None,
    custom_hf_model_id: str | None = None,
    include_sae: list[str] | None = None,
    exclude_sae: list[str] | None = None,
):
    """
    Alternative function-based interface for running inference jobs.
    This is useful for batch processing or custom workflows.
    """
    import json
    import os
    import sys
    from pathlib import Path

    inference_dir = Path(__file__).parent
    sys.path.insert(0, str(inference_dir))

    sae_sets = sae_sets or ["res-jb"]
    include_sae = include_sae or []
    exclude_sae = exclude_sae or []

    os.environ["MODEL_ID"] = model_id
    os.environ["SAE_SETS"] = json.dumps(sae_sets)
    os.environ["MODEL_DTYPE"] = model_dtype
    os.environ["SAE_DTYPE"] = sae_dtype
    os.environ["TOKEN_LIMIT"] = str(token_limit)
    os.environ["MAX_LOADED_SAES"] = str(max_loaded_saes)
    os.environ["INCLUDE_SAE"] = json.dumps(include_sae)
    os.environ["EXCLUDE_SAE"] = json.dumps(exclude_sae)
    os.environ["MODEL_FROM_PRETRAINED_KWARGS"] = "{}"

    if override_model_id:
        os.environ["OVERRIDE_MODEL_ID"] = override_model_id
    if custom_hf_model_id:
        os.environ["CUSTOM_HF_MODEL_ID"] = custom_hf_model_id

    return {"status": "Model initialized", "model_id": model_id}


@app.local_entrypoint()
def main(
    model_id: str = "gpt2-small",
    sae_sets: str = "res-jb",
):
    """
    Local entrypoint for testing Modal deployment.

    Usage:
        modal run apps/inference/modal_server.py --model-id gpt2-small --sae-sets res-jb
    """
    sae_sets_list = [s.strip() for s in sae_sets.split(",")]
    result = run_inference_with_config.remote(
        model_id=model_id,
        sae_sets=sae_sets_list,
    )
    print(f"Inference server initialized: {result}")
