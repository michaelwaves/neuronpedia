# Modal Deployment Guide for Neuronpedia Inference Server

This guide explains how to deploy your Neuronpedia inference server on Modal, a serverless platform that handles GPU provisioning, scaling, and deployment automatically.

## Why Modal?

**Benefits:**
- **No Docker/Docker Compose needed**: Modal handles containerization automatically
- **Automatic GPU provisioning**: Specify GPU type, Modal provides it
- **Serverless scaling**: Scales to zero when not in use (no charges), scales up automatically
- **Built-in HuggingFace cache**: Persistent volumes prevent re-downloading models
- **Simple deployment**: One command to deploy, one URL to access

**Trade-offs:**
- Runs on Modal's infrastructure (not your local machine)
- Cold starts when scaling from zero (can be optimized with keep-warm settings)
- Requires Modal account (free tier available)

## Setup

### 1. Install Modal

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal setup
```

This opens a browser and logs you into your Modal workspace.

### 3. Create Modal Secrets (Optional but Recommended)

The inference server uses a single secret named `neuronpedia-inference` with the following environment variables:

```bash
modal secret create neuronpedia-inference \
    HF_TOKEN=your_hf_token_here \
    SECRET=your_api_secret_here \
    SENTRY_DSN=your_sentry_dsn_here
```

**Secret variables explained:**

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `HF_TOKEN` | Optional | HuggingFace token for downloading gated models (e.g., Llama, Gemma). Get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `SECRET` | Optional | API authentication key. If set, clients must include `X-SECRET-KEY: your_secret` header in requests |
| `SENTRY_DSN` | Optional | Sentry error tracking DSN for monitoring. Get from your Sentry project settings |

**Minimal setup (no secrets):**
```bash
# Skip this step if you're only using public models like GPT-2
# The server will work without secrets, just without auth or gated model access
```

**Only HuggingFace token:**
```bash
modal secret create neuronpedia-inference HF_TOKEN=hf_your_token_here
```

**With authentication:**
```bash
modal secret create neuronpedia-inference \
    HF_TOKEN=hf_your_token_here \
    SECRET=my-secure-api-key-123
```

## Deployment Options

### Option 1: Deploy as Web Endpoint (Recommended)

This deploys your FastAPI server as a persistent web endpoint.

```bash
cd /home/michaelwaves/repos/alignarena/backend/neuronpedia
modal deploy apps/inference/modal_server.py
```

**What happens:**
1. Modal builds a Docker image with CUDA, Python, and all dependencies
2. Creates a persistent deployment with a public HTTPS URL
3. Server auto-scales based on traffic (scales to zero when idle)

**Access the server:**
```bash
# Modal prints the URL after deployment, something like:
# https://your-workspace--neuronpedia-inference-serve.modal.run

curl https://your-workspace--neuronpedia-inference-serve.modal.run/health
```

### Option 2: Run Ephemerally (For Testing)

This runs the server temporarily for development/testing:

```bash
modal run apps/inference/modal_server.py
```

The server shuts down when you stop the script (Ctrl+C).

### Option 3: Function-Based Inference (For Batch Jobs)

Use the `run_inference_with_config` function for batch processing:

```python
import modal

app = modal.App.lookup("neuronpedia-inference")
run_inference = modal.Function.lookup("neuronpedia-inference", "run_inference_with_config")

# Run inference job
result = run_inference.remote(
    model_id="gpt2-small",
    sae_sets=["res-jb"],
    max_loaded_saes=200,
)
print(result)
```

## Configuration

### GPU Selection

Edit `modal_server.py` to change GPU type:

```python
@app.function(
    gpu="H100",  # Options: "T4", "L4", "A10", "A100", "A100-40GB", "A100-80GB", "L40S", "H100", "H200", "B200"
    ...
)
```

**Available GPU types:**
- `T4` - Budget option (~$0.60/hr)
- `L4` - Good price/performance (~$0.80/hr)
- `A10` - 24GB VRAM (~$1.10/hr)
- `A100` / `A100-40GB` - High performance, 40GB VRAM (~$3.00/hr)
- `A100-80GB` - High performance, 80GB VRAM (~$4.00/hr)
- `L40S` - 48GB VRAM (~$2.00/hr)
- `H100` - Highest performance, 80GB VRAM (~$4.50/hr)
- `H200` - Latest generation (~$6.00/hr)
- `B200` - Cutting edge (~$8.00/hr)

See [Modal pricing](https://modal.com/pricing) for exact costs.

### Model Configuration

**Option 1: Environment Variables (in Modal secrets)**

```bash
modal secret create neuronpedia-inference \
    MODEL_ID=gemma-2-2b \
    SAE_SETS='["gemmascope-res-16k"]' \
    MODEL_DTYPE=bfloat16 \
    SAE_DTYPE=bfloat16
```

**Option 2: Edit modal_server.py defaults**

```python
os.environ.setdefault("MODEL_ID", "gemma-2-2b")
os.environ.setdefault("SAE_SETS", '["gemmascope-res-16k"]')
os.environ.setdefault("MODEL_DTYPE", "bfloat16")
```

**Option 3: Pass at runtime (function interface)**

```python
run_inference.remote(
    model_id="meta-llama/Llama-3.1-8B",
    custom_hf_model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    sae_sets=["llamascope-r1-res-32k"],
    model_dtype="bfloat16",
)
```

### Persistent Model Cache

Modal automatically mounts a persistent volume at `/cache` for HuggingFace models:

```python
hf_cache_volume = modal.Volume.from_name("hf-cache-neuronpedia", create_if_missing=True)

@app.function(
    volumes={HF_CACHE_PATH: hf_cache_volume},
    ...
)
```

**First run**: Downloads models (~500MB-5GB)
**Subsequent runs**: Uses cached models (instant startup)

## Comparing Docker vs Modal

### Docker Setup (Current)

```bash
# Build
make inference-localhost-build-gpu USE_LOCAL_HF_CACHE=1

# Run
make inference-localhost-dev-gpu \
    MODEL_SOURCESET=gpt2-small.res-jb \
    USE_LOCAL_HF_CACHE=1
```

**Requires:**
- Docker installed locally
- NVIDIA drivers + CUDA toolkit
- GPU on your machine
- Manual port mapping, volume mounts, env vars

### Modal Setup (Proposed)

```bash
# Deploy
modal deploy apps/inference/modal_server.py
```

**Requires:**
- Modal CLI (`pip install modal`)
- Modal account (free tier available)

**Modal handles:**
- Container building (automatic)
- GPU provisioning (any type you specify)
- HTTPS endpoints (automatic)
- Persistent volumes (automatic)
- Scaling (automatic)

## Architecture Comparison

### Docker Architecture
```
YOU
 ↓
Docker Compose (orchestration)
 ↓
Dockerfile (image build)
 ↓
Container (with GPU access via nvidia-container-toolkit)
 ↓
FastAPI server on localhost:5002
```

### Modal Architecture
```
YOU
 ↓
Modal CLI (deployment)
 ↓
Modal Image builder (automatic containerization)
 ↓
Modal Function (serverless container with GPU)
 ↓
FastAPI ASGI app on https://your-workspace.modal.run
```

## Advanced Features

### Keep Containers Warm

Prevent cold starts by keeping containers alive:

```python
@app.function(
    keep_warm=1,  # Keep 1 container always running
    ...
)
```

### Custom Domains

Deploy on your own domain:

```python
@modal.web_endpoint(custom_domain="inference.neuronpedia.org")
```

### Monitoring

View logs and metrics in Modal dashboard:
```bash
modal logs neuronpedia-inference
```

Or visit: https://modal.com/apps

### Multiple Model Configurations

Deploy multiple endpoints with different models:

```python
@app.function(gpu="A10")
@modal.asgi_app(label="gpt2")
def serve_gpt2():
    os.environ["MODEL_ID"] = "gpt2-small"
    from neuronpedia_inference.server import app
    return app

@app.function(gpu="H100")
@modal.asgi_app(label="llama")
def serve_llama():
    os.environ["MODEL_ID"] = "meta-llama/Llama-3.1-8B"
    from neuronpedia_inference.server import app
    return app
```

## Cost Estimation

Modal charges only for actual compute time:

- **T4**: ~$0.60/hour (16GB VRAM) - Budget option
- **L4**: ~$0.80/hour (24GB VRAM) - Good price/performance
- **A10**: ~$1.10/hour (24GB VRAM) - Default in template
- **A100-40GB**: ~$3.00/hour (40GB VRAM) - High performance
- **A100-80GB**: ~$4.00/hour (80GB VRAM) - Large models
- **H100**: ~$4.50/hour (80GB VRAM) - Highest performance
- **H200**: ~$6.00/hour - Latest generation
- **B200**: ~$8.00/hour - Cutting edge

**With autoscaling:**
- Idle (scaled to zero): $0/hour
- Under load: Only charged for actual compute time

See [Modal pricing](https://modal.com/pricing) for exact current rates.

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure the inference code is in the same directory:

```python
sys.path.insert(0, str(Path(__file__).parent))
```

### GPU Not Available

Check GPU is specified correctly:
```python
@app.function(gpu="A10")  # Valid options: T4, L4, A10, A100, H100, etc.
```

Note: "A10G" is not valid - use "A10" instead!

### HuggingFace Auth Errors

If you see "401 Unauthorized" when downloading gated models:

```bash
# Check if secrets exist
modal secret list

# Create or update the secret with your HF token
modal secret create neuronpedia-inference HF_TOKEN=hf_your_token_here

# Or update existing secret
modal secret update neuronpedia-inference HF_TOKEN=hf_your_new_token
```

Get your token from: https://huggingface.co/settings/tokens

### Authentication Errors (X-SECRET-KEY)

If you set the `SECRET` environment variable, all API requests must include the header:

```bash
curl -H "X-SECRET-KEY: your_secret_here" https://your-url.modal.run/health
```

To disable authentication, remove `SECRET` from your Modal secret:
```bash
modal secret create neuronpedia-inference HF_TOKEN=hf_your_token  # no SECRET
```

### Slow Cold Starts

Use `keep_warm` or pre-download models in image build:

```python
def download_models():
    from transformers import AutoModel
    AutoModel.from_pretrained("gpt2-small")

inference_image = (
    modal.Image.from_registry(...)
    .run_function(
        download_models,
        volumes={HF_CACHE_PATH: hf_cache_volume},
    )
)
```

## Next Steps

1. **Deploy**: `modal deploy apps/inference/modal_server.py`
2. **Test**: `curl https://your-url.modal.run/health`
3. **Monitor**: Visit https://modal.com/apps
4. **Optimize**: Add `keep_warm`, tune GPU type, cache models in image

## Resources

- [Modal Docs](https://modal.com/docs)
- [Modal GPU Guide](https://modal.com/docs/guide/gpu)
- [Modal CUDA Guide](https://modal.com/docs/guide/cuda)
- [Modal Pricing](https://modal.com/pricing)
