# Using the Modal Inference Server

## Quick Start

### 1. Deploy the Server

```bash
cd /home/michaelwaves/repos/alignarena/backend/neuronpedia/apps/inference
modal deploy modal_server.py
```

**Output:**
```
✓ Created web function serve => https://your-workspace--neuronpedia-inference-serve.modal.run
```

The server is now live! 🎉

### 2. Test the Health Endpoint

```bash
curl https://your-workspace--neuronpedia-inference-serve.modal.run/health
```

**Expected response:**
```json
{"status": "ok", "initialized": true}
```

### 3. View API Documentation

Open in browser:
```
https://your-workspace--neuronpedia-inference-serve.modal.run/docs
```

This shows the FastAPI Swagger UI with all available endpoints.

## Development vs Production

### Development (Auto-reload)

```bash
modal serve modal_server.py
```

- Watches for file changes
- Auto-reloads on save
- Uses `-dev` URL: `https://your-workspace--neuronpedia-inference-serve-dev.modal.run`
- Press Ctrl+C to stop

### Production (Persistent)

```bash
modal deploy modal_server.py
```

- Runs indefinitely
- No auto-reload
- Uses stable URL: `https://your-workspace--neuronpedia-inference-serve.modal.run`
- Manage via Modal dashboard: https://modal.com/apps

## Using the Inference Client

### Install the Client

```bash
cd packages/python/neuronpedia-inference-client
pip install -e .
```

### Example: Get Activations

```python
import neuronpedia_inference_client
from pprint import pprint

# Configure client
configuration = neuronpedia_inference_client.Configuration(
    host="https://your-workspace--neuronpedia-inference-serve.modal.run/v1"
)

# If you set a SECRET, add it here
# configuration.api_key["SimpleSecretAuth"] = "your-secret-key"

with neuronpedia_inference_client.ApiClient(configuration) as api_client:
    api_instance = neuronpedia_inference_client.DefaultApi(api_client)

    # Get all activations for a prompt
    request = neuronpedia_inference_client.ActivationAllPostRequest(
        prompt="Hello world",
        model="gpt2-small",
        source_set="res-jb",
        selected_sources=[],  # Empty = all sources
        sort_by_token_indexes=[],
        ignore_bos=True,
        num_results=10,
    )

    response = api_instance.activation_all_post(request)
    pprint(response)
```

### Example: Steered Completion

```python
# Steer generation with SAE features
steer_request = neuronpedia_inference_client.SteerCompletionRequest(
    prompt="The cat sat on the",
    model="gpt2-small",
    max_new_tokens=20,
    steer_vectors=[
        neuronpedia_inference_client.NpSteerVector(
            features=[
                neuronpedia_inference_client.NpSteerFeature(
                    source="5-res-jb",
                    index=12345,
                    strength=2.0,
                )
            ],
            type="feature",
            method="add",
        )
    ],
)

response = api_instance.steer_completion_post(steer_request)
print(response.completion)
```

### Example: Tokenize Text

```python
tokenize_request = neuronpedia_inference_client.TokenizePostRequest(
    text="Hello, world!",
    model="gpt2-small",
)

response = api_instance.tokenize_post(tokenize_request)
print(f"Tokens: {response.tokens}")
print(f"Token IDs: {response.token_ids}")
```

## Available Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/v1/activation/all` | POST | Get all activations for prompt |
| `/v1/activation/single` | POST | Get activation for single feature |
| `/v1/activation/topk-by-token` | POST | Get top-k features per token |
| `/v1/steer/completion` | POST | Generate text with steering |
| `/v1/steer/completion/chat` | POST | Chat completion with steering |
| `/v1/tokenize` | POST | Tokenize text |
| `/v1/util/sae-vector` | POST | Get SAE feature vector |
| `/v1/util/sae-topk-by-decoder-cossim` | POST | Find similar features by cosine similarity |

See `/docs` for full API specification.

## Using with Authentication

If you set the `SECRET` environment variable in your Modal secret:

```bash
modal secret create neuronpedia-inference SECRET=my-secret-key-123
```

Then all requests must include the header:

```python
configuration = neuronpedia_inference_client.Configuration(
    host="https://your-url.modal.run/v1"
)
configuration.api_key["SimpleSecretAuth"] = "my-secret-key-123"
```

Or with curl:
```bash
curl -H "X-SECRET-KEY: my-secret-key-123" \
     https://your-url.modal.run/health
```

## Testing Your Deployment

Use the provided test script:

```bash
# Update MODAL_URL in test_modal_deployment.py with your URL
python test_modal_deployment.py
```

Or test manually with curl:

```bash
# Health check
curl https://your-url.modal.run/health

# Tokenize
curl -X POST https://your-url.modal.run/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "model": "gpt2-small"}'

# Get activations
curl -X POST https://your-url.modal.run/v1/activation/all \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello world",
    "model": "gpt2-small",
    "source_set": "res-jb",
    "selected_sources": [],
    "sort_by_token_indexes": [],
    "ignore_bos": true,
    "num_results": 10
  }'
```

## Monitoring

### View Logs

```bash
modal logs neuronpedia-inference
```

Or with follow (like `tail -f`):
```bash
modal logs neuronpedia-inference --follow
```

### View Dashboard

Visit https://modal.com/apps and find `neuronpedia-inference`.

You can see:
- Active containers
- Request metrics
- GPU usage
- Costs
- Recent errors

### Check Container Status

```bash
modal container list neuronpedia-inference
```

## Updating the Deployment

### Change Model Configuration

Edit `modal_server.py`:

```python
os.environ.setdefault("MODEL_ID", "gemma-2-2b")
os.environ.setdefault("SAE_SETS", '["gemmascope-res-16k"]')
os.environ.setdefault("MODEL_DTYPE", "bfloat16")
```

Then redeploy:
```bash
modal deploy modal_server.py
```

### Change GPU Type

Edit `modal_server.py`:

```python
@app.function(
    gpu="H100",  # Changed from A10
    ...
)
```

Redeploy:
```bash
modal deploy modal_server.py
```

### Update Dependencies

Edit the `inference_image` definition in `modal_server.py`, then:

```bash
modal deploy modal_server.py
```

Modal will rebuild the image automatically.

## Stopping the Deployment

### Via CLI

```bash
modal app stop neuronpedia-inference
```

### Via Dashboard

1. Go to https://modal.com/apps
2. Find `neuronpedia-inference`
3. Click "Stop"

## Cost Optimization

### Scale to Zero (Default)

The server automatically scales to 0 containers when idle. You're only charged when requests are being processed.

**Current setting:**
```python
scaledown_window=300  # Wait 5 minutes of inactivity before scaling to 0
```

### Keep Warm (Reduce Cold Starts)

If you have consistent traffic and want to avoid cold starts:

```python
@app.function(
    keep_warm=1,  # Always keep 1 container running
    ...
)
```

**Cost impact:** You'll be charged for 1 GPU continuously (~$1.10/hr for A10).

### Use Cheaper GPU for Small Models

For GPT-2 or small models:

```python
@app.function(
    gpu="L4",  # $0.80/hr instead of $1.10/hr
    ...
)
```

## Troubleshooting

### Server Not Responding

Check if it's running:
```bash
modal app list | grep neuronpedia-inference
```

Check logs:
```bash
modal logs neuronpedia-inference
```

### Cold Start Taking Too Long

The first request after idle will be slow (model loading). Options:

1. Use `keep_warm=1` to avoid cold starts
2. Pre-download models in image build (see MODAL_DEPLOYMENT.md)
3. Use smaller model for faster loading

### "Model not initialized" Error

The server starts before models are loaded. Wait ~30-60 seconds after deployment, then check `/health`:

```bash
curl https://your-url.modal.run/health
```

Should return:
```json
{"status": "ok", "initialized": true}
```

### Authentication Errors

If you get "Invalid or missing X-SECRET-KEY header":

1. Check if you set `SECRET` in Modal secret
2. If yes, include in requests:
   ```python
   configuration.api_key["SimpleSecretAuth"] = "your-secret"
   ```
3. If no, remove `SECRET` from Modal secret

### Out of Memory

If you see OOM errors, use larger GPU:

```python
@app.function(
    gpu="A100",  # 40GB or 80GB VRAM
    ...
)
```

## Next Steps

- See `MODAL_DEPLOYMENT.md` for full deployment guide
- See `MODAL_SECRETS.md` for secrets management
- Check `packages/python/neuronpedia-inference-client/` for more client examples
- Visit https://modal.com/docs for Modal documentation
