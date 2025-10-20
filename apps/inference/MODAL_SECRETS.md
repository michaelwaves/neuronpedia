# Modal Secrets Reference for Neuronpedia Inference

## Single Secret: `neuronpedia-inference`

The inference server uses **one** Modal secret named `neuronpedia-inference` containing up to 3 environment variables.

## Quick Setup

### For Public Models (GPT-2)
```bash
# No secrets needed!
modal deploy apps/inference/modal_server.py
```

### For Gated Models (Llama, Gemma)
```bash
modal secret create neuronpedia-inference \
    HF_TOKEN=hf_your_huggingface_token_here
```

### With API Authentication
```bash
modal secret create neuronpedia-inference \
    HF_TOKEN=hf_your_token \
    SECRET=my-secure-api-key-123
```

### Full Setup (All Secrets)
```bash
modal secret create neuronpedia-inference \
    HF_TOKEN=hf_your_huggingface_token \
    SECRET=my-secure-api-key-123 \
    SENTRY_DSN=https://your-sentry-dsn@sentry.io/12345
```

## Environment Variables

| Variable | Required? | What it does | Example |
|----------|-----------|--------------|---------|
| **`HF_TOKEN`** | Optional | HuggingFace access token for gated models | `hf_AbCdEfGh123456...` |
| **`SECRET`** | Optional | API key for `X-SECRET-KEY` header auth | `my-secret-key-123` |
| **`SENTRY_DSN`** | Optional | Sentry error tracking endpoint | `https://...@sentry.io/123` |

## Where to Get These Values

### `HF_TOKEN` - HuggingFace Token
1. Go to https://huggingface.co/settings/tokens
2. Click "Create new token"
3. Select "Read" permissions
4. Copy the token (starts with `hf_`)

**Needed for:**
- Llama models (meta-llama/*)
- Gemma models (google/gemma-*)
- Any gated/private models

**Not needed for:**
- GPT-2 (public)
- Most public models on HuggingFace

### `SECRET` - API Authentication Key
- **This is YOUR choice** - can be any string
- Used to protect your API endpoint
- Clients must send `X-SECRET-KEY: your_secret` header

**Example:**
```bash
# You set this
modal secret create neuronpedia-inference SECRET=banana-pancake-42

# Clients must use it
curl -H "X-SECRET-KEY: banana-pancake-42" https://your-url.modal.run/health
```

**When to use:**
- ✅ Production deployments (prevent unauthorized access)
- ✅ Public-facing endpoints
- ❌ Development/testing (skip it for simplicity)

### `SENTRY_DSN` - Error Tracking
1. Create account at https://sentry.io
2. Create a new project
3. Copy the DSN from project settings
4. Format: `https://[key]@[org].ingest.sentry.io/[project]`

**When to use:**
- ✅ Production (track errors remotely)
- ❌ Development (not needed)

## Managing Secrets

### Create
```bash
modal secret create neuronpedia-inference HF_TOKEN=hf_abc123
```

### Update (add/change variables)
```bash
modal secret update neuronpedia-inference SECRET=new-secret-key
```

### List all secrets
```bash
modal secret list
```

### Delete
```bash
modal secret delete neuronpedia-inference
```

### View (doesn't show values, just names)
```bash
modal secret list neuronpedia-inference
```

## How Secrets Work in Code

In `modal_server.py`:
```python
@app.function(
    secrets=[modal.Secret.from_name("neuronpedia-inference")],
    ...
)
```

Inside the container:
```python
import os

# These are automatically available as environment variables
hf_token = os.getenv("HF_TOKEN")        # From secret
api_secret = os.getenv("SECRET")         # From secret
sentry_dsn = os.getenv("SENTRY_DSN")     # From secret
```

## Common Scenarios

### Scenario 1: Just Testing GPT-2
```bash
# No secrets needed!
modal deploy apps/inference/modal_server.py
```

### Scenario 2: Using Llama-3.1-8B
```bash
# Need HF token for gated model
modal secret create neuronpedia-inference \
    HF_TOKEN=hf_your_token_from_huggingface

# Edit modal_server.py to use Llama
# Then deploy
modal deploy apps/inference/modal_server.py
```

### Scenario 3: Production Deployment with Auth
```bash
# Full setup with authentication
modal secret create neuronpedia-inference \
    HF_TOKEN=hf_your_token \
    SECRET=$(openssl rand -hex 32)  # Generate random secret

# Deploy
modal deploy apps/inference/modal_server.py

# Test (must include auth header)
curl -H "X-SECRET-KEY: your_generated_secret" \
     https://your-url.modal.run/health
```

### Scenario 4: Forgot to Set HF_TOKEN
```bash
# Error: "401 Unauthorized" when downloading model

# Fix: Update secret
modal secret create neuronpedia-inference HF_TOKEN=hf_your_token

# Redeploy (picks up new secret)
modal deploy apps/inference/modal_server.py
```

## Security Best Practices

1. **Never commit secrets to git**
   ```bash
   # Bad
   export HF_TOKEN=hf_abc123
   git add .env

   # Good
   modal secret create neuronpedia-inference HF_TOKEN=hf_abc123
   ```

2. **Use strong SECRET values**
   ```bash
   # Bad
   SECRET=123

   # Good
   SECRET=$(openssl rand -hex 32)  # Generates: 8f3a2b...
   ```

3. **Rotate secrets regularly**
   ```bash
   # Every 90 days
   modal secret update neuronpedia-inference SECRET=new_random_value
   ```

4. **Use different secrets per environment**
   ```bash
   # Development
   modal secret create neuronpedia-inference-dev HF_TOKEN=hf_dev_token

   # Production
   modal secret create neuronpedia-inference HF_TOKEN=hf_prod_token
   ```

## Troubleshooting

### "Secret not found: neuronpedia-inference"
```bash
# Create it
modal secret create neuronpedia-inference HF_TOKEN=hf_token
```

### "401 Unauthorized" from HuggingFace
```bash
# HF_TOKEN is missing or invalid
# Update it
modal secret update neuronpedia-inference HF_TOKEN=hf_new_valid_token

# Verify token at: https://huggingface.co/settings/tokens
```

### "Invalid or missing X-SECRET-KEY header"
```bash
# You set SECRET in modal secret, must use it in requests

# Option 1: Include header
curl -H "X-SECRET-KEY: your_secret" https://url.modal.run/health

# Option 2: Remove SECRET from secret (disables auth)
modal secret delete neuronpedia-inference
modal secret create neuronpedia-inference HF_TOKEN=hf_token  # no SECRET
```

### Secrets not updating
```bash
# After updating secrets, redeploy
modal deploy apps/inference/modal_server.py
```

## Summary

**TL;DR:**
- Secret name: `neuronpedia-inference`
- Contains: `HF_TOKEN`, `SECRET`, `SENTRY_DSN` (all optional)
- Create with: `modal secret create neuronpedia-inference KEY=value`
- Most common: Just `HF_TOKEN` for gated models
- Skip entirely for public models like GPT-2
