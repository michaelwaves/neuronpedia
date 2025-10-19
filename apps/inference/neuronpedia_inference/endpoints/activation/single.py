import logging
from typing import Any

import einops
import torch
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from neuronpedia_inference_client.models.activation_single_post200_response import (
    ActivationSinglePost200Response,
)
from neuronpedia_inference_client.models.activation_single_post200_response_activation import (
    ActivationSinglePost200ResponseActivation,
)
from neuronpedia_inference_client.models.activation_single_post_request import (
    ActivationSinglePostRequest,
)
from transformer_lens import ActivationCache, HookedTransformer

from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import Model, with_request_lock

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/activation/single")
@with_request_lock()
async def activation_single(
    request: ActivationSinglePostRequest = Body(
        ...,
        example={
            "prompt": "The Jedi in Star Wars wield lightsabers.",
            "model": "gpt2-small",
            "source": "0-res-jb",
            "index": "14057",
        },
    ),
):
    model = Model.get_instance()
    config = Config.get_instance()
    sae_manager = SAEManager.get_instance()
    # Ensure exactly one of features or vector is provided
    if (request.source is not None and request.index is not None) == (
        request.vector is not None and request.hook is not None
    ):
        logger.error(
            "Invalid request data: exactly one of layer/index or vector must be provided"
        )
        return JSONResponse(
            content={
                "error": "Invalid request data: exactly one of layer/index or vector must be provided"
            },
            status_code=400,
        )

    prompt = request.prompt

    if request.source is not None and request.index is not None:
        source = request.source
        layer_num = get_layer_num_from_sae_id(source)
        index = int(request.index)

        sae = sae_manager.get_sae(source)

        # TODO: we assume that if either SAE or model prepends bos, then we should prepend bos
        # this is not exactly correct, but sometimes the SAE doesn't have the prepend_bos flag set
        prepend_bos = sae.cfg.metadata.prepend_bos or model.cfg.tokenizer_prepends_bos

        tokens = model.to_tokens(
            prompt,
            prepend_bos=prepend_bos,
            truncate=False,
        )[0]

        if len(tokens) > config.token_limit:
            logger.error(
                "Text too long: %s tokens, max is %s",
                len(tokens),
                config.token_limit,
            )
            return JSONResponse(
                content={
                    "error": f"Text too long: {len(tokens)} tokens, max is {config.token_limit}"
                },
                status_code=400,
            )

        str_tokens: list[str] = model.to_str_tokens(prompt, prepend_bos=prepend_bos)  # type: ignore
        result, cache = process_activations_with_cache(model, source, index, tokens)

        # Calculate DFA if enabled
        if sae_manager.is_dfa_enabled(source):
            dfa_result = calculate_dfa_from_cache(
                model,
                sae,
                layer_num,
                index,
                result.max_value_index,
                cache,
            )
            result.dfa_values = dfa_result["dfa_values"]  # type: ignore
            result.dfa_target_index = dfa_result["dfa_target_index"]  # type: ignore
            result.dfa_max_value = dfa_result["dfa_max_value"]  # type: ignore

    else:
        vector = request.vector
        hook = request.hook
        prepend_bos = model.cfg.tokenizer_prepends_bos
        tokens = model.to_tokens(
            prompt,
            prepend_bos=prepend_bos,
            truncate=False,
        )[0]
        if len(tokens) > config.token_limit:
            logger.error(
                "Text too long: %s tokens, max is %s",
                len(tokens),
                config.token_limit,
            )
            return JSONResponse(
                content={
                    "error": f"Text too long: {len(tokens)} tokens, max is {config.token_limit}"
                },
                status_code=400,
            )

        str_tokens: list[str] = model.to_str_tokens(prompt, prepend_bos=prepend_bos)  # type: ignore
        # Optimize: extract layer number from hook name and stop at that layer
        stop_at_layer = _get_stop_layer_from_hook(hook, model.cfg.n_layers)
        _, cache = model.run_with_cache(tokens, stop_at_layer=stop_at_layer)
        result = process_vector_activations(vector, cache, hook, sae_manager.device)  # type: ignore

    logger.info("Returning result: %s", result)

    return ActivationSinglePost200Response(activation=result, tokens=str_tokens)


def _get_safe_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Convert float16 to float32, leave other dtypes unchanged.
    """
    return torch.float32 if dtype == torch.float16 else dtype


def _safe_cast(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """
    Safely cast a tensor to the target dtype, creating a copy if needed.
    Convert float16 to float32, leave other dtypes unchanged.
    """
    safe_dtype = _get_safe_dtype(tensor.dtype)
    if safe_dtype != tensor.dtype or safe_dtype != target_dtype:
        return tensor.to(target_dtype)
    return tensor


def get_layer_num_from_sae_id(sae_id: str) -> int:
    return int(sae_id.split("-")[0]) if not sae_id.isdigit() else int(sae_id)


def _get_stop_layer_from_hook(hook_name: str, n_layers: int) -> int | None:
    """
    Extract layer number from hook name and return stop_at_layer parameter.
    Returns None if we need to run through all layers.

    Examples:
    - "blocks.5.hook_resid_post" -> 6
    - "blocks.10.mlp.hook_post" -> 11
    """
    import re

    pattern = r"blocks\.(\d+)\."
    match = re.search(pattern, hook_name)
    if match:
        layer_num = int(match.group(1))
        stop_at_layer = layer_num + 1
        return stop_at_layer if stop_at_layer < n_layers else None
    return None


def process_activations(
    model: HookedTransformer, layer: str, index: int, tokens: torch.Tensor
) -> ActivationSinglePost200ResponseActivation:
    """Legacy function for backward compatibility. Use process_activations_with_cache instead."""
    result, _ = process_activations_with_cache(model, layer, index, tokens)
    return result


def process_activations_with_cache(
    model: HookedTransformer, layer: str, index: int, tokens: torch.Tensor
) -> tuple[ActivationSinglePost200ResponseActivation, ActivationCache]:
    """
    Process activations and return both the result and the cache.
    This allows reusing the cache for DFA calculations without re-running the forward pass.
    """
    sae_manager = SAEManager.get_instance()
    # Optimize: stop forward pass at the layer we need
    layer_num = get_layer_num_from_sae_id(layer)
    stop_at_layer = layer_num + 1 if layer_num + 1 < model.cfg.n_layers else None
    _, cache = model.run_with_cache(tokens, stop_at_layer=stop_at_layer)
    hook_name = sae_manager.get_sae_hook(layer)
    sae_type = sae_manager.get_sae_type(layer)

    if sae_type == "neurons":
        result = process_neuron_activations(cache, hook_name, index, sae_manager.device)
    elif sae_manager.get_sae(layer) is not None:
        result = process_feature_activations(
            sae_manager.get_sae(layer),
            sae_type,
            cache,
            hook_name,
            index,
        )
    else:
        raise ValueError(f"Invalid layer: {layer}")

    return result, cache


def process_neuron_activations(
    cache: ActivationCache | dict[str, torch.Tensor],
    hook_name: str,
    index: int,
    device: str,
) -> ActivationSinglePost200ResponseActivation:
    mlp_activation_data = cache[hook_name].to(device)
    values = torch.transpose(mlp_activation_data[0], 0, 1)[index].detach().tolist()
    max_value = max(values)
    return ActivationSinglePost200ResponseActivation(
        values=values,
        max_value=max_value,
        max_value_index=values.index(max_value),
    )


def process_feature_activations(
    sae: Any,
    sae_type: str,
    cache: ActivationCache | dict[str, torch.Tensor],
    hook_name: str,
    index: int,
) -> ActivationSinglePost200ResponseActivation:
    if sae_type == "saelens-1":
        return process_saelens_activations(sae, cache, hook_name, index)
    raise ValueError(f"Unsupported SAE type: {sae_type}")


def process_saelens_activations(
    sae: Any,
    cache: ActivationCache | dict[str, torch.Tensor],
    hook_name: str,
    index: int,
) -> ActivationSinglePost200ResponseActivation:
    feature_acts = sae.encode(cache[hook_name])
    values = torch.transpose(feature_acts.squeeze(0), 0, 1)[index].detach().tolist()
    max_value = max(values)
    return ActivationSinglePost200ResponseActivation(
        values=values,
        max_value=max_value,
        max_value_index=values.index(max_value),
    )


def process_vector_activations(
    vector: torch.Tensor | list[float],
    cache: ActivationCache | dict[str, torch.Tensor],
    hook_name: str,
    device: torch.device,
) -> ActivationSinglePost200ResponseActivation:
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, device=device)
    # not normalizing it for now
    # vector = vector / torch.linalg.norm(vector)
    activations = cache[hook_name].to(device)
    # ensure vector has the same dtype as activations
    vector = vector.to(dtype=activations.dtype)
    feature_acts = torch.matmul(activations, vector)
    values = feature_acts.squeeze(0).detach().tolist()
    max_value = max(values)
    return ActivationSinglePost200ResponseActivation(
        values=values,
        max_value=max_value,
        max_value_index=values.index(max_value),
    )


def calculate_dfa(
    model: HookedTransformer,
    sae: Any,
    layer_num: int,
    index: int,
    max_value_index: int,
    tokens: torch.Tensor,
) -> dict[str, list[float] | int | float]:
    """Legacy function for backward compatibility. Use calculate_dfa_from_cache instead."""
    _, cache = model.run_with_cache(tokens)
    return calculate_dfa_from_cache(
        model, sae, layer_num, index, max_value_index, cache
    )


def calculate_dfa_from_cache(
    model: HookedTransformer,
    sae: Any,
    layer_num: int,
    index: int,
    max_value_index: int,
    cache: ActivationCache,
) -> dict[str, list[float] | int | float]:
    """
    Calculate DFA values from an existing activation cache.
    This avoids re-running the forward pass when we already have the cache.
    """
    v = cache["v", layer_num]  # [batch, src_pos, n_heads, d_head]
    attn_weights = cache["pattern", layer_num]  # [batch, n_heads, dest_pos, src_pos]

    # Determine the safe dtype for operations
    v_dtype = _get_safe_dtype(v.dtype)
    attn_weights_dtype = _get_safe_dtype(attn_weights.dtype)
    sae_dtype = _get_safe_dtype(sae.W_enc.dtype)

    # Use the highest precision dtype
    op_dtype = max(v_dtype, attn_weights_dtype, sae_dtype, key=lambda x: x.itemsize)

    # Check if the model uses GQA
    use_gqa = (
        hasattr(model.cfg, "n_key_value_heads")
        and model.cfg.n_key_value_heads is not None
        and model.cfg.n_key_value_heads < model.cfg.n_heads
    )

    if use_gqa:
        n_query_heads = attn_weights.shape[1]
        n_kv_heads = v.shape[2]
        expansion_factor = n_query_heads // n_kv_heads
        v = v.repeat_interleave(expansion_factor, dim=2)

    # Cast tensors to operation dtype
    v = _safe_cast(v, op_dtype)
    attn_weights = _safe_cast(attn_weights, op_dtype)

    v_cat = einops.rearrange(
        v, "batch src_pos n_heads d_head -> batch src_pos (n_heads d_head)"
    )
    attn_weights_bcast = einops.repeat(
        attn_weights,
        "batch n_heads dest_pos src_pos -> batch dest_pos src_pos (n_heads d_head)",
        d_head=model.cfg.d_head,
    )
    decomposed_z_cat = attn_weights_bcast * v_cat.unsqueeze(1)

    # Cast SAE weights to operation dtype
    W_enc = _safe_cast(sae.W_enc[:, index], op_dtype)

    per_src_pos_dfa = einops.einsum(
        decomposed_z_cat,
        W_enc,
        "batch dest_pos src_pos d_model, d_model -> batch dest_pos src_pos",
    )
    per_src_dfa = per_src_pos_dfa[torch.arange(1), torch.tensor([max_value_index]), :]
    dfa_values = per_src_dfa[0].tolist()
    return {
        "dfa_values": dfa_values,
        "dfa_target_index": max_value_index,
        "dfa_max_value": max(dfa_values),
    }
