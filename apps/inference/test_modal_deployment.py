"""
Test script for Modal-deployed inference server.

Usage:
    # First deploy the server
    modal deploy modal_server.py

    # Then run this test
    python test_modal_deployment.py
"""

import neuronpedia_inference_client
from neuronpedia_inference_client.rest import ApiException
from pprint import pprint

MODAL_URL = "https://michaelwaves--neuronpedia-inference-serve-dev.modal.run/v1"

configuration = neuronpedia_inference_client.Configuration(
    host=MODAL_URL
)

configuration.api_key["SimpleSecretAuth"] = "Solvealignment!1"

with neuronpedia_inference_client.ApiClient(configuration) as api_client:
    api_instance = neuronpedia_inference_client.DefaultApi(api_client)

    activation_all_post_request = neuronpedia_inference_client.ActivationAllPostRequest(
        prompt="Hello world",
        model="gpt2-small",
        source_set="res-jb",
        selected_sources=[],
        sort_by_token_indexes=[],
        ignore_bos=True,
        num_results=10,
    )

    try:
        print("Testing activation endpoint...")
        api_response = api_instance.activation_all_post(activation_all_post_request)
        print("✓ Success! Response:")
        pprint(api_response)
    except ApiException as e:
        print(f"✗ Exception: {e}")
