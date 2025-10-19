import neuronpedia_inference_client
from neuronpedia_inference_client.rest import ApiException
from pprint import pprint

configuration = neuronpedia_inference_client.Configuration(
    host="http://localhost:5002/v1"
)

configuration.api_key["SimpleSecretAuth"] = "localhost-secret"

with neuronpedia_inference_client.ApiClient(configuration) as api_client:
    api_instance = neuronpedia_inference_client.DefaultApi(api_client)

    # Create request with keyword arguments (Pydantic models don't accept positional args)
    activation_all_post_request = neuronpedia_inference_client.ActivationAllPostRequest(
        prompt="Hello world",
        model="gpt2-small",
        source_set="res-jb",
        selected_sources=[],  # Empty list means use all sources in the set
        sort_by_token_indexes=[],  # Empty list means sort by max activation
        ignore_bos=True,
        num_results=10  # Optional: get top 10 results
    )

    try:
        api_response = api_instance.activation_all_post(activation_all_post_request)
        print("The response of DefaultApi->activation_all_post:\n")
        pprint(api_response)
    except ApiException as e:
        print("Exception when calling DefaultApi->activation_all_post: %s\n" % e)