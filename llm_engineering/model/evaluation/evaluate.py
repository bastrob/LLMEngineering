import concurrent.futures
import gc
import json
import os
from dotenv import load_dotenv

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from openai import OpenAI
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DATASET_HUGGINGFACE_WORKSPACE = os.environ["DATASET_HUGGINGFACE_WORKSPACE"]
MODEL_HUGGINGFACE_WORKSPACE = os.environ["MODEL_HUGGINGFACE_WORKSPACE"]
IS_DUMMY = os.environ.get("IS_DUMMY", False)


def check_if_huggingface_model_exists(model_id: str, default_value: str) -> str:
    api = HfApi()

    try:
        api.model_info(model_id)
        print(f"Found model on HF: '{model_id}'.")
    except RepositoryNotFoundError:
        print(f"Model '{model_id}' does not exist.")
        model_id = default_value
        print(f"Defaulting to '{model_id}'")
        print("Train your own model to avoid this behavior.")

    return model_id

def check_if_huggingface_dataset_exists(dataset_id: str, default_value: str) -> str:
    api = HfApi()

    try:
        api.dataset_info(dataset_id)
        print(f"Found dataset on HF: '{dataset_id}'.")
    except RepositoryNotFoundError:
        print(f"Dataset '{dataset_id}' does not exist.")
        dataset_id = default_value
        print(f"Defaulting to '{dataset_id}'")
        print("Use a valid dataset or create your own to avoid this behavior.")

    return dataset_id

model_ids = [
    check_if_huggingface_model_exists(
        f"{MODEL_HUGGINGFACE_WORKSPACE}/TwinLlama-3.1-8B", default_value="mlabonne/TwinLlama-3.1-8B"
    ),
    check_if_huggingface_model_exists(
        f"{MODEL_HUGGINGFACE_WORKSPACE}/TwinLlama-3.1-8B-DPO", default_value="mlabonne/TwinLlama-3.1-8B-DPO"
    ),
    "meta-llama/Llama-3.1-8B-Instruct",
]

if __name__ == "__main__":
    # Run generation
    for model_id in model_ids:
        dataset_name = check_if_huggingface_dataset_exists(
            f"{DATASET_HUGGINGFACE_WORKSPACE}/llmtwin", default_value="mlabonne/llmtwin"
        )
        generate_answers(model_id, dataset_name=dataset_name)

    # Run evaluation
    for model_id in model_ids:
        evaluate_answers(model_id)

    # Analyze results
    for model_id in model_ids:
        dataset = load_dataset(f"{DATASET_HUGGINGFACE_WORKSPACE}/{model_id.split('/')[-1]}-results", split="all")

        score = sum(dataset["accuracy"]) / len(dataset["accuracy"])
        print(f"{model_id.split('/')[-1]} - Accuracy: {score:.2f}")  # noqa

        score = sum(dataset["style"]) / len(dataset["style"])
        print(f"{model_id.split('/')[-1]} - Style: {score:.2f}")  # noqa