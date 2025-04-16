from zenml import pipeline

from llm_engineering.domain.dataset import DatasetType
from steps import generate_datasets as cd_steps


@pipeline
def generate_datasets(
    dataset_type: DatasetType = DatasetType.INSTRUCTION,
    test_split_size: float = 0.1,
    push_to_huggingface: bool = False,
    datased_id: str | None = None,
    mock: bool = False,
) -> None:
    cleaned_documents = cd_steps.query_feature_store()
    prompts = cd_steps.create_prompts(documents=cleaned_documents, dataset_type=dataset_type)
