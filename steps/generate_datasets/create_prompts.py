from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.domain.dataset import DatasetType
from llm_engineering.domain.prompt import GeneratedDatasetSamplesPrompt
from llm_engineering.domain.types import DataCategory

@step
def create_prompts(
    documents: Annotated[list, "queried_cleaned_documents"], 
    dataset_type: Annotated[DatasetType, "dataset_type"]
) -> Annotated[dict[DataCategory, list[GeneratedDatasetSamplesPrompt]], "prompts"]:
    pass