from typing import Any

from typing_extensions import Annotated
from zenml import ArtifactConfig, get_step_context, step

from llm_engineering.application.dataset import generation
from llm_engineering.domain.dataset import DatasetType, InstructTrainTestSplit
from llm_engineering.domain.prompt import GenerateDatasetSamplesPrompt
from llm_engineering.domain.types import DataCategory

@step
def generate_instruction_dataset(
    prompts: Annotated[dict[DataCategory, list[GenerateDatasetSamplesPrompt]], "prompts"],
    test_split_size: Annotated[float, "test_split_size"],
    mock: Annotated[bool, "mock_generation"] = False,
) -> Annotated[
    InstructTrainTestSplit,
    ArtifactConfig(
        name="instruct_datasets",
        tags=["dataset", "instruct", "cleaned"]
    )
]:
    dataset_generator = generation.get_dataset_generator(DatasetType.INSTRUCTION)
    datasets = dataset_generator.generate(prompts, test_size=test_split_size, mock=mock)