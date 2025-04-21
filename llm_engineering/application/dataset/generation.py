from abc import ABC

import tiktoken
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger


from llm_engineering.domain.cleaned_documents import CleanedDocument
from llm_engineering.domain.dataset import DatasetType, TrainTestSplit
from llm_engineering.domain.prompt import GenerateDatasetSamplesPrompt, Prompt
from llm_engineering.domain.types import DataCategory
from llm_engineering.settings import settings

from . import utils as generation_utils


class DatasetGenerator(ABC):
    tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL_ID)
    dataset_type: DatasetType | None = None

    system_prompt_template = """You are a helpful assistant who generates {dataset_format} based on the given context. \
Provide your response in JSON format.
"""
    prompt_template_str: str | None = None
    
    @classmethod
    def get_system_prompt(cls) -> Prompt:
        assert cls.dataset_type is not None, "Dataset type must be set before calling get_system_prompt()"

        dataset_format = (
            "instruction-answer pairs" if cls.dataset_type == DatasetType.INSTRUCTION else "instruction-answer triples"
        )
        input_variables = {
            "dataset_format": dataset_format,
        }
        system_prompt = cls.system_prompt_template.format(**input_variables)

        return Prompt(
            template=cls.system_prompt_template,
            input_variables=input_variables,
            content=system_prompt,
        )

    @classmethod
    def get_prompts(cls, documents: list[CleanedDocument]) -> dict[DataCategory, list[GenerateDatasetSamplesPrompt]]:
        documents = generation_utils.extract_subtrings(documents)
        grouped_prompts = {}
        grouped_cleaned_documents = CleanedDocument.group_by_class(documents)
        for category, category_documents in grouped_cleaned_documents.items():
            category_prompts = [cls.get_prompt(document) for document in category_documents]
            grouped_prompts[category] = category_prompts
        
        return grouped_prompts
    
    @classmethod
    def get_prompt(cls, document: CleanedDocument) -> GenerateDatasetSamplesPrompt:
        assert cls.prompt_template_str is not None, "Prompt template must be set before calling get_prompt()"

        data_category = document.get_category()

        prompt_template = PromptTemplate.from_template(
            template=cls.prompt_template_str,
            template_format="jinja2",
        )

        input_variables = {
            "extract": document.content,
        }
        prompt = prompt_template.format(**input_variables)
        prompt_tokens = cls.tokenizer.encode(prompt)
        if len(prompt_tokens) > settings.OPENAI_MAX_TOKEN_WINDOW:
            prompt_tokens = prompt_tokens[: settings.OPENAI_MAX_TOKEN_WINDOW]
            prompt = cls.tokenizer.decode(prompt_tokens)
        
        prompt = GenerateDatasetSamplesPrompt(
            template=prompt_template.template,
            input_variables=input_variables,
            content=prompt,
            num_tokens=len(prompt_tokens),
            data_category=data_category,
            document=document,
        )

        return prompt
    
    @classmethod
    def generate(
        cls,
        prompts: dict[DataCategory, list[GenerateDatasetSamplesPrompt]],
        test_size: float = 0.2,
        mock: bool = False,
    ) -> TrainTestSplit:
        assert cls.dataset_type is not None, "Dataset type must be set before calling generate()"

        def _to_langchain(
            prompt: GenerateDatasetSamplesPrompt,
        ) -> list[BaseMessage]:
            messages = [
                SystemMessage(content=cls.get_system_prompt().content),
                HumanMessage(content=prompt)
            ]

            return messages

        if mock:
            pass
        else:
            assert settings.OPENAI_API_KEY is not None, "OpenAI API key must be set to generate datasets"


class InstructionDatasetGenerator(DatasetGenerator):
    dataset_type = DatasetType.INSTRUCTION

    prompt_template_str = """Based on the following extract, generate five instruction-answer pairs. Each instruction \
must ask to write about a specific topic contained in the context. Each answer \
must provide a relevant paragraph based on the information found in the \
context. Only use concepts from the context to generate the instructions. \
Instructions must never explicitly mention a context, a system, a course, or an extract. \
Instructions must be self-contained and general. \
Answers must imitate the writing style of the context. \
    
Example instruction: Explain the concept of an LLM Twin. \
Example answer: An LLM Twin is essentially an AI character that mimics your writing style, personality, and voice. \
It's designed to write just like you by incorporating these elements into a language model. \
The idea is to create a digital replica of your writing habits using advanced AI techniques. \

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {"instruction": "...", "answer": "..."},
    ...
]

Extract:
{{extract}}
"""


def get_dataset_generator(dataset_type: DatasetType) -> type[DatasetGenerator]:
    if dataset_type == DatasetType.INSTRUCTION:
        return InstructionDatasetGenerator
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")