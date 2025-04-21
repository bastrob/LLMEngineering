from enum import Enum

from llm_engineering.domain.base import VectorBaseDocument

class DatasetType(Enum):
    INSTRUCTION = "instruction"
    PREFERENCE = "preference"

class TrainTestSplit(VectorBaseDocument):
    pass

class InstructTrainTestSplit(TrainTestSplit):
    pass