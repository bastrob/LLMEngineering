from llm_engineering.application.preprocessing.operations.chunking import chunk_document
from llm_engineering.domain.cleaned_documents import CleanedDocument

def extract_subtrings(
        documents: list[CleanedDocument], min_length: int = 1000, max_length: int = 2000
) -> list[CleanedDocument]:
    extracts = []
    for document in documents:
        document_extracts = chunk_document(document.content, min_length, max_length)
        for extract in document_extracts:
            subdocument = document.model_copy()
            subdocument.content = extract

            extracts.append(subdocument)
    
    return extracts