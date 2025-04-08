from zenml import pipeline

from steps.etl import feature_engineering as fe_steps

@pipeline
def feature_engineering(author_full_names: list[str]) -> None:
    # Extract raw documents.
    raw_documents = fe_steps.query_data_warehouse(author_full_names)

    # Cleaning.
    cleaned_documents = fe_steps.clean_documents(raw_documents)
    # Push to store
    last_step_1 = fe_steps.load_to_vector_db(cleaned_documents)
    
    # Chunking.
    # Embedding.
    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
    # Push to store
    last_step_2 = fe_steps.load_to_vector_db(embedded_documents)
    
    return [last_step_1.invocation_id, last_step_2.invocation_id]