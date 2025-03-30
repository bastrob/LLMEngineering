from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MongoDB database
    DATABASE_HOST: str = "mongodb://llm_engineering:llm_engineering@127.0.0.1:27017"
    DATABASE_NAME: str = "llm_engineering_mongo"
    
    # LinkedIn Credentials
    LINKEDIN_USERNAME: str | None = None
    LINKEDIN_PASSWORD: str | None = None

    @classmethod
    def load_settings(cls) -> "Settings":
        settings = Settings()
        return settings
        



settings = Settings.load_settings()