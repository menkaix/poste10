from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    backlog_service_url: str

    imap_host: str
    imap_port: int
    imap_username: str
    imap_password: str

    google_api_key: str

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
