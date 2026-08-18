from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Gatewatch"
    database_url: str = "sqlite:///./db/gatewatch.db"
    jwt_secret: str = "change-this-before-production"
    token_expire_minutes: int = 480
    upload_dir: Path = Path("db/gatewatch-media")
    recognition_threshold: float = 0.32
    frame_queue_limit: int = 8
    redis_url: str = "redis://127.0.0.1:6379/0"
    demo_admin_username: str = "admin"
    demo_admin_password: str = "change-me-now"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="GATEWATCH_")


settings = Settings()