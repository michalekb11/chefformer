from pydantic import model_validator
from pydantic_settings import BaseSettings
from configs.shared.settings import BasePromptSettings, ModelSettings, get_latest_checkpoint

class FrontEndSettings(BaseSettings):
    wallpaper_path: str = "app/wallpaper/cuttingboard.png"
    chat_url: str = ""  # Will be synced from AppSettings

class APISettings(BaseSettings):
    chat_url: str = ""  # Will be synced from AppSettings
    checkpoint_path: str | None = None

class AppSettings(BaseSettings):
    # Single Source of Truth
    chat_url: str = "http://127.0.0.1:8000/v1/chat/generate"
    front_end: FrontEndSettings = FrontEndSettings()
    api: APISettings = APISettings()
    model: ModelSettings = ModelSettings()
    prompt: BasePromptSettings = BasePromptSettings()

    @model_validator(mode='after')
    def sync_chat_urls(self) -> 'AppSettings':
        """Ensures all sub-settings use the shared chat_url."""
        self.front_end.chat_url = self.chat_url
        self.api.chat_url = self.chat_url
        return self
    
    @model_validator(mode='after')
    def set_default_checkpoint(self) -> 'AppSettings':
        """Automatically discovers the latest checkpoint if path is not explicitly provided."""
        if self.api.checkpoint_path is None:
            self.api.checkpoint_path = get_latest_checkpoint("./checkpoints", "finetune")
        return self

app_settings = AppSettings()

if __name__ == '__main__':
    print(app_settings)