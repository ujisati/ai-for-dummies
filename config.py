from functools import cached_property
from pydantic import BaseModel, field_validator
from pathlib import Path


class AppSettings(BaseModel):
    MODELS_DIR: Path = Path("/models")
    MINUTES: int = 60
    HOURS: int = 60 * MINUTES

    @field_validator("MODELS_DIR")
    def validate_models_dir(cls, v):
        assert v.is_absolute()
        return v

    @cached_property
    def MODELS_FOLDER_NAME(self):
        return str(self.MODELS_DIR)[1:]


class DownloadSettings(BaseModel):
    hf_path: Path
    revision: str | None = None
    multipart: bool = False
    pet_name: str
    modelfile: str

    @field_validator("hf_path", mode="before")
    def ensure_path_is_path_object(cls, v):
        return Path(v) if not isinstance(v, Path) else v

    @classmethod
    def from_config(cls, config_id):
        config_data = {
            "luminum": {
                "hf_path": "mradermacher/Luminum-v0.1-123B-i1-GGUF/Luminum-v0.1-123B.i1-IQ3_XS.gguf",
                "multipart": True,
                "pet_name": "Luminum",
                "modelfile": "luminum",
            },
            "qwen": {
                "hf_path": "Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-fp16.gguf",
                "pet_name": "Qwen",
                "modelfile": "qwen-test",
            },
        }
        return cls(**config_data[config_id])


AppSettings = AppSettings()
