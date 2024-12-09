from enum import Enum
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, field_validator

import args
import modal


class DownloadType(str, Enum):
    single = "single"
    multipart = "multipart"
    snapshot = "snapshot"


class DownloadSettings(BaseModel):
    hf_path: Path
    revision: str | None = None
    download_type: DownloadType
    pet_name: str
    modelfile: str
    gpu: str | None = "t4:1"
    allow_patterns: list[str] | None = None

    @field_validator("hf_path", mode="before")
    def ensure_path_is_path_object(cls, v):
        return Path(v) if not isinstance(v, Path) else v

    @classmethod
    def from_config(cls, config_id):
        return cls(**args.DOWNLOAD[config_id])


class PullSettings(BaseModel):
    gpu: str | None = "t4:1"
    ollama_id: str

    @classmethod
    def from_config(cls, config_id):
        return cls(**args.PULL[config_id], ollama_id=config_id)


class AppSettings(BaseModel):
    MODELS_DIR: Path = Path("/models")
    MINUTES: int = 60
    HOURS: int = 60 * MINUTES
    download: DownloadSettings
    pull: PullSettings
    gpu: str | None = "t4:1"

    @field_validator("MODELS_DIR")
    def validate_models_dir(cls, v):
        assert v.is_absolute()
        return v

    @cached_property
    def MODELS_FOLDER_NAME(self):
        return str(self.MODELS_DIR)[1:]

    @cached_property
    def models_volume(self):
        return modal.Volume.from_name(str(self.MODELS_DIR)[1:], create_if_missing=True)

    @classmethod
    def init(cls):
        gpu = (
            args.DOWNLOAD[args.DOWNLOAD_DEFAULT]["gpu"]
            if args.CHOSEN_SOURCE == "download"
            else args.PULL[args.PULL_DEFAULT]["gpu"]
        )
        return cls(
            download=DownloadSettings.from_config(args.DOWNLOAD_DEFAULT),
            pull=PullSettings.from_config(args.PULL_DEFAULT),
            gpu=gpu,
        )


AppSettings = AppSettings.init()
