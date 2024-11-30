from ast import Module
from pathlib import Path
from config import DownloadSettings, AppSettings
import modal


volume = modal.Volume.from_name(str(AppSettings.MODELS_DIR)[1:], create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "huggingface_hub",
            "hf-transfer",
            "pydantic",
        ]
    )
    .apt_install("curl", "lshw")
    .add_local_dir("modelfiles", "/modelfiles", copy=True)
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "OLLAMA_HOST": "127.0.0.1:11434",
            "OLLAMA_MODELS": "/models/.ollama",
        }
    )
)

app = modal.App(image=image, secrets=[modal.Secret.from_name("huggingface-secret")])


@app.function(
    volumes={str(AppSettings.MODELS_DIR): volume}, timeout=4 * AppSettings.HOURS
)
def download(config_id):
    import os
    from huggingface_hub import hf_hub_download, HfFileSystem

    settings = DownloadSettings.from_config(config_id)
    hf_path, revision, multipart = (
        settings.hf_path,
        settings.revision,
        settings.multipart,
    )

    volume.reload()

    final_filename = hf_path.name

    repo_id = "/".join([part for part in hf_path.parts if part not in ("", "/")][0:2])

    if multipart:
        fs = HfFileSystem()
        part_filenames = [
            file.split("/")[-1]
            for file in fs.glob(str(hf_path) + ".part*of*", revision=revision)
        ]
        if not part_filenames:
            raise Exception("No files found.")

        for filename in part_filenames:
            print(f"Downloading {filename}")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=str(AppSettings.MODELS_DIR / repo_id),
            )
            print(f"Downloaded {filename}")
            volume.commit()

        volume.reload()
        final_filepath = AppSettings.MODELS_DIR / repo_id / final_filename
        with open(final_filepath, "wb") as outfile:
            print("Concatenating parts.")
            for filename in part_filenames:
                with open(AppSettings.MODELS_DIR / repo_id / filename, "rb") as infile:
                    outfile.write(infile.read())
                print(f"Appended {filename} to {final_filename}")

        print(f"Reconstructed {final_filepath} successfully.")
    else:
        print(f"Downloading snapshot for {hf_path}")
        hf_hub_download(
            repo_id=repo_id,
            filename=final_filename,
            revision=revision,
            local_dir=AppSettings.MODELS_DIR / repo_id,
        )
        print(f"Downloaded snapshot for {hf_path}")

    volume.commit()


@app.function(
    volumes={str(AppSettings.MODELS_DIR): volume},
    gpu=modal.gpu.A100(count=1),
    timeout=4 * AppSettings.HOURS,
)
def compile(config_id: str):
    import subprocess
    import time

    settings = DownloadSettings.from_config(config_id)
    subprocess.Popen(["ollama", "serve"], close_fds=True)
    time.sleep(2)
    subprocess.run(
        [
            "ollama",
            "create",
            settings.pet_name,
            "-f",
            f"/modelfiles/{settings.modelfile}",
        ],
        check=True,
    )
    volume.commit()


@app.local_entrypoint()
def main(config_id):
    download.remote(config_id)
    compile.remote(config_id)
