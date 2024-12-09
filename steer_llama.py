import control_prompts
import modal
from settings import AppSettings

volume = AppSettings.models_volume

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "pydantic",
            "transformers",
            "sentencepiece",
            "huggingface-hub",
            "numpy==1.26.3",
            "scikit-learn==1.4.0",
            "torch==2.1.2",
            "transformers==4.36.2",
            "accelerate==0.26.1",
            "tqdm==4.66.1",
            "gguf==0.10.0",
        ]
    ).run_commands(
        "pip install repeng --no-deps",
    )
)

app = modal.App(image=image, secrets=[modal.Secret.from_name("huggingface-secret")])


@app.function(
    volumes={str(AppSettings.MODELS_DIR): volume},
    gpu="l4",
    timeout=4 * AppSettings.HOURS,
)
def steer_llama(filename: str, load: bool = False):
    import json
    import logging
    from pathlib import Path

    import torch
    from repeng import ControlModel, ControlVector, DatasetEntry
    from repeng.control import model_layer_list
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def setup_model(model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, gguf_file="Luminum-v0.1-123B.i1-IQ3_XS.gguf"
        )
        tokenizer.pad_token_id = 0

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            gguf_file="Luminum-v0.1-123B.i1-IQ3_XS.gguf",
            torch_dtype=torch.float16,
        )
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model = ControlModel(model, range(-1, -10, -2))
        return model, tokenizer

    def print_model_layers(model):
        layers = model_layer_list(model)
        for i, layer in enumerate(layers):
            print(i, layer.__class__.__name__)
        return layers

    def build_dataset(tokenizer, user_tag: str, asst_tag: str):
        positive_things = ["tiktok brainrot"]
        negative_things = ["love of learning"]
        things = zip(positive_things, negative_things)

        dataset = []
        for prompt_template in control_prompts.templates:
            tokens = tokenizer.tokenize(prompt_template)
            prompt = tokenizer.convert_tokens_to_string(tokens)
            for positive_thing, negative_thing in things:
                dataset.append(
                    DatasetEntry(
                        positive=prompt.format(
                            user_tag=user_tag, asst_tag=asst_tag, concept=positive_thing
                        ),
                        negative=prompt.format(
                            user_tag=user_tag, asst_tag=asst_tag, concept=negative_thing
                        ),
                    )
                )
        return dataset

    def train_control_vector(model, tokenizer, dataset):
        model.reset()
        control_vector = ControlVector.train(
            model, tokenizer, dataset, method="pca_center", batch_size=200
        )
        return control_vector

    def save_control_vector(control_vector, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        control_vector.export_gguf(str(save_path))
        volume.commit()

    def load_control_vector(load_path: Path):
        return ControlVector.import_gguf(str(load_path))

    def run_inference(model, tokenizer, control_vector):
        user_tag, asst_tag = "[INST]", "[/INST]"
        input_text = (
            f"{user_tag} Fruits vs vegetable war: you decide winner. {asst_tag}"
        )
        input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
        settings = {
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": True,
            "max_new_tokens": 150,
            "repetition_penalty": 1.2,
        }

        print("==baseline")
        model.reset()
        baseline_output = tokenizer.decode(
            model.generate(**input_ids, **settings).squeeze()
        )
        print(baseline_output)

        print("\n++control")
        model.set_control(control_vector, 2.2)
        positive_output = tokenizer.decode(
            model.generate(**input_ids, **settings).squeeze()
        )
        print(positive_output)
        model.reset()

        print("\n--control")
        model.set_control(control_vector, -2.2)
        negative_output = tokenizer.decode(
            model.generate(**input_ids, **settings).squeeze()
        )
        print(negative_output)
        model.reset()

    model_name = "/models/mradermacher/Luminum-v0.1-123B-i1-GGUF"
    user_tag, asst_tag = "[INST]", "[/INST]"

    model, tokenizer = setup_model(model_name)

    dataset = build_dataset(tokenizer, user_tag, asst_tag)

    control_vectors_dir = Path("/models/control_vectors")
    control_vectors_dir.mkdir(parents=True, exist_ok=True)
    file_path = control_vectors_dir / filename

    if load:
        if not file_path.exists():
            raise FileNotFoundError(
                f"The file '{file_path}' does not exist for loading."
            )
        control_vector = load_control_vector(file_path)
        print(f"Control vector loaded from '{file_path}'.")
    else:
        control_vector = train_control_vector(model, tokenizer, dataset)
        save_control_vector(control_vector, file_path)
        print(f"Control vector trained and saved to '{file_path}'.")

    run_inference(model, tokenizer, control_vector)
