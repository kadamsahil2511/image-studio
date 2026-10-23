"""Expand a short idea locally and generate one pixel art image."""

import base64
import binascii
import json
import subprocess
import sys
import uuid
import zlib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_config():
    try:
        lines = Path(".env").read_text().splitlines()
    except FileNotFoundError as exc:
        raise RuntimeError("Create .env from .env.example and set UNSLOTH_API_KEY.") from exc
    config = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RuntimeError(f"Invalid .env line: {line}")
        name, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        config[name.strip()] = value
    required = ("UNSLOTH_API_KEY", "UNSLOTH_API_URL", "OLLAMA_API_URL", "OLLAMA_MODEL",
                "IMAGE_MODEL_FAMILY", "PROMPT_INSTRUCTION", "OUTPUT_DIR", "PREVIEW_APP")
    for name in required:
        if not config.get(name):
            raise RuntimeError(f"Set {name} in .env.")
    return config


def api_json(url, label, payload=None, key=None, timeout=120):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(url, data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        exc.close()
        raise RuntimeError(f"{label} returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach {label}: {exc.reason}") from exc
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError(f"{label} returned invalid JSON") from exc


def valid_png(data):
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    position = 8
    chunks = []
    while position + 12 <= len(data):
        length = int.from_bytes(data[position:position + 4], "big")
        end = position + 12 + length
        if end > len(data):
            return False
        kind = data[position + 4:position + 8]
        content = data[position + 4:end - 4]
        if zlib.crc32(content) != int.from_bytes(data[end - 4:end], "big"):
            return False
        chunks.append(kind)
        position = end
        if kind == b"IEND":
            break
    return (position == len(data) and len(chunks) >= 3 and
            chunks[0] == b"IHDR" and b"IDAT" in chunks and chunks[-1] == b"IEND")


def check_ollama(config):
    result = api_json(f"{config['OLLAMA_API_URL'].rstrip('/')}/api/tags", "Ollama")
    models = result.get("models") if isinstance(result, dict) else None
    if not isinstance(models, list) or not any(
        isinstance(entry, dict) and entry.get("name") == config["OLLAMA_MODEL"]
        for entry in models
    ):
        raise RuntimeError(f"Ollama model {config['OLLAMA_MODEL']} is unavailable.")


def loaded_image_model(config):
    key = config["UNSLOTH_API_KEY"]
    models = api_json(f"{config['UNSLOTH_API_URL'].rstrip('/')}/v1/models", "Unsloth", key=key)
    if not isinstance(models, dict) or not isinstance(models.get("data"), list):
        raise RuntimeError("Unsloth returned an invalid model list.")
    model = next((entry["id"] for entry in models["data"]
                  if isinstance(entry, dict) and entry.get("loaded") is True
                  and isinstance(entry.get("id"), str)
                  and config["IMAGE_MODEL_FAMILY"].lower() in entry["id"].lower()), None)
    if not model:
        raise RuntimeError(f"Load a {config['IMAGE_MODEL_FAMILY']} image model in Unsloth Desktop first.")
    return model


def expand_prompt(idea, config):
    instruction = f"{config['PROMPT_INSTRUCTION']}\n\nIdea: {idea}"
    result = api_json(f"{config['OLLAMA_API_URL'].rstrip('/')}/api/generate", "Ollama", {
        "model": config["OLLAMA_MODEL"], "prompt": instruction, "stream": False, "keep_alive": 0,
    })
    prompt = result.get("response", "").strip() if isinstance(result, dict) and isinstance(result.get("response"), str) else ""
    if not prompt:
        raise RuntimeError("Ollama returned an empty prompt.")
    return prompt


def generate_image(prompt, model, config):
    result = api_json(f"{config['UNSLOTH_API_URL'].rstrip('/')}/v1/images/generations", "Unsloth", {
        "model": model, "prompt": prompt, "n": 1, "response_format": "b64_json",
    }, key=config["UNSLOTH_API_KEY"], timeout=1800)
    try:
        image = base64.b64decode(result["data"][0]["b64_json"], validate=True)
    except (KeyError, IndexError, TypeError, ValueError, binascii.Error) as exc:
        raise RuntimeError("Unsloth returned invalid image data.") from exc
    if not valid_png(image):
        raise RuntimeError("Unsloth returned invalid PNG data.")
    return image


def save_image(image, config):
    output = Path(config["OUTPUT_DIR"])
    output.mkdir(exist_ok=True)
    path = output / f"pixel-anime-{uuid.uuid4().hex}.png"
    path.write_bytes(image)
    return path


def open_image(path, config):
    try:
        subprocess.run(["open", "-a", config["PREVIEW_APP"], str(path)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Could not open {config['PREVIEW_APP']}: {exc}. Image saved at {path}", file=sys.stderr)


def main():
    idea = input("Image idea: ").strip()
    if not idea:
        raise RuntimeError("Enter an image idea.")
    config = load_config()
    model = loaded_image_model(config)
    prompt = expand_prompt(idea, config)
    print(f"\nExpanded prompt:\n{prompt}\n")
    path = save_image(generate_image(prompt, model, config), config)
    print(f"Saved image: {path}")
    open_image(path, config)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
