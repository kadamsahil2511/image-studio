"""Flask routes for the local prompt expander."""

from flask import Flask, request

from prompt_expander import (check_ollama, expand_prompt, generate_image,
                             load_config, loaded_image_model, save_image)


app = Flask(__name__)


@app.get("/health")
def health():
    try:
        config = load_config()
        check_ollama(config)
        model = loaded_image_model(config)
        return {"ollama": "ok", "unsloth": "ok", "image_model": model}
    except RuntimeError as exc:
        return {"error": str(exc)}, 503


@app.post("/expand")
def expand():
    body = request.get_json(silent=True)
    idea = body.get("idea") if isinstance(body, dict) else None
    if not isinstance(idea, str) or not idea.strip():
        return {"error": "Enter an image idea."}, 422
    try:
        return {"prompt": expand_prompt(idea.strip(), load_config())}
    except RuntimeError as exc:
        return {"error": str(exc)}, 503


@app.post("/images")
def images():
    body = request.get_json(silent=True)
    prompt = body.get("prompt") if isinstance(body, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "Enter an image prompt."}, 422
    try:
        config = load_config()
        model = loaded_image_model(config)
        image = generate_image(prompt.strip(), model, config)
        return {"path": str(save_image(image, config))}
    except RuntimeError as exc:
        return {"error": str(exc)}, 503
