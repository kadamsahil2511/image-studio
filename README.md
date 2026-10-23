# Image Studio

In Unsloth Desktop, load a FLUX.2 image model and create an API key in **Settings → API keys**. Keep Unsloth running locally. Ollama must be running locally with the model named in `.env` installed.

In Terminal, copy the example settings, add your Unsloth API key to `.env`, and run the tool:

```sh
cp .env.example .env
# Edit UNSLOTH_API_KEY in .env
chmod 600 .env
python3 prompt_expander.py
```

Enter one image idea. The tool prints the expanded prompt, saves one PNG in the directory set by `OUTPUT_DIR`, and opens it in the app set by `PREVIEW_APP`. All local endpoints, model names, and prompt instructions come from `.env`. It uses only Python's standard library.

## Flask API

The same workflow is available through Flask. Install Flask in the project virtual environment if needed, then start the server:

```sh
.venv/bin/python -m pip install 'Flask>=3.0.0'
.venv/bin/flask --app api run --host 127.0.0.1 --port 8000
```

The routes are:

| Route | Purpose |
| --- | --- |
| `GET /health` | Check Ollama, its configured model, and the loaded Unsloth image model. |
| `POST /expand` with `{"idea":"moonlit train station"}` | Return `{"prompt":"..."}` from Ollama. |
| `POST /images` with `{"prompt":"..."}` | Generate one PNG with Unsloth and return `{"path":"generated/...png"}`. |

The API saves images without opening Preview. The CLI still opens them. For example:

```sh
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/expand -H 'Content-Type: application/json' -d '{"idea":"moonlit train station"}'
```

API references: [Ollama generate](https://docs.ollama.com/api/generate), [Unsloth API authentication](https://unsloth.ai/docs/basics/api), and your running Unsloth server's `/openapi.json` for its image generation request and response fields.
