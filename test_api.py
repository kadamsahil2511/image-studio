import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import api
from test_prompt_expander import PNG


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.old_cwd = Path.cwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, self.old_cwd)
        Path(".env").write_text(
            "UNSLOTH_API_KEY=test-key\nUNSLOTH_API_URL=http://127.0.0.1:8888\n"
            "OLLAMA_API_URL=http://127.0.0.1:11434\nOLLAMA_MODEL=gemma4:e2b-mlx\n"
            "IMAGE_MODEL_FAMILY=FLUX.2\nPROMPT_INSTRUCTION=Describe pixelated anime art.\n"
            "OUTPUT_DIR=generated\nPREVIEW_APP=Preview\n"
        )
        self.requests = []
        self.client = api.app.test_client()

    def fake_api(self, request, timeout):
        self.requests.append(request)
        if request.full_url.endswith("/api/tags"):
            data = {"models": [{"name": "gemma4:e2b-mlx"}]}
        elif request.full_url.endswith("/v1/models"):
            data = {"object": "list", "data": [{"id": "unsloth/FLUX.2-klein-4B-GGUF", "loaded": True}]}
        elif request.full_url.endswith("/api/generate"):
            data = {"response": "pixelated anime train station"}
        elif request.full_url.endswith("/v1/images/generations"):
            data = {"created": 1, "data": [{"b64_json": base64.b64encode(PNG).decode()}]}
        else:
            raise AssertionError(request.full_url)
        return io.BytesIO(json.dumps(data).encode())

    def test_health_checks_both_services_and_loaded_model(self):
        with patch("prompt_expander.urlopen", side_effect=self.fake_api):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ollama": "ok", "unsloth": "ok", "image_model": "unsloth/FLUX.2-klein-4B-GGUF"})
        self.assertEqual(len(self.requests), 2)

    def test_expand_only_calls_ollama(self):
        with patch("prompt_expander.urlopen", side_effect=self.fake_api):
            response = self.client.post("/expand", json={"idea": "train station"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"prompt": "pixelated anime train station"})
        self.assertEqual([r.full_url.rsplit("/", 1)[-1] for r in self.requests], ["generate"])
        self.assertEqual(json.loads(self.requests[0].data)["keep_alive"], 0)

    def test_images_saves_png_without_opening_preview(self):
        with patch("prompt_expander.urlopen", side_effect=self.fake_api), patch("prompt_expander.subprocess.run") as preview:
            response = self.client.post("/images", json={"prompt": "pixelated anime train station"})
        self.assertEqual(response.status_code, 200)
        path = Path(response.get_json()["path"])
        self.assertEqual(path.read_bytes(), PNG)
        self.assertEqual([r.full_url.rsplit("/", 1)[-1] for r in self.requests], ["models", "generations"])
        preview.assert_not_called()

    def test_unavailable_model_reports_503_without_file(self):
        def no_image_model(request, timeout):
            if request.full_url.endswith("/v1/models"):
                return io.BytesIO(json.dumps({"object": "list", "data": []}).encode())
            return self.fake_api(request, timeout)
        with patch("prompt_expander.urlopen", side_effect=no_image_model):
            health = self.client.get("/health")
            image = self.client.post("/images", json={"prompt": "train"})
        self.assertEqual(health.status_code, 503)
        self.assertEqual(image.status_code, 503)
        self.assertFalse(Path("generated").exists())

    def test_malformed_ollama_health_response_reports_503(self):
        def malformed_tags(request, timeout):
            if request.full_url.endswith("/api/tags"):
                return io.BytesIO(b'{"models":null}')
            return self.fake_api(request, timeout)
        with patch("prompt_expander.urlopen", side_effect=malformed_tags):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)

    def test_empty_idea_is_rejected(self):
        with patch("prompt_expander.urlopen", side_effect=self.fake_api):
            response = self.client.post("/expand", json={"idea": "   "})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
