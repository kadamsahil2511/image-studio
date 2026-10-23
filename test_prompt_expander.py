import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import prompt_expander


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
)


class PromptExpanderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.old_cwd = Path.cwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, self.old_cwd)
        self.requests = []
        Path(".env").write_text(
            "UNSLOTH_API_KEY=test-key\n"
            "UNSLOTH_API_URL=http://127.0.0.1:8888\n"
            "OLLAMA_API_URL=http://127.0.0.1:11434\n"
            "OLLAMA_MODEL=gemma4:e2b-mlx\n"
            "IMAGE_MODEL_FAMILY=FLUX.2\n"
            "PROMPT_INSTRUCTION=Expand into detailed pixelated anime art. Return only the prompt.\n"
            "OUTPUT_DIR=generated\n"
            "PREVIEW_APP=Preview\n"
        )

    def api(self, request, timeout):
        self.requests.append(request)
        url = request.full_url
        if url.endswith("/v1/models"):
            data = {"object": "list", "data": [{"id": "black-forest-labs/FLUX.2-dev", "loaded": True}]}
        elif url.endswith("/api/generate"):
            data = {"response": "pixelated anime art, moonlit train station, blue palette"}
        elif url.endswith("/v1/images/generations"):
            data = {"created": 1, "data": [{"b64_json": base64.b64encode(PNG).decode()}]}
        else:
            raise AssertionError(url)
        return io.BytesIO(json.dumps(data).encode())

    @patch("prompt_expander.subprocess.run")
    @patch("prompt_expander.input", return_value="moonlit train station")
    def test_generates_png_and_opens_preview(self, _input, preview):
        with patch("prompt_expander.urlopen", side_effect=self.api):
            prompt_expander.main()

        images = list(Path("generated").glob("*.png"))
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].read_bytes(), PNG)
        self.assertEqual([r.full_url.rsplit("/", 1)[-1] for r in self.requests],
                         ["models", "generate", "generations"])
        self.assertEqual(self.requests[0].get_header("Authorization"), "Bearer test-key")
        self.assertEqual(self.requests[2].get_header("Authorization"), "Bearer test-key")
        ollama = json.loads(self.requests[1].data)
        self.assertEqual(ollama["model"], "gemma4:e2b-mlx")
        self.assertEqual((ollama["stream"], ollama["keep_alive"]), (False, 0))
        self.assertIn("moonlit train station", ollama["prompt"])
        self.assertIn("pixel", ollama["prompt"].lower())
        image_request = json.loads(self.requests[2].data)
        self.assertEqual(image_request, {"model": "black-forest-labs/FLUX.2-dev", "prompt": "pixelated anime art, moonlit train station, blue palette", "n": 1, "response_format": "b64_json"})
        preview.assert_called_once_with(["open", "-a", "Preview", str(images[0])], check=True)

    @patch("prompt_expander.input", return_value="moonlit train station")
    def test_missing_key_stops_before_requests(self, _input):
        Path(".env").write_text(Path(".env").read_text().replace("UNSLOTH_API_KEY=test-key", "UNSLOTH_API_KEY="))
        with patch("prompt_expander.urlopen", side_effect=self.api):
            with self.assertRaisesRegex(RuntimeError, "UNSLOTH_API_KEY"):
                prompt_expander.main()
        self.assertEqual(self.requests, [])
        self.assertFalse(Path("generated").exists())

    @patch("prompt_expander.input", return_value="idea")
    def test_unloaded_or_other_model_stops_before_ollama(self, _input):
        for models in ([{"id": "FLUX.2", "loaded": False}], [{"id": "SDXL", "loaded": True}]):
            self.requests.clear()
            def only_models(request, timeout):
                self.requests.append(request)
                return io.BytesIO(json.dumps({"object": "list", "data": models}).encode())
            with patch("prompt_expander.urlopen", side_effect=only_models):
                with self.assertRaisesRegex(RuntimeError, "FLUX.2"):
                    prompt_expander.main()
            self.assertEqual(len(self.requests), 1)
            self.assertFalse(Path("generated").exists())

    @patch("prompt_expander.input", return_value="idea")
    def test_api_failure_leaves_no_image(self, _input):
        def failing_api(request, timeout):
            if request.full_url.endswith("/api/generate"):
                raise HTTPError(request.full_url, 500, "failed", {}, None)
            return self.api(request, timeout)
        with patch("prompt_expander.urlopen", side_effect=failing_api):
            with self.assertRaisesRegex(RuntimeError, "Ollama"):
                prompt_expander.main()
        self.assertFalse(Path("generated").exists())

    @patch("prompt_expander.input", return_value="idea")
    def test_invalid_png_leaves_no_image(self, _input):
        def invalid_api(request, timeout):
            if request.full_url.endswith("/v1/images/generations"):
                return io.BytesIO(json.dumps({"created": 1, "data": [{"b64_json": base64.b64encode(b"not a PNG").decode()}]}).encode())
            return self.api(request, timeout)
        with patch("prompt_expander.urlopen", side_effect=invalid_api):
            with self.assertRaisesRegex(RuntimeError, "PNG"):
                prompt_expander.main()
        self.assertFalse(Path("generated").exists())

    @patch("prompt_expander.input", return_value="idea")
    def test_image_api_failure_leaves_no_image(self, _input):
        def failing_api(request, timeout):
            if request.full_url.endswith("/v1/images/generations"):
                raise HTTPError(request.full_url, 503, "unavailable", {}, None)
            return self.api(request, timeout)
        with patch("prompt_expander.urlopen", side_effect=failing_api):
            with self.assertRaisesRegex(RuntimeError, "Unsloth"):
                prompt_expander.main()
        self.assertFalse(Path("generated").exists())

    @patch("prompt_expander.subprocess.run", side_effect=OSError("Preview unavailable"))
    @patch("prompt_expander.input", return_value="idea")
    def test_preview_failure_keeps_image(self, _input, _preview):
        with patch("prompt_expander.urlopen", side_effect=self.api):
            prompt_expander.main()
        self.assertEqual(len(list(Path("generated").glob("*.png"))), 1)

    @patch("prompt_expander.subprocess.run")
    @patch("prompt_expander.input", return_value="idea")
    def test_uses_values_from_env_file(self, _input, preview):
        config = Path(".env").read_text().replace("gemma4:e2b-mlx", "custom-ollama")
        config = config.replace("127.0.0.1:8888", "127.0.0.1:9999")
        config = config.replace("127.0.0.1:11434", "127.0.0.1:9998")
        config = config.replace("generated", "pictures")
        config = config.replace("Preview", "OtherViewer")
        config = config.replace("IMAGE_MODEL_FAMILY=FLUX.2", "IMAGE_MODEL_FAMILY=OtherImage")
        config = config.replace("Expand into detailed pixelated anime art.", "Describe the scene as a watercolor.")
        Path(".env").write_text(config)
        def custom_api(request, timeout):
            if request.full_url.endswith("/v1/models"):
                self.requests.append(request)
                return io.BytesIO(json.dumps({"object": "list", "data": [{"id": "OtherImage-v1", "loaded": True}]}).encode())
            return self.api(request, timeout)
        with patch("prompt_expander.urlopen", side_effect=custom_api):
            prompt_expander.main()
        self.assertTrue(all(r.full_url.startswith(("http://127.0.0.1:9999", "http://127.0.0.1:9998")) for r in self.requests))
        self.assertEqual(json.loads(self.requests[1].data)["model"], "custom-ollama")
        self.assertIn("watercolor", json.loads(self.requests[1].data)["prompt"])
        self.assertEqual(json.loads(self.requests[2].data)["model"], "OtherImage-v1")
        image = next(Path("pictures").glob("*.png"))
        preview.assert_called_once_with(["open", "-a", "OtherViewer", str(image)], check=True)


if __name__ == "__main__":
    unittest.main()
