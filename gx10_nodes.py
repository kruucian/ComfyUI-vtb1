from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from typing import Dict, List, Sequence, Tuple

from PIL import Image
import numpy as np
import torch


def _encode_image_to_png_b64(image: torch.Tensor) -> str:
    array = (image.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    with io.BytesIO() as buf:
        Image.fromarray(array).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _normalize_metadata(value: object) -> Dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _coerce_images(images: Sequence[object]) -> List[torch.Tensor]:
    tensor_images: List[torch.Tensor] = []
    for image in images:
        if isinstance(image, torch.Tensor):
            tensor_images.append(image)
        else:
            tensor_images.append(torch.as_tensor(image))
    return tensor_images


class GX10TextInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 2_147_483_647}),
                "width": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "run_id": ("STRING", {"default": "", "multiline": False}),
                "metadata_json": ("STRING", {"multiline": True, "default": "{}"}),
                "callback_url": ("STRING", {"default": "", "multiline": False}),
                "mode": ("STRING", {"default": "t2i", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "prompt",
        "negative_prompt",
        "seed",
        "width",
        "height",
        "run_id",
        "metadata_json",
        "callback_url",
        "mode",
    )
    FUNCTION = "pack"
    OUTPUT_NODE = False
    CATEGORY = "GX10"

    def pack(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        run_id: str,
        metadata_json: str,
        callback_url: str,
        mode: str,
    ) -> Tuple[str, str, int, int, int, str, str, str, str]:
        return (
            str(prompt),
            str(negative_prompt),
            int(seed),
            int(width),
            int(height),
            str(run_id),
            str(metadata_json),
            str(callback_url),
            str(mode),
        )


class GX10ImageUpload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "run_id": ("STRING", {"default": "", "multiline": False}),
                "callback_url": ("STRING", {"default": "", "multiline": False}),
                "metadata_json": ("STRING", {"multiline": True, "default": "{}"}),
                "mode": ("STRING", {"default": "t2i", "multiline": False}),
                "auth_header": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "upload"
    OUTPUT_NODE = True
    CATEGORY = "GX10"

    def upload(
        self,
        images: torch.Tensor,
        run_id: str,
        callback_url: str,
        metadata_json: str,
        mode: str = "t2i",
        auth_header: str = "",
    ) -> Tuple[str]:
        run_id = str(run_id).strip()
        callback_url = str(callback_url).strip()
        metadata = _normalize_metadata(metadata_json)
        metadata.setdefault("mode", str(mode))

        if not callback_url:
            status = "skip"
            message = "callback_url is empty"
            return (json.dumps({"status": status, "message": message, "run_id": run_id}),)

        image_list = _coerce_images(images)
        if not image_list:
            return (json.dumps({"status": "skip", "message": "no images", "run_id": run_id}),)

        encoded: List[Dict[str, object]] = []
        for index, image in enumerate(image_list):
            encoded.append(
                {
                    "image_b64": _encode_image_to_png_b64(image),
                    "format": "png",
                    "index": index,
                    "metadata": metadata,
                    "signature": None,
                }
            )

        if len(encoded) == 1:
            payload = dict(
                run_id=run_id,
                format=encoded[0]["format"],
                index=encoded[0]["index"],
                image_b64=encoded[0]["image_b64"],
                metadata=encoded[0]["metadata"],
                signature=encoded[0]["signature"],
            )
        else:
            payload = {
                "run_id": run_id,
                "images": [
                    {
                        "run_id": run_id,
                        "format": item["format"],
                        "index": item["index"],
                        "image_b64": item["image_b64"],
                        "metadata": item["metadata"],
                        "signature": item["signature"],
                    }
                    for item in encoded
                ],
                "metadata": metadata,
            }

        req = urllib.request.Request(
            callback_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if auth_header:
            req.add_header("Authorization", auth_header)

        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                body = response.read()
                text = body.decode("utf-8", errors="ignore") if body else ""
            status_code = getattr(response, "status", None)
            detail = {"status": "ok", "status_code": status_code, "run_id": run_id, "response": text[:200]}
            print(f"[GX10ImageUpload] callback success: run_id={run_id} status_code={status_code}")
            return (json.dumps(detail),)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            print(f"[GX10ImageUpload] callback failed (HTTP {exc.code}): run_id={run_id} body={body}")
            return (
                json.dumps(
                    {
                        "status": "error",
                        "status_code": getattr(exc, "code", None),
                        "run_id": run_id,
                        "error": body[:200],
                    }
                ),
            )
        except Exception as exc:
            print(f"[GX10ImageUpload] callback exception: run_id={run_id} error={exc}")
            return (
                json.dumps(
                    {
                        "status": "error",
                        "run_id": run_id,
                        "error": str(exc),
                    }
                ),
            )


NODE_CLASS_MAPPINGS = {
    "GX10TextInput": GX10TextInput,
    "GX10ImageUpload": GX10ImageUpload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GX10TextInput": "GX10 Text Input",
    "GX10ImageUpload": "GX10 Image Upload (Callback)",
}
