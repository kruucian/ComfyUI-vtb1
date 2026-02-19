from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

from PIL import Image
import numpy as np
import torch


def _encode_image_to_png_b64(image: torch.Tensor) -> str:
    array = (image.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    with io.BytesIO() as buf:
        Image.fromarray(array).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _coerce_text_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]

    lines = [item.strip() for item in text.splitlines() if item.strip()]
    if lines:
        return lines

    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


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


def _normalize_text_inputs(prompt: str, text: str, texts: str) -> Tuple[str, str, str]:
    primary = str(prompt or text or "").strip()
    extra = _coerce_text_list(texts)
    merged_items: List[str] = []
    if primary:
        merged_items.append(primary)
    for item in extra:
        if item not in merged_items:
            merged_items.append(item)

    if not merged_items:
        return "", "", "[]"
    if len(merged_items) == 1:
        merged_prompt = merged_items[0]
        merged_json = json.dumps([merged_items[0]], ensure_ascii=False)
    else:
        merged_prompt = "\n".join(merged_items)
        merged_json = json.dumps(merged_items, ensure_ascii=False)
    return merged_prompt, primary, merged_json


def _to_tensor_from_file(path: str) -> torch.Tensor | None:
    path = str(path or "").strip()
    if not path:
        return None
    if path.startswith("file://"):
        path = path[len("file://") :]

    try:
        if path.startswith("http://") or path.startswith("https://"):
            with urllib.request.urlopen(path, timeout=10) as response:
                data = response.read()
            image = Image.open(io.BytesIO(data))
        else:
            image = Image.open(path)
    except Exception:
        return None

    image = image.convert("RGB")
    array = np.array(image).astype(np.float32) / 255.0
    if array.ndim != 3:
        return None
    return torch.from_numpy(array)[None, ...]


class GX10TextInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "texts": ("STRING", {"multiline": True, "default": ""}),
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

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "INT",
        "INT",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "prompt",
        "text",
        "texts",
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
        text: str,
        texts: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        run_id: str,
        metadata_json: str,
        callback_url: str,
        mode: str,
    ) -> Tuple[str, str, str, str, int, int, int, str, str, str, str]:
        merged_prompt, merged_text, merged_texts = _normalize_text_inputs(prompt, text, texts)
        return (
            merged_prompt,
            merged_text,
            merged_texts,
            str(negative_prompt),
            int(seed),
            int(width),
            int(height),
            str(run_id),
            str(metadata_json),
            str(callback_url),
            str(mode),
        )


class GX10ImageInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "texts": ("STRING", {"multiline": True, "default": ""}),
                "first_frame_image": ("STRING", {"default": "", "multiline": False}),
                "image_path": ("STRING", {"default": "", "multiline": False}),
                "run_id": ("STRING", {"default": "", "multiline": False}),
                "metadata_json": ("STRING", {"multiline": True, "default": "{}"}),
                "mode": ("STRING", {"default": "i2v", "multiline": False}),
                "callback_url": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("prompt", "text", "texts", "image")
    FUNCTION = "pack"
    OUTPUT_NODE = False
    CATEGORY = "GX10"

    def pack(
        self,
        prompt: str,
        text: str,
        texts: str,
        first_frame_image: str,
        image: Optional[torch.Tensor],
        image_path: str,
        run_id: str,
        metadata_json: str,
        mode: str,
        callback_url: str,
    ) -> Tuple[str, str, str, torch.Tensor]:
        merged_prompt, merged_text, merged_texts = _normalize_text_inputs(prompt, text, texts)
        del run_id
        del metadata_json
        del mode
        del callback_url

        image_candidates = [image_path, first_frame_image]
        output_image: torch.Tensor | None = None
        for path in image_candidates:
            if not path:
                continue
            output_image = _to_tensor_from_file(path)
            if output_image is not None:
                break

        if output_image is None:
            if image is not None:
                output_image = _ensure_image_batch(image)
        if output_image is None:
            output_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        return (
            merged_prompt,
            merged_text,
            merged_texts,
            output_image,
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


class GX10SaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "run_id": ("STRING", {"default": "", "multiline": False}),
                "callback_url": ("STRING", {"default": "", "multiline": False}),
                "metadata_json": ("STRING", {"multiline": True, "default": "{}"}),
                "mode": ("STRING", {"default": "t2i", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "gx10", "multiline": False}),
                "image_format": ("STRING", {"default": "png", "multiline": False}),
                "auth_header": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "GX10"

    def save_images(
        self,
        images: torch.Tensor,
        run_id: str,
        callback_url: str,
        metadata_json: str,
        mode: str = "t2i",
        filename_prefix: str = "gx10",
        image_format: str = "png",
        auth_header: str = "",
    ) -> Tuple[str]:
        run_id = str(run_id).strip()
        callback_url = str(callback_url).strip()
        metadata = _normalize_metadata(metadata_json)
        metadata.setdefault("mode", str(mode))
        metadata.setdefault("filename_prefix", str(filename_prefix))
        metadata.setdefault("format", str(image_format))

        if not callback_url:
            return (json.dumps({"status": "skip", "message": "callback_url is empty", "run_id": run_id}),)

        image_list = _coerce_images(images)
        if not image_list:
            return (json.dumps({"status": "skip", "message": "no images", "run_id": run_id}),)

        images_payload = [
            {
                "run_id": run_id,
                "format": str(image_format),
                "index": index,
                "image_b64": _encode_image_to_png_b64(image),
                "metadata": metadata,
                "signature": None,
            }
            for index, image in enumerate(image_list)
        ]

        if len(images_payload) == 1:
            first = images_payload[0]
            payload = {
                "run_id": first["run_id"],
                "format": first["format"],
                "index": first["index"],
                "image_b64": first["image_b64"],
                "metadata": first["metadata"],
                "signature": first["signature"],
            }
        else:
            payload = {
                "run_id": run_id,
                "images": images_payload,
                "metadata": metadata,
                "signature": None,
            }

        detail = _post_callback(callback_url, payload, auth_header=auth_header)
        if isinstance(detail, dict):
            detail.setdefault("run_id", run_id)
            detail.setdefault("status", "error")
        return (json.dumps(detail),)


class GX10SaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("STRING", {"default": "", "multiline": False}),
                "run_id": ("STRING", {"default": "", "multiline": False}),
                "callback_url": ("STRING", {"default": "", "multiline": False}),
                "metadata_json": ("STRING", {"multiline": True, "default": "{}"}),
                "status": ("STRING", {"default": "succeeded", "multiline": False}),
                "mode": ("STRING", {"default": "i2v", "multiline": False}),
                "auth_header": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "GX10"

    def save_video(
        self,
        video: str,
        run_id: str,
        callback_url: str,
        metadata_json: str,
        status: str = "succeeded",
        mode: str = "i2v",
        auth_header: str = "",
    ) -> Tuple[str]:
        run_id = str(run_id).strip()
        callback_url = str(callback_url).strip()
        result_url = _coerce_video_path(video)
        if not result_url:
            return (json.dumps({"status": "skip", "message": "video is empty", "run_id": run_id}),)
        if not callback_url:
            return (json.dumps({"status": "skip", "message": "callback_url is empty", "run_id": run_id}),)

        payload = {
            "run_id": run_id,
            "status": str(status).strip() or "succeeded",
            "result_url": _coerce_video_path(video),
            "metadata": _normalize_metadata(metadata_json),
            "signature": None,
        }
        payload["metadata"].setdefault("mode", str(mode))

        detail = _post_callback(callback_url, payload, auth_header=auth_header)
        if isinstance(detail, dict):
            detail.setdefault("run_id", run_id)
            detail.setdefault("status", "error")
        return (json.dumps(detail),)


def _ensure_image_batch(images: torch.Tensor) -> torch.Tensor:
    if isinstance(images, torch.Tensor):
        tensor = images
    else:
        tensor = torch.as_tensor(images)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim == 1 and tensor.shape and tensor.shape[0] == 0:
        return torch.empty((1, 0, 0, 3), dtype=torch.float32)
    return tensor


def _coerce_video_path(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Path):
        return str(value).strip()
    for attr in ("path", "filename", "file", "uri"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    if isinstance(value, dict):
        for key in ("result_url", "video_path", "file", "filename", "path", "uri"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    if isinstance(value, (list, tuple)):
        for item in value:
            path = _coerce_video_path(item)
            if path:
                return path
    return ""


def _post_callback(url: str, payload: Dict, auth_header: str = "", timeout: float = 20.0) -> Dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if auth_header:
        request.add_header("Authorization", auth_header)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
        return {
            "status": "ok",
            "status_code": getattr(response, "status", None),
            "response": body.decode("utf-8", errors="ignore")[:200] if body else "",
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        return {
            "status": "error",
            "status_code": getattr(exc, "code", None),
            "error": body[:200],
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


NODE_CLASS_MAPPINGS = {
    "GX10TextInput": GX10TextInput,
    "GX10ImageInput": GX10ImageInput,
    "GX10ImageUpload": GX10ImageUpload,
    "GX10SaveImage": GX10SaveImage,
    "GX10SaveVideo": GX10SaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GX10TextInput": "GX10 Text Input",
    "GX10ImageInput": "GX10 Image Input",
    "GX10ImageUpload": "GX10 Image Upload (Callback)",
    "GX10SaveImage": "GX10 Save Image (Callback)",
    "GX10SaveVideo": "GX10 Save Video (Callback)",
}
