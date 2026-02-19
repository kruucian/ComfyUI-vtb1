from __future__ import annotations

import base64
import io
import json
import os
import shutil
from urllib.parse import urljoin, urlparse
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import folder_paths
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


def _normalize_text_inputs(prompt: str = "", text: str = "", texts: str = "") -> Tuple[str, str, str]:
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
                "texts": ("STRING", {"multiline": True, "default": ""}),
                "run_id": ("STRING", {"default": "", "multiline": False}),
                "callback_url": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "prompt",
        "text",
        "texts",
        "run_id",
        "callback_url",
    )
    FUNCTION = "pack"
    OUTPUT_NODE = False
    CATEGORY = "GX10"

    def pack(
        self,
        texts: str,
        run_id: str,
        callback_url: str,
    ) -> Tuple[str, str, str, str, str]:
        merged_prompt, merged_text, merged_texts = _normalize_text_inputs(texts=texts)
        return (
            merged_prompt,
            merged_text,
            merged_texts,
            str(run_id),
            str(callback_url),
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
                "auth_header": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "GX10"

    def save_images(
        self,
        images: torch.Tensor,
        run_id: str,
        callback_url: str,
        auth_header: str = "",
    ) -> Dict[str, object]:
        run_id = str(run_id).strip()
        callback_url = str(callback_url).strip()
        metadata = {}

        if not callback_url:
            return {"ui": {"images": []}}

        image_list = _coerce_images(images)
        if not image_list:
            return {"ui": {"images": []}}

        images_payload = [
            {
                "run_id": run_id,
                "format": "png",
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
        ui_images = _preview_image_items(image_list, filename_prefix="gx10")
        return {
            "ui": {"images": ui_images},
        }


class GX10SaveVideo:
    _VIDEO_PREVIEW_HISTORY: List[Dict[str, str]] = []

    @classmethod
    def _normalize_video_history(cls, new_items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if new_items:
            cls._VIDEO_PREVIEW_HISTORY.extend(new_items)
        if len(cls._VIDEO_PREVIEW_HISTORY) > 6:
            cls._VIDEO_PREVIEW_HISTORY = cls._VIDEO_PREVIEW_HISTORY[-6:]
        return list(cls._VIDEO_PREVIEW_HISTORY)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("STRING", {"default": "", "multiline": False}),
                "run_id": ("STRING", {"default": "", "multiline": False}),
                "callback_url": ("STRING", {"default": "", "multiline": False}),
                "auth_header": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "hls_url": ("STRING", {"default": "", "multiline": False}),
                "hls_base_url": ("STRING", {"default": "", "multiline": False}),
                "prefer_hls": ("BOOLEAN", {"default": True}),
                "hls_only": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "GX10"

    def save_video(
        self,
        video: str,
        run_id: str,
        callback_url: str,
        auth_header: str = "",
        hls_url: str = "",
        hls_base_url: str = "",
        prefer_hls: bool = True,
        hls_only: bool = True,
    ) -> Dict[str, object]:
        run_id = str(run_id).strip()
        callback_url = str(callback_url).strip()
        video = str(video)
        source_url = _coerce_video_path(video)
        if not source_url:
            return {"ui": {"video": []}}

        result_url = _resolve_hls_media(
            video=video,
            prefer_hls=bool(prefer_hls),
            hls_only=bool(hls_only),
            hls_url=hls_url,
            hls_base_url=hls_base_url,
        )
        if not result_url:
            return {"ui": {"video": []}}

        payload = {
            "run_id": run_id,
            "status": "succeeded",
            "result_url": source_url,
            "hls_url": result_url if _is_hls_url(result_url) else None,
            "metadata": {},
            "signature": None,
        }
        payload["metadata"]["streaming_protocol"] = "hls" if _is_hls_url(result_url) else "file"
        payload["metadata"]["streaming_source"] = "hls" if _is_hls_url(result_url) else "file"
        if _is_hls_url(result_url):
            payload["metadata"]["stream_hls_url"] = result_url

        if callback_url:
            detail = _post_callback(callback_url, payload, auth_header=auth_header)
            if isinstance(detail, dict):
                detail.setdefault("run_id", run_id)
                detail.setdefault("status", "error")

        ui_videos = self._normalize_video_history(
            _preview_video_items(
                result_url=str(source_url),
                filename_prefix="gx10",
            )
        )
        return {"ui": {"video": ui_videos}}


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
        return value.strip().replace("file://", "", 1) if value.strip().startswith("file://") else value.strip()
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


def _sanitize_uimeta_path(path: str) -> str:
    if not path:
        return ""
    path = path.replace("file://", "", 1) if path.startswith("file://") else path
    return path


def _is_remote_url(value: str) -> bool:
    if not value:
        return False
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return bool(parsed.scheme and parsed.netloc)


def _is_hls_url(value: str) -> bool:
    value = value.lower()
    if not value:
        return False
    if value.startswith("file://"):
        value = value[len("file://") :]
    if "?" in value:
        value = value.split("?", 1)[0]
    if value.startswith("http://") or value.startswith("https://"):
        return value.split("?")[0].endswith(".m3u8")
    return value.endswith((".m3u8", ".ts"))


def _to_public_url(local_path: str, base_url: str) -> str:
    if not local_path or not base_url:
        return local_path
    if _is_remote_url(local_path):
        return local_path
    try:
        normalized_base = base_url.rstrip("/")
        source = Path(local_path)
        if source.exists():
            return urljoin(f"{normalized_base}/", source.as_posix().lstrip("/"))
        return local_path
    except Exception:
        return local_path


def _collect_hls_candidates(video_path: Path) -> List[Path]:
    candidates: List[Path] = []
    base = video_path
    if base.suffix.lower() == ".m3u8":
        return [base]

    name_candidates = [
        base.with_name(base.name).with_suffix(".m3u8"),
        base.with_name("index.m3u8"),
        base.with_name("playlist.m3u8"),
        base.with_name("manifest.m3u8"),
        base.with_name("stream.m3u8"),
    ]
    candidates.extend([p for p in name_candidates if p != base])

    same_stem = sorted(
        [
            child
            for child in base.parent.glob(f"{base.stem}*.m3u8")
            if child.is_file() and child.suffix.lower() == ".m3u8"
        ]
    )
    candidates.extend(same_stem)

    if base.parent.exists():
        all_hls = sorted([p for p in base.parent.glob("*.m3u8") if p.is_file()])
        candidates.extend(all_hls)
    return list(dict.fromkeys(candidates))


def _resolve_hls_media(
    video: str,
    prefer_hls: bool = True,
    hls_only: bool = True,
    hls_url: str = "",
    hls_base_url: str = "",
) -> str:
    source = _coerce_video_path(video)
    if not source:
        return ""

    explicit_hls = _coerce_video_path(hls_url) or (hls_url or "").strip()
    if explicit_hls:
        resolved_hls = _coerce_video_path(explicit_hls)
        if resolved_hls and _is_hls_url(resolved_hls):
            return _to_public_url(resolved_hls, hls_base_url)
        if not hls_only:
            return _to_public_url(source, hls_base_url)

    if not prefer_hls:
        return _to_public_url(_coerce_video_path(video), hls_base_url)

    if _is_hls_url(source):
        return _to_public_url(_coerce_video_path(source), hls_base_url)

    path = Path(_coerce_video_path(source))
    if path.exists():
        candidates = _collect_hls_candidates(path)
        for candidate in candidates:
            if candidate.exists():
                return _to_public_url(str(candidate), hls_base_url)

    if source.startswith("http") and not source.lower().endswith(".m3u8"):
        parsed = source.rsplit(".", 1)
        if len(parsed) == 2:
            alt = f"{parsed[0]}.m3u8"
            if alt:
                return alt

    if hls_only:
        return ""
    return _to_public_url(_coerce_video_path(video), hls_base_url)


def _ensure_folder_compatible_preview_file(
    source_path: str,
    subdir: str = "gx10_previews",
) -> tuple[str, str]:
    source_path = _sanitize_uimeta_path(source_path)
    source = Path(source_path)
    output_dir = Path(folder_paths.get_output_directory())
    if not source.exists():
        return "", ""

    try:
        rel = source.relative_to(output_dir)
        return str(rel).replace("\\", "/"), "output"
    except ValueError:
        # Copy heavy media only for ui display fallback.
        preview_dir = output_dir / subdir
        preview_dir.mkdir(parents=True, exist_ok=True)
        dest = preview_dir / source.name
        try:
            if dest.exists():
                return (f"{subdir}/{source.name}", "output")
            shutil.copy2(source, dest)
            return f"{subdir}/{source.name}", "output"
        except Exception:
            return "", ""


def _preview_image_items(image_list: Sequence[torch.Tensor], filename_prefix: str = "gx10") -> List[Dict[str, str]]:
    if not isinstance(image_list, (list, tuple)):
        return []
    image_list = list(image_list)
    if not image_list:
        return []
    try:
        first_image = image_list[0]
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            int(first_image.shape[1]),
            int(first_image.shape[0]),
        )
    except Exception:
        return []

    results: List[Dict[str, str]] = []
    for batch_number, image in enumerate(image_list):
        array = (image.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if array.ndim != 3:
            continue
        img = Image.fromarray(array)
        file_name = filename.replace("%batch_num%", str(batch_number))
        file_name = f"{file_name}_{counter:05}_.png"
        try:
            img.save(os.path.join(full_output_folder, file_name))
        except Exception:
            continue
        results.append({"filename": file_name, "subfolder": subfolder, "type": "output"})
        counter += 1
    return results


def _preview_video_items(result_url: str, filename_prefix: str = "gx10") -> List[Dict[str, str]]:
    result_url = _sanitize_uimeta_path(result_url)
    if not result_url:
        return []
    path = Path(result_url)
    if path.suffix.lower() not in {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".flv", ".m3u8"}:
        return []
    rel_name, dir_type = _ensure_folder_compatible_preview_file(result_url, subdir=f"{filename_prefix}")
    if not rel_name:
        return []
    parts = rel_name.split("/")
    if len(parts) == 1:
        subfolder = ""
        filename = parts[0]
    else:
        subfolder = "/".join(parts[:-1])
        filename = parts[-1]
    return [
        {
            "filename": filename,
            "subfolder": subfolder,
            "type": dir_type or "output",
            "format": "video/x-mpegurl" if path.suffix.lower() == ".m3u8" else "video/mp4",
        }
    ]


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
