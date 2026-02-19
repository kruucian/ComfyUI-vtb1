from __future__ import annotations

import base64
import wave
import io
import json
import os
import shutil
import subprocess
import tempfile
from urllib.parse import urljoin, urlparse
import urllib.error
import urllib.request
from typing import Dict, List, Sequence, Tuple
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


def _split_prompt_triplet(texts: str = "") -> Tuple[str, str, str]:
    items = _coerce_text_list(texts)
    if not items:
        return "", "", ""
    tts_prompt = items[0]
    positive_prompt = items[1] if len(items) > 1 else tts_prompt
    negative_prompt = items[2] if len(items) > 2 else ""
    return tts_prompt, positive_prompt, negative_prompt


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


def _coerce_video_frames(frames: object) -> List[torch.Tensor]:
    if frames is None:
        return []
    if isinstance(frames, torch.Tensor):
        if frames.ndim == 4:
            return [torch.clamp(frame, 0.0, 1.0) for frame in frames]
        if frames.ndim == 3:
            return [torch.clamp(frames, 0.0, 1.0)]
        return []
    if isinstance(frames, (list, tuple)):
        return [torch.clamp(torch.as_tensor(frame), 0.0, 1.0) for frame in frames if frame is not None]
    return []


def _normalize_frame_shape(frame: torch.Tensor) -> torch.Tensor:
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim == 3 and frame.shape[0] in {1, 3, 4} and frame.shape[-1] not in {1, 3, 4}:
        frame = frame.permute(1, 2, 0)
    if frame.ndim == 2:
        return torch.stack([frame, frame, frame], dim=-1)
    if frame.ndim == 3 and frame.shape[-1] == 4:
        return frame[:, :, :3]
    if frame.ndim == 3 and frame.shape[-1] == 1:
        return torch.cat([frame, frame, frame], dim=-1)
    return frame


def _apply_pingpong(frames: Sequence[torch.Tensor], pingpong: bool) -> List[torch.Tensor]:
    if not pingpong:
        return list(frames)
    if len(frames) <= 2:
        return list(frames)
    return list(frames) + list(reversed(frames[1:-1]))


def _build_video_from_frames(
    frames: Sequence[torch.Tensor],
    frame_rate: float = 8.0,
    filename_prefix: str = "gx10_video",
    audio: str = "",
    save_output: bool = True,
    loop_count: int = 0,
    pingpong: bool = False,
    format_value: str = "video/mp4",
) -> str:
    if not frames:
        return ""
    first_frame = _normalize_frame_shape(frames[0]).cpu()
    width = int(first_frame.shape[1]) if first_frame.ndim >= 2 else 512
    height = int(first_frame.shape[0]) if first_frame.ndim >= 2 else 512
    if width <= 0 or height <= 0:
        return ""

    try:
        output_dir = Path(
            folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_str = str(output_dir)
        output_dir = Path(output_dir_str)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        prepared_frames = list(_apply_pingpong(frames, pingpong))
        if not prepared_frames:
            return ""

        output_dir.mkdir(parents=True, exist_ok=True)
        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix,
            output_dir_str,
            width,
            height,
        )
    except Exception:
        return ""

    format_value = str(format_value or "video/mp4").strip().lower()
    ext_map = {
        "video/mp4": "mp4",
        "video/webm": "webm",
        "video/mov": "mov",
        "video/x-m4v": "m4v",
        "video/avi": "avi",
        "video/x-flv": "flv",
        "video/x-matroska": "mkv",
        "video/mkv": "mkv",
        "video/x-mpegurl": "m3u8",
        "image/gif": "gif",
        "image/webp": "webp",
    }
    if format_value.startswith("video/") and format_value not in ext_map:
        ext = format_value.split("/", 1)[1]
    elif format_value.startswith("."):
        ext = format_value.lstrip(".")
    else:
        ext = ext_map.get(format_value, "mp4")
    if "." in ext:
        ext = ext.lstrip(".")

    output_file = os.path.join(full_output_folder, f"{filename}_{counter:05}.{ext}")
    frames_dir = Path(tempfile.mkdtemp(prefix="gx10_save_video_frames_"))
    frame_paths = []
    try:
        for idx, frame in enumerate(prepared_frames):
            normalized = _normalize_frame_shape(frame)
            normalized = torch.clamp(normalized, 0.0, 1.0)
            array = (normalized.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            if array.ndim != 3 or array.shape[2] != 3:
                return ""
            image = Image.fromarray(array)
            path = frames_dir / f"frame_{idx:08d}.png"
            image.save(path)
            frame_paths.append(str(path))

        if ext == "gif":
            duration_ms = int(1000 / max(1.0, float(frame_rate)))
            first_img = Image.open(frame_paths[0])
            frames = [Image.open(p) for p in frame_paths[1:]]
            first_img.save(
                output_file,
                format="GIF",
                save_all=True,
                append_images=frames,
                duration=duration_ms,
                loop=loop_count,
            )
            for f in frames:
                try:
                    f.close()
                except Exception:
                    pass
            return output_file

        if ext == "webp":
            duration_ms = int(1000 / max(1.0, float(frame_rate)))
            first_img = Image.open(frame_paths[0])
            frames = [Image.open(p) for p in frame_paths[1:]]
            first_img.save(
                output_file,
                format="WEBP",
                save_all=True,
                append_images=frames,
                duration=duration_ms,
                loop=loop_count,
                lossless=True,
            )
            for f in frames:
                try:
                    f.close()
                except Exception:
                    pass
            return output_file

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return ""

        frame_pattern = str(frames_dir / "frame_%08d.png")
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-r",
            str(max(1.0, float(frame_rate))),
            "-i",
            frame_pattern,
        ]
        loop_count_int = int(loop_count or 0)
        if loop_count_int > 0:
            command.extend(
                [
                    "-vf",
                    f"loop=loop={loop_count_int}:size={len(prepared_frames)}:start=0",
                ]
            )

        audio_path = _coerce_video_path(audio)
        command.extend(["-map", "0:v:0"])
        if audio_path and Path(audio_path).exists():
            command.extend(["-i", audio_path, "-map", "1:a?"])

        if audio_path and Path(audio_path).exists():
            command.extend(["-c:a", "aac", "-shortest"])
        else:
            command.extend(["-an"])

        if ext == "m3u8":
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-f",
                    "hls",
                    "-hls_list_size",
                    "0",
                    "-hls_flags",
                    "append_list",
                    "-hls_segment_filename",
                    str(Path(output_file).with_suffix(".%03d.ts")),
                ]
            )
        else:
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                ]
            )
        command.append(output_file)

        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        if not Path(output_file).exists():
            return ""
        return output_file
    finally:
        for path in frame_paths:
            try:
                Path(path).unlink()
            except Exception:
                pass
        try:
            shutil.rmtree(frames_dir, ignore_errors=True)
        except Exception:
            pass


class GX10TextInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texts": ("STRING", {"multiline": True, "default": ""}),
                "auth_header": ("STRING", {"default": "", "multiline": False}),
                "run_id": ("STRING", {"default": "", "multiline": False}),
                "callback_url": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("texts", "auth_header", "run_id", "callback_url")
    FUNCTION = "pack"
    OUTPUT_NODE = False
    CATEGORY = "GX10"

    def pack(
        self,
        texts: str,
        auth_header: str,
        run_id: str,
        callback_url: str,
    ) -> Tuple[str, str, str, str]:
        _, _, merged_texts = _normalize_text_inputs(texts=texts)
        return (merged_texts, str(auth_header), str(run_id), str(callback_url))


class GX10ImageInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texts": ("STRING", {"multiline": True, "default": ""}),
                "auth_header": ("STRING", {"default": "", "multiline": False}),
                "first_frame_image": ("STRING", {"default": "", "multiline": False}),
                "image_path": ("STRING", {"default": "", "multiline": False}),
                "run_id": ("STRING", {"default": "", "multiline": False}),
                "callback_url": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "tts_prompt",
        "image",
        "auth_header",
        "run_id",
        "callback_url",
        "positive_prompt",
        "negative_prompt",
    )
    FUNCTION = "pack"
    OUTPUT_NODE = False
    CATEGORY = "GX10"

    def pack(
        self,
        texts: str,
        auth_header: str,
        first_frame_image: str,
        image_path: str,
        run_id: str,
        callback_url: str,
    ) -> Tuple[str, torch.Tensor, str, str, str, str, str]:
        tts_prompt, positive_prompt, negative_prompt = _split_prompt_triplet(texts=texts)

        image_candidates = [image_path, first_frame_image]
        output_image: torch.Tensor | None = None
        for path in image_candidates:
            if not path:
                continue
            output_image = _to_tensor_from_file(path)
            if output_image is not None:
                break
        if output_image is None:
            output_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        return (
            tts_prompt,
            output_image,
            str(auth_header),
            str(run_id),
            str(callback_url),
            positive_prompt,
            negative_prompt,
        )


class GX10ImageUpload:
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
        auth_header: str = "",
    ) -> Tuple[str]:
        run_id = str(run_id).strip()
        callback_url = str(callback_url).strip()

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
                }
            )

        if len(encoded) == 1:
            payload = dict(
                run_id=run_id,
                format=encoded[0]["format"],
                index=encoded[0]["index"],
                image_b64=encoded[0]["image_b64"],
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
                    }
                    for item in encoded
                ],
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
                "audio": ("STRING", {"default": "", "multiline": False}),
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
        audio: str,
        run_id: str,
        callback_url: str,
        auth_header: str = "",
    ) -> Dict[str, object]:
        run_id = str(run_id).strip()
        callback_url = str(callback_url).strip()

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
                "audio": str(audio),
            }
        else:
            payload = {
                "run_id": run_id,
                "images": images_payload,
                "audio": str(audio),
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
                "run_id": ("STRING", {"default": "", "multiline": False}),
                "callback_url": ("STRING", {"default": "", "multiline": False}),
                "auth_header": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_path": ("STRING", {"default": "", "multiline": False}),
                "video": ("STRING", {"default": "", "multiline": False}),
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 120.0, "step": 0.1}),
                "filename_prefix": ("STRING", {"default": "gx10_video", "multiline": False}),
                "format": ("STRING", {"default": "video/mp4", "multiline": False}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "save_output": ("BOOLEAN", {"default": True}),
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
        run_id: str,
        callback_url: str,
        auth_header: str = "",
        audio=None,
        audio_path: str = "",
        video: str = "",
        images=None,
        frame_rate: float = 8.0,
        filename_prefix: str = "gx10_video",
        format: str = "video/mp4",
        pingpong: bool = False,
        loop_count: int = 0,
        save_output: bool = True,
        hls_url: str = "",
        hls_base_url: str = "",
        prefer_hls: bool = True,
        hls_only: bool = True,
    ) -> Dict[str, object]:
        audio_temp_files: List[str] = []
        run_id = str(run_id).strip()
        callback_url = str(callback_url).strip()
        audio_path_input = str(audio_path or "").strip()
        video = str(video or "").strip()
        frame_rate_value = float(frame_rate) if frame_rate else 8.0
        try:
            audio_path = _coerce_audio_path(audio, audio_temp_files)
            if not audio_path and audio_path_input:
                audio_path = _coerce_audio_path(audio_path_input, audio_temp_files)
            source_url = ""
            if images is not None:
                source_url = _build_video_from_frames(
                    frames=_coerce_video_frames(images),
                    frame_rate=frame_rate_value,
                    filename_prefix=str(filename_prefix or "gx10_video").strip(),
                    audio=audio_path,
                    save_output=bool(save_output),
                    loop_count=int(loop_count or 0),
                    pingpong=bool(pingpong),
                    format_value=str(format or "video/mp4"),
                )
            if not source_url:
                source_url = _coerce_video_path(video)
            if not source_url:
                return {"ui": {"video": []}}

            result_url = _resolve_hls_media(
                video=source_url,
                prefer_hls=bool(prefer_hls),
                hls_only=bool(hls_only),
                hls_url=hls_url,
                hls_base_url=hls_base_url,
            )
            if not result_url:
                result_url = source_url

            payload = {
                "run_id": run_id,
                "status": "succeeded",
                "result_url": source_url,
                "hls_url": result_url if _is_hls_url(result_url) else None,
                "metadata": {
                    "format": str(format or "").strip(),
                    "frame_rate": frame_rate_value,
                    "pingpong": bool(pingpong),
                    "loop_count": int(loop_count or 0),
                    "save_output": bool(save_output),
                },
            }
            if audio_path:
                payload["audio"] = audio_path
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
        finally:
            for path in audio_temp_files:
                try:
                    Path(path).unlink()
                except Exception:
                    pass


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


def _coerce_audio_path(value: object, temp_audio_paths: List[str] | None = None) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value.strip().replace("file://", "", 1) if value.strip().startswith("file://") else value.strip()
    if isinstance(value, Path):
        return str(value).strip()
    if isinstance(value, dict):
        for key in ("path", "filename", "file", "uri", "audio", "source", "url"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        waveform = value.get("waveform") if isinstance(value, dict) else None
        sample_rate = value.get("sample_rate") if isinstance(value, dict) else None
        if waveform is not None:
            return _coerce_audio_waveform(waveform, sample_rate, temp_audio_paths)
    if isinstance(value, (list, tuple)):
        for item in value:
            candidate = _coerce_audio_path(item, temp_audio_paths)
            if candidate:
                return candidate
    return ""


def _coerce_audio_waveform(
    waveform: object,
    sample_rate: object = 44100,
    temp_audio_paths: List[str] | None = None,
) -> str:
    try:
        arr = torch.as_tensor(waveform, dtype=torch.float32)
    except Exception:
        return ""
    if not torch.is_floating_point(arr):
        arr = arr.to(dtype=torch.float32)
    if arr.ndim > 3:
        return ""
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.unsqueeze(0)
    if arr.ndim != 2:
        return ""

    # Normalize to (channels, samples).
    if arr.shape[0] <= 16 and arr.shape[0] <= arr.shape[1]:
        pass
    elif arr.shape[1] <= 16:
        arr = arr.t()
    else:
        arr = arr.reshape(arr.shape[0], -1)

    channels = arr.shape[0]
    if channels <= 0:
        return ""

    try:
        sr = int(sample_rate) if sample_rate else 44100
    except Exception:
        sr = 44100
    if sr <= 0:
        sr = 44100

    arr = arr.clip(-1.0, 1.0)
    arr = (arr * 32767.0).short().cpu().numpy()
    if arr.ndim != 2:
        return ""
    if channels > 0:
        samples = arr.transpose(1, 0)
    else:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(int(channels))
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(samples.tobytes())
        if temp_audio_paths is not None:
            temp_audio_paths.append(path)
        return path
    except Exception:
        try:
            Path(path).unlink()
        except Exception:
            pass
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


def _build_hls_path(video_path: str) -> str:
    base = Path(video_path)
    if not base.suffix:
        return str(base.with_name(f"{base.name}.m3u8"))
    return str(base.with_suffix(".m3u8"))


def _ensure_hls_playlist(video_path: str) -> str:
    source = Path(_coerce_video_path(video_path))
    if not source.exists() or not source.is_file():
        return ""
    if source.suffix.lower() == ".m3u8":
        return str(source)
    if source.suffix.lower() not in {
        ".mp4",
        ".webm",
        ".mov",
        ".avi",
        ".mkv",
        ".m4v",
        ".flv",
    }:
        return ""

    playlist = Path(_build_hls_path(str(source)))
    if playlist.exists():
        if not playlist.is_file():
            return ""
        return str(playlist)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return ""

    segment_template = str(playlist.with_name(f"{playlist.stem}_%03d.ts"))
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        "-f",
        "hls",
        "-hls_list_size",
        "0",
        "-hls_flags",
        "append_list",
        "-hls_segment_filename",
        segment_template,
        str(playlist),
    ]

    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        return ""
    if not playlist.exists():
        return ""
    return str(playlist)


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
        generated = _ensure_hls_playlist(str(path))
        if generated:
            return _to_public_url(generated, hls_base_url)

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
