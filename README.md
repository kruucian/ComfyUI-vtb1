# ComfyUI GX10 Nodes (`extsrc/ComfyUI-vtb1`)

This folder contains custom ComfyUI nodes for GX10 orchestrator integration.

Nodes
- `GX10TextInput`
  - Prompt input node with multi-prompt support.
  - Uses one entry point:
    - `texts` (newline, comma, or JSON array style)
  - If `texts` has multiple values, they are merged into a single `prompt` with newline separators.
  - Also outputs:
    - `prompt`: merged prompt string
    - `text`: base prompt
    - `texts`: JSON array string for downstream handling
    - `run_id`, `callback_url`
  - `seed`, `width`, `height`, `negative_prompt`, `metadata_json`, `mode` are not exposed on this node.
- `GX10ImageInput`
  - i2v input node that carries `IMAGE` and i2v-related prompt inputs.
  - Supports `prompt`, `text`, `texts` just like `GX10TextInput`.
  - Accepts first-frame source as `first_frame_image` or `image_path` and converts to ComfyUI image tensor when provided.
  - `image` input is optional if `image_path` / `first_frame_image` is given.
  - Outputs merged prompt plus image tensor for downstream WAN/i2v pipelines.
- `GX10ImageUpload`
  - Callback output node for image artifacts.
  - Sends callback payload to the `callback_url` injected by GX10 orchestrator.
  - Supports single image and multi-image payloads.
- `GX10SaveImage`
  - Save-image style callback node.
  - Input shape is close to `SaveImage`:
    - `images`
    - `callback_url`, `run_id`, `auth_header`
  - Sends the same artifact payload as `GX10ImageUpload` so you can wire it where `SaveImage` is usually used.
  - 동시에 프리뷰 이미지(`ui.images`)를 출력해 ComfyUI 캔버스에 출력합니다.
- `GX10SaveVideo`
  - Save-video style callback node (VHS output compatible).
  - Input:
    - `video` (string path / URI)
    - `callback_url`, `run_id`
    - `hls_url`, `hls_base_url`, `prefer_hls`, `hls_only` (optional)
  - Sends `/callbacks/artifact`-compatible payload with `result_url` and HLS metadata (`stream_hls_url`, `streaming_protocol`).
  - 동시에 `ui.video`에 최근 6개 완료 영상을 오른쪽 리스트로 누적해서 출력합니다.

Files
- `gx10_nodes.py`: node implementations.
- `__init__.py`: exposes `NODE_CLASS_MAPPINGS` to ComfyUI loader.

How to install into ComfyUI
1. Copy this folder into ComfyUI `custom_nodes/` (or mount it in Docker as you prefer).
2. Restart ComfyUI.
3. Prefer `GX10TextInput` with `GX10SaveImage`/`GX10SaveVideo` at the end of the workflow.
4. If legacy path is needed, use `GX10ImageUpload` for image callbacks.
5. Ensure callback node has:
  - `images`
  - `run_id`
  - `callback_url`
  - `auth_header` (optional)

권장: 기존 `SaveImage`/VHS 노드 대신 `GX10SaveImage`, `GX10SaveVideo`를 사용하면 콜백 인터페이스를 통일하기 쉽습니다.

Required input keys match orchestrator injection:
- `GX10TextInput` path: `texts`, `run_id`, `callback_url`
- `GX10ImageInput` path: `prompt` / `text` / `texts`, `first_frame_image` / `image_path`, `run_id`, `callback_url`
- `first_frame_image` / `image_path` (for `GX10ImageInput`)
- `image` (optional for `GX10ImageInput`, fallback when no image path)
- `run_id`, `callback_url`
- `auth_header` for upload endpoint auth when token is required

Callback endpoint behavior
- `t2i workflow` : orchestrator points `callback_url` to `/callbacks/t2i-image`
- `i2v workflow` : orchestrator points `callback_url` to `/callbacks/artifact`
- payload includes `result_url`; when HLS is used, includes `hls_url` and `metadata.streaming_protocol= hls`.
