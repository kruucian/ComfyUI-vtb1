# ComfyUI GX10 Nodes (`extsrc/ComfyUI-vtb1`)

This folder contains custom ComfyUI nodes for GX10 orchestrator integration.

Nodes
- `GX10TextInput`
  - Prompt input node with multi-prompt support.
  - Uses three entry points:
    - `prompt`
    - `text`
    - `texts` (newline, comma, or JSON array style)
  - If `texts` has multiple values, they are merged into a single `prompt` with newline separators.
  - Also outputs:
    - `prompt`: merged prompt string
    - `text`: base prompt
    - `texts`: JSON array string for downstream handling
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
    - `callback_url`, `run_id`, `metadata_json`, `mode`
    - optional: `filename_prefix`, `image_format`
  - Sends the same artifact payload as `GX10ImageUpload` so you can wire it where `SaveImage` is usually used.
- `GX10SaveVideo`
  - Save-video style callback node (VHS output compatible).
  - Input:
    - `video` (string path / URI)
    - `callback_url`, `run_id`, `metadata_json`, `status`, `mode`
  - Sends `/callbacks/artifact`-compatible payload with `result_url`.

Files
- `gx10_nodes.py`: node implementations.
- `__init__.py`: exposes `NODE_CLASS_MAPPINGS` to ComfyUI loader.

How to install into ComfyUI
1. Copy this folder into ComfyUI `custom_nodes/` (or mount it in Docker as you prefer).
2. Restart ComfyUI.
3. Use `GX10TextInput` and `GX10ImageUpload` in your workflow.
4. Ensure `GX10ImageUpload` has:
   - `images`
   - `run_id`
   - `callback_url`
   - `metadata_json`
   - `mode`
   - `auth_header` (optional)

권장: 기존 `SaveImage`/VHS 노드 대신 `GX10SaveImage`, `GX10SaveVideo`를 사용하면 콜백 인터페이스를 통일하기 쉽습니다.

Required input keys match orchestrator injection:
- `prompt` / `text` / `texts` / `negative_prompt`
- `seed`, `width`, `height`
- `first_frame_image` / `image_path` (for `GX10ImageInput`)
- `image` (optional for `GX10ImageInput`, fallback when no image path)
- `run_id`
- `callback_url`
- `metadata_json`
- `mode`
- `auth_header` for upload endpoint auth when token is required

Callback endpoint behavior
- `mode=t2i` : orchestrator points `callback_url` to `/callbacks/t2i-image`
- `mode=i2v` : orchestrator points `callback_url` to `/callbacks/artifact`
- payload includes metadata (from `metadata_json` if JSON string)
