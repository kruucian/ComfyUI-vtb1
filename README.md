# ComfyUI GX10 Nodes (`extsrc/ComfyUI-vtb1`)

This folder contains custom ComfyUI nodes for GX10 orchestrator integration.

Nodes
- `GX10TextInput`
  - Pass-through node for prompt/negative/seed/width/height plus GX metadata hints.
- `GX10ImageUpload`
  - Callback output node for image artifacts.
  - Sends callback payload to the `callback_url` injected by GX10 orchestrator.
  - Supports single image and multi-image payloads.

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

Required input keys match orchestrator injection:
- `prompt` / `negative_prompt`
- `seed`, `width`, `height`
- `run_id`
- `callback_url`
- `metadata_json`
- `mode`
- `auth_header` for upload endpoint auth when token is required

Callback endpoint behavior
- `mode=t2i` : orchestrator points `callback_url` to `/callbacks/t2i-image`
- `mode=i2v` : orchestrator points `callback_url` to `/callbacks/artifact`
- payload includes metadata (from `metadata_json` if JSON string)
