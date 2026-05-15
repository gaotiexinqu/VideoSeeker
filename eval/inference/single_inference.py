#!/usr/bin/env python3
"""
Unified inference script: supports dynamic tool selection via --tools parameter.

Supported tools:
  - view_visual_prompt : load pre-extracted visual prompt frames
  - crop_video         : crop video segments by time range

Usage examples:
  python single_inference.py --dataset_info_path xxx.json --tools view_visual_prompt
  python single_inference.py --dataset_info_path xxx.json --tools view_visual_prompt,crop_video
  python single_inference.py --no_tool                           # disable all tools
"""

import argparse
import base64
import json
import os
import re
import sys
from io import BytesIO
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

# ============================================================
# Tool Schema definitions
# ============================================================

VIEW_VISUAL_PROMPT_TOOL = {
    "type": "function",
    "function": {
        "name": "view_visual_prompt",
        "description": "View the visual prompt image for the current question.",
        "parameters": {
            "type": "object",
            "properties": {
                "frame_path": {
                    "type": "string",
                    "description": "Path to the visual prompt frame image file",
                },
            },
            "required": ["frame_path"],
        },
    },
}

CROP_VIDEO_TOOL = {
    "type": "function",
    "function": {
        "name": "crop_video",
        "description": "Crop a video to a specified duration, returning extracted frames for inspection.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_path": {
                    "type": "string",
                    "description": "Path to the video file",
                },
                "start_time": {
                    "type": "number",
                    "description": "Start time in seconds",
                },
                "end_time": {
                    "type": "number",
                    "description": "End time in seconds, must be greater than start_time",
                },
            },
            "required": ["video_path", "start_time", "end_time"],
        },
    },
}

# ============================================================
# Tool name -> Schema mapping (for dynamic building)
# ============================================================
TOOL_SCHEMA_REGISTRY = {
    "view_visual_prompt": VIEW_VISUAL_PROMPT_TOOL,
    "crop_video": CROP_VIDEO_TOOL,
}


# ============================================================
# Path config (defaults, can be overridden via CLI)
# ============================================================
DEFAULT_FRAMES_ROOT = "/mnt/tidal-alsh01/dataset/zeus/zhaoy/.cache/huggingface/hub/datasets--gaotiexinqu--V2P-Bench/snapshots/f6f0f5bd11cbbc592d9f4a40591669f4649b204d/frames"
DEFAULT_VIDEOS_ROOT = "/mnt/tidal-alsh01/dataset/zeus/zhaoy/.cache/huggingface/hub/datasets--gaotiexinqu--V2P-Bench/snapshots/f6f0f5bd11cbbc592d9f4a40591669f4649b204d/videos"
DEFAULT_DATASET_INFO_PATH = "/mnt/tidal-alsh01/dataset/zeus/zhaoy/.cache/huggingface/hub/datasets--gaotiexinqu--V2P-Bench/snapshots/f6f0f5bd11cbbc592d9f4a40591669f4649b204d/dataset_info_1148.json"


# ============================================================
# Prompt templates
# ============================================================

# view_visual_prompt tool prompt
# General video understanding dataset tool prompt (no frame_path, for VideoMME etc.)
TOOL_PROMPT_TEMPLATE_CROP = (
    "Think first, call **crop_video** if needed, then answer. "
    "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer>. "
    "The Video path for this video is: {video_path}."
)

# +++ TODO: refine tool switching
TOOL_PROMPT_TEMPLATE_VP = (
    "Think first, call **view_visual_prompt** and if needed, then answer. "
    "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer>. "
    "The visual prompt frame path for this video is: {frame_path}."
)

TOOL_PROMPT_TEMPLATE = (
    "Think first, call **view_visual_prompt** and **crop_video** if needed, then answer. "
    "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer>. "
    "The Video path for this video is: {video_path}, "
    "The visual prompt frame path for this video is: {frame_path}."
)

# Prompt without tools (used with --no_tool)
SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, "
    "then the final answer enclosed in <answer>...</answer> tags. "
    "Please provide a clear, concise response within <answer></answer> tags that directly addresses the question."
)

MC_PROMPT = "\nSelect the best option that accurately addresses the question. Give only your option letter, no other words."


# ============================================================
# Local tool execution functions
# ============================================================

def view_visual_prompt_local(frame_path: str, frames_root: str = DEFAULT_FRAMES_ROOT) -> list:
    """
    Read visual prompt frame image and encode to base64.
    Corresponds to view_visual_prompt tool.
    """
    from PIL import Image

    if not os.path.isabs(frame_path):
        full_path = os.path.join(frames_root, frame_path)
    else:
        full_path = frame_path

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Visual prompt image not found: {full_path}")

    with open(full_path, "rb") as f:
        byte_data = f.read()

    # Auto-detect MIME type
    ext = os.path.splitext(full_path)[-1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    b64 = base64.b64encode(byte_data).decode("utf-8")
    return [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]


def crop_video_local(
    video_path: str,
    start_time: float,
    end_time: float,
    media_root: str = DEFAULT_VIDEOS_ROOT,
    max_frames: int = 64,
    max_pixels: int = 336 * 336,
) -> list:
    """
    Crop video by time range [start_time, end_time] and sample frames, return base64 images.
    Corresponds to crop_video tool. Raises on error, caught by run_inference caller.
    """
    import torch
    import cv2
    from qwen_vl_utils import fetch_video
    from torchvision.transforms.functional import to_pil_image
    from verl.utils.dataset.vision_utils import process_image

    # video_path may be relative (relative to media_root)
    if not os.path.isabs(video_path):
        full_video_path = os.path.join(media_root, video_path)
    else:
        full_video_path = video_path

    if not os.path.exists(full_video_path):
        raise FileNotFoundError(f"Video file not found: {full_video_path}")

    # Validate video duration
    cap = cv2.VideoCapture(full_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {full_video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    if start_time < 0:
        raise ValueError(f"start_time ({start_time}) must be non-negative")
    if end_time <= start_time:
        raise ValueError(f"end_time ({end_time}) must be greater than start_time ({start_time})")
    if start_time >= duration:
        raise ValueError(f"start_time ({start_time}s) exceeds video duration ({duration:.2f}s)")
    if end_time > duration:
        raise ValueError(f"end_time ({end_time}s) exceeds video duration ({duration:.2f}s)")

    # Sample frames
    video_ele = {
        "type": "video",
        "video": f"file://{full_video_path}",
        "fps": 1,
        "min_frames": 4,
        "max_frames": max_frames,
        "min_pixels": 28 * 28,
        "max_pixels": max_pixels,
        "video_start": start_time,
        "video_end": end_time,
    }
    video_frames = fetch_video(video_ele)
    video_frames = video_frames.to(torch.uint8)
    pil_images = [to_pil_image(frame) for frame in video_frames]

    # Return base64 images with interleaved timestamps
    result = []
    for idx, img in enumerate(pil_images):
        timestamp = start_time + idx
        result.append({"type": "text", "text": f"<{timestamp:.1f} seconds>"})
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        result.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    return result


def execute_tool_local(func_name: str, func_args: dict, frames_root: str, media_root: str, dataset: str = "V2P-Bench") -> list:
    """
    Dispatch tool calls to corresponding local execution functions.
    Returns tool_content list (conforms to Qwen-VL multimodal message format).
    Raises on error, caught by caller.
    """
    if func_name == "view_visual_prompt":
        return view_visual_prompt_local(
            frame_path=func_args["frame_path"],
            frames_root=frames_root,
        )
    elif func_name == "crop_video":
        if dataset == "VideoReferSuit":
            return crop_video_from_folder_local(
                folder_path=func_args["video_path"],
                start_time=func_args["start_time"],
                end_time=func_args["end_time"],
                media_root=media_root,
            )
        else:
            return crop_video_local(
                video_path=func_args["video_path"],
                start_time=func_args["start_time"],
                end_time=func_args["end_time"],
                media_root=media_root,
            )
    else:
        raise ValueError(f"Unknown tool: {func_name}")


# ============================================================
# Video frame encoding
# ============================================================

def encode_video_frames(
    video_path: str,
    fps: int = 1,
    max_frames: int = 512,
    max_pixels: int = 200704,
) -> list:
    """
    Uniformly sample frames from video and encode to base64, as visual input for initial question.
    Each frame is preceded by a timestamp text (format aligned with RL rollout training data).

    Args:
        video_path:  absolute path to video file
        fps:         frame sampling rate (default 1 fps, aligned with training)
        max_frames:  max number of frames (default 64, aligned with parquet data)
        max_pixels:  max pixels per frame (default 200704 = 448*448, aligned with training)

    Returns:
        list: interleaved {"type": "text", "text": "<X.X seconds>"} and
              {"type": "image_url", "image_url": {...}} list
    """
    import torch
    from qwen_vl_utils import fetch_video
    from torchvision.transforms.functional import to_pil_image

    video_ele = {
        "type": "video",
        "video": f"file://{video_path}",
        "fps": fps,
        "min_frames": 1,
        "max_frames": max_frames,
        "min_pixels": 28 * 28,
        "max_pixels": max_pixels,
    }
    video_frames = fetch_video(video_ele)
    video_frames = video_frames.to(torch.uint8)
    images = [to_pil_image(frame) for frame in video_frames]

    # frame index / fps = timestamp (seconds), aligned with rollout input "<0.7 seconds>" format
    fps = float(fps) if fps else 1.0
    contents = []
    for idx, img in enumerate(images):
        timestamp = idx / fps
        contents.append({
            "type": "text",
            "text": f"<{timestamp:.1f} seconds>",
        })
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    return contents


def encode_video_frames_from_folder(
    folder_path: str,
    fps: float = 1.0,
    max_frames: int = 512,
    max_pixels: int = 200704,
) -> list:
    """
    Read multiple frames from folder and encode to base64, as visual input for initial question.
    Each frame is preceded by a timestamp text (format: <X seconds>).

    Used for VideoReferSuit dataset (video as folder of consecutive frames, 1s interval).

    Args:
        folder_path:  absolute path to frame folder (contains consecutive frame image files)
        fps:          frame sampling rate (default 1 fps, 1s interval between adjacent frames)
        max_frames:  max number of frames
        max_pixels:  max pixels per frame

    Returns:
        list: interleaved {"type": "text", "text": "<X seconds>"} and
              {"type": "image_url", "image_url": {...}} list
    """
    from PIL import Image

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Frame folder not found: {folder_path}")

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    frame_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in valid_exts
    ]
    frame_files.sort()

    if not frame_files:
        raise ValueError(f"No image files found in folder: {folder_path}")

    if fps <= 0:
        fps = 1.0

    total_frames = len(frame_files)
    if total_frames > max_frames:
        step = max(1, total_frames / max_frames)
        indices = [int(i * step) for i in range(max_frames)]
    else:
        indices = list(range(total_frames))

    contents = []
    for idx, frame_idx in enumerate(indices):
        frame_path = os.path.join(folder_path, frame_files[frame_idx])
        img = Image.open(frame_path).convert("RGB")

        w, h = img.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        timestamp = idx / fps
        contents.append({
            "type": "text",
            "text": f"<{timestamp:.1f} seconds>",
        })
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    return contents


def crop_video_from_folder_local(
    folder_path: str,
    start_time: float,
    end_time: float,
    media_root: str = "",
    max_frames: int = 128,
    max_pixels: int = 224 * 224,
) -> list:
    """
    Crop frames from folder within specified time range and return base64 images.
    Corresponds to crop_video tool, but for VideoReferSuit dataset (folder of frames).

    Args:
        folder_path: relative path to frame folder (relative to media_root) or absolute path
        media_root:  media root directory for resolving relative paths
    """
    from PIL import Image

    # VideoReferSuit: strip .mp4 suffix that model tool call may append
    if folder_path.endswith(".mp4"):
        folder_path = folder_path[:-4]

    if not os.path.isabs(folder_path):
        full_folder_path = os.path.join(media_root, folder_path)
    else:
        full_folder_path = folder_path

    if not os.path.isdir(full_folder_path):
        raise FileNotFoundError(f"Frame folder not found: {full_folder_path}")

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    frame_files = [
        f for f in os.listdir(full_folder_path)
        if os.path.splitext(f.lower())[1] in valid_exts
    ]
    frame_files.sort()

    if not frame_files:
        raise ValueError(f"No image files found in folder: {full_folder_path}")

    total_frames = len(frame_files)

    start_frame = max(0, int(start_time))
    end_frame = min(total_frames - 1, int(end_time))

    if start_frame >= total_frames:
        raise ValueError(f"start_time ({start_time}) exceeds total frames ({total_frames})")
    if end_frame <= start_frame:
        raise ValueError(f"end_time ({end_time}) must be greater than start_time ({start_time})")

    num_frames = end_frame - start_frame + 1
    if num_frames > max_frames:
        step = num_frames / max_frames
        indices = [start_frame + int(i * step) for i in range(max_frames)]
    else:
        indices = list(range(start_frame, end_frame + 1))

    result = []
    for idx, frame_idx in enumerate(indices):
        frame_path = os.path.join(full_folder_path, frame_files[frame_idx])
        img = Image.open(frame_path).convert("RGB")

        w, h = img.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        timestamp = start_frame + idx
        result.append({"type": "text", "text": f"<{timestamp:.1f} seconds>"})
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        result.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    return result


# ============================================================
# Main inference function
# ============================================================

def run_inference(
    video_path: str,
    question: str,
    frame_path: str,
    tools: list[dict],
    frames_root: str = DEFAULT_FRAMES_ROOT,
    media_root: str = DEFAULT_VIDEOS_ROOT,
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    model_name: Optional[str] = None,
    max_tokens: int = 8192,
    fps: int = 1,
    max_frames: int = 512,
    max_pixels: int = 200704,
    max_tool_rounds: int = 5,
    mode: str = "tool",
    verbose: bool = True,
    dataset: str = "V2P-Bench",
) -> dict:
    """
    Run inference for a single video + question, supports dynamic tool calling.

    Args:
        video_path:      absolute video path
        question:        question text
        frame_path:      relative path to visual prompt frame (can be empty string)
        tools:           tool schema list (e.g. [VIEW_VISUAL_PROMPT_TOOL, CROP_VIDEO_TOOL])
        frames_root:     root directory for visual prompt frames
        media_root:      root directory for video files (for crop_video relative path resolution)
        api_base:        vLLM server URL
        api_key:         API key
        model_name:      model name (None to auto-fetch from server)
        max_tokens:      max tokens to generate
        fps:             video frame sampling rate
        max_frames:      max frames to sample
        max_pixels:      max pixels per frame
        max_tool_rounds: max tool calling rounds
        mode:            inference mode: "direct"=direct answer, "reasoning"=reasoning, "tool"=tool reasoning
        verbose:         whether to print verbose logs

    Returns:
        dict: {
            "response": final text (concatenated across rounds),
            "raw_output": complete raw output string,
            "tool_calls": tool call records list,
            "num_rounds": total rounds,
        }
    """
    client = OpenAI(api_key=api_key, base_url=api_base, timeout=600)  # 600s timeout

    if model_name is None:
        models = client.models.list()
        model_name = models.data[0].id
        if verbose:
            print(f"[INFO] Using model: {model_name}")

    if verbose:
        print(f"[INFO] Encoding video: {video_path}")
    if dataset == "VideoReferSuit":
        video_contents = encode_video_frames_from_folder(
            video_path, fps=fps, max_frames=max_frames, max_pixels=max_pixels
        )
    else:
        video_contents = encode_video_frames(
            video_path, fps=fps, max_frames=max_frames, max_pixels=max_pixels
        )
    if verbose:
        print(f"[INFO] Encoded {len(video_contents)} frames, tools={ [t['function']['name'] for t in tools] if tools else 'none' }")

    # Build user content
    if mode == "direct":
        # Direct mode: no SYSTEM_PROMPT, no tools
        if dataset == "VideoSIAH-Eval":
            user_text = question
        else:
            user_text = f"{question} {MC_PROMPT}"
    elif mode == "reasoning":
        # Reasoning mode: with SYSTEM_PROMPT, no tools
        user_text = f"{SYSTEM_PROMPT}\n\n{question}"
    else:  # mode == "tool"
        # Tool reasoning mode
        if tools:
            tool_names = ",".join([t['function']['name'] for t in tools])
            if dataset == "VideoMME" or dataset == "LongVideoBench" or dataset == "VideoSIAH-Eval":
                video_path_in_prompt = os.path.basename(video_path)
            elif dataset == "VideoReferSuit":
                # VideoReferSuit: video_path is folder number (e.g. "0")
                # Add .mp4 suffix for model to recognize as video; tool execution strips it
                video_path_in_prompt = os.path.basename(video_path) + ".mp4"
            elif dataset == "V2P-Bench":
                # video_path = .../videos/xxx.mp4, keep original relative path format
                video_path_in_prompt = os.path.basename(os.path.dirname(video_path)) + "/" + os.path.basename(video_path)
            else:
                print(f"[ERROR] Unknown dataset: {dataset}")
                sys.exit(1)

            if dataset == "VideoMME" or dataset == "LongVideoBench" or dataset == "VideoSIAH-Eval":
                tool_prompt = TOOL_PROMPT_TEMPLATE_CROP.format(video_path=video_path_in_prompt)
            else:
                if tool_names == "view_visual_prompt":
                    tool_prompt = TOOL_PROMPT_TEMPLATE_VP.format(
                        video_path=video_path_in_prompt,
                        frame_path=frame_path,
                    )
                elif tool_names == "view_visual_prompt,crop_video":
                    tool_prompt = TOOL_PROMPT_TEMPLATE.format(
                    video_path=video_path_in_prompt,
                    frame_path=frame_path,
                )
                else:
                    print(f"[ERROR] Unknown tools: {tools}")
                    sys.exit(1)
            user_text = f"{SYSTEM_PROMPT}\n\n{question} {tool_prompt}"
        else:
            # No tools provided, degrade to direct answer
            if dataset == "VideoSIAH-Eval":
                user_text = question
            else:
                user_text = f"{question} {MC_PROMPT}"

    user_content = video_contents + [{"type": "text", "text": user_text}]
    messages = [{"role": "user", "content": user_content}]

    if verbose:
        print(f"user_text: \n {user_text}")

    all_responses = []
    tool_call_history = []
    # Concatenate complete raw output string (aligned with training predict_str format, directly usable in compute_score)
    raw_output_parts: list[str] = []

    for round_idx in range(max_tool_rounds + 1):
        if verbose:
            print(f"\n[ROUND {round_idx}] Calling model...")

        kwargs = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }

        if mode == "tool" and tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if message.content:
            all_responses.append(message.content)
            raw_output_parts.append(message.content)
            if verbose:
                content = (
                    message.content[:500] + "..."
                    if len(message.content) > 500
                    else message.content
                )
                print(f"[RESPONSE] {content}")

        if finish_reason == "tool_calls" and message.tool_calls:
            if verbose:
                print(f"[TOOL CALLS] Processing {len(message.tool_calls)} call(s)")

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if verbose:
                    print(f"[TOOL] {func_name}: {func_args}")

                tool_call_history.append({"name": func_name, "arguments": func_args})

                raw_output_parts.append(
                    f'<tool_call>\n{{"name": "{func_name}", "arguments": {json.dumps(func_args, ensure_ascii=False)}}}\n</tool_call>'
                )

                # Execute tool, on error append error text to context for continued inference
                try:
                    result_content = execute_tool_local(
                        func_name=func_name,
                        func_args=func_args,
                        frames_root=frames_root,
                        media_root=media_root,
                        dataset=dataset,
                    )
                    if verbose:
                        print(f"[TOOL RESULT] {func_name} succeeded, {len(result_content)} items")

                    tool_content = result_content + [
                        {"type": "text", "text": "The tool executed successfully. Here are the processed result: "}
                    ]
                    raw_output_parts.append(
                        "user\n<tool_response>\nThe tool executed successfully. Here are the processed result: \n</tool_response>\nassistant\n"
                    )
                except Exception as e:
                    if verbose:
                        print(f"[TOOL ERROR] {e}")
                    err_text = f"Tool execution failed: {e}"
                    tool_content = [{"type": "text", "text": err_text}]
                    raw_output_parts.append(
                        f"user\n<tool_response>\n{err_text}\n</tool_response>\nassistant\n"
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content,
                    }
                )
        else:
            break

    return {
        "response": "\n".join(all_responses),
        "raw_output": "".join(raw_output_parts),
        "user_text": user_text,
        "tool_calls": tool_call_history,
        "num_rounds": round_idx + 1,
    }


# ============================================================
# Helper functions
# ============================================================

def _strip_eval_instruction(question: str) -> str:
    """
    Strip the original evaluation instruction from the end of dataset_info.json questions.
    "Select the best option that accurately addresses the question. ..."
    """
    question = re.sub(r"\nSelect the best option.*$", "", question, flags=re.DOTALL)
    return question.strip()


def _print_result(result: dict):
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Tool calls : {len(result['tool_calls'])}")
    for i, tc in enumerate(result["tool_calls"]):
        print(f"  [{i+1}] {tc['name']}: {tc['arguments']}")
    print(f"Rounds     : {result['num_rounds']}")
    print("-" * 60)
    print(result["response"])


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="V2P / Refer / General Video Understanding Inference with dynamic tools"
    )

    # ── Single inference params ────────────────────────────────────────────────────────
    parser.add_argument("--video_path", type=str, default="", help="absolute video path (single mode)")
    parser.add_argument("--frame_path", type=str, default="",
                        help="relative path to visual prompt frame (single mode)")
    parser.add_argument("--question", type=str, default="", help="question text (single mode)")
    parser.add_argument("--meta_path", type=str, default="", help="(reserved)")

    # ── Batch inference params ───────────────────────────────────────────────────────
    parser.add_argument("--dataset", type=str, default="V2P-Bench",
                        choices=["V2P-Bench", "VideoReferSuit", "VideoMME", "LongVideoBench", "VideoSIAH-Eval"],
                        help="dataset type: V2P-Bench=video files, VideoReferSuit=folder frames, VideoMME/LongVideoBench/VideoSIAH-Eval=general video understanding")
    parser.add_argument("--dataset_info_path", type=str, default=DEFAULT_DATASET_INFO_PATH,
                        help="dataset JSON path")
    parser.add_argument("--videos_root", type=str, default=DEFAULT_VIDEOS_ROOT,
                        help="video root directory (prefix for video_path in dataset_info.json)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="start index (default 0)")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="end index (exclusive, -1 means end of dataset)")
    parser.add_argument("--save_path", type=str, default="", help="result save path (batch mode)")

    # ── Inference mode config ─────────────────────────────────────────────────────────
    parser.add_argument("--mode", type=str, default="tool",
                        choices=["direct", "reasoning", "tool"],
                        help="inference mode: direct=direct answer, reasoning=with reasoning, tool=tool reasoning (degrades to direct when no tools)")
    parser.add_argument("--tools", type=str, default="view_visual_prompt",
                        help="comma-separated tool names, e.g. 'view_visual_prompt' or 'view_visual_prompt,crop_video'. Only effective when mode=tool.")
    parser.add_argument("--no_tool", action="store_true",
                        help="(deprecated, kept for compatibility)")

    # ── Common params ─────────────────────────────────────────────────────────────
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--max_pixels", type=int, default=200704,
                        help="max pixels per frame (448*448=200704)")
    parser.add_argument("--frames_root", type=str, default=DEFAULT_FRAMES_ROOT,
                        help="visual prompt frames root directory")
    parser.add_argument("--media_root", type=str, default=DEFAULT_VIDEOS_ROOT,
                        help="video files root directory (for crop_video relative path resolution)")
    parser.add_argument("--quiet", action="store_true", help="reduce output")

    args = parser.parse_args()

    # ── Parse mode and tool list ───────────────────────────────────────────────────────
    mode = args.mode
    if args.no_tool:
        mode = "direct"

    if mode == "tool" and args.tools.strip() == "":
        mode = "direct"
        print(f"[INFO] Tools empty in tool mode, degraded to direct mode")

    if mode == "tool":
        tool_names = [t.strip() for t in args.tools.split(",") if t.strip()]
        tool_schemas = [TOOL_SCHEMA_REGISTRY[name] for name in tool_names]
        for name in tool_names:
            if name not in TOOL_SCHEMA_REGISTRY:
                print(f"[ERROR] Unknown tool: '{name}'. Available: {list(TOOL_SCHEMA_REGISTRY.keys())}")
                sys.exit(1)
    else:
        tool_names = []
        tool_schemas = []

    print(f"[INFO] Mode: {mode}, Tools: {tool_names if tool_names else 'none'}")

    # Common inference params
    _common = dict(
        frames_root=args.frames_root,
        media_root=args.media_root,
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        fps=args.fps,
        max_frames=args.max_frames,
        max_pixels=args.max_pixels,
        mode=mode,
        verbose=not args.quiet,
        dataset=args.dataset,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Mode 1: Single inference
    # ══════════════════════════════════════════════════════════════════════════
    if not args.save_path and not args.meta_path:
        if not args.video_path or not os.path.exists(args.video_path):
            print(f"Error: Video not found: {args.video_path}")
            sys.exit(1)
        if mode == "tool" and not args.frame_path and "view_visual_prompt" in tool_names:
            print("Error: --frame_path is required when view_visual_prompt tool is enabled.")
            sys.exit(1)

        result = run_inference(
            video_path=args.video_path,
            question=args.question,
            frame_path=args.frame_path,
            tools=tool_schemas,
            **_common,
        )
        _print_result(result)

    # ══════════════════════════════════════════════════════════════════════════
    # Mode 2: Batch inference
    # ══════════════════════════════════════════════════════════════════════════
    elif args.save_path and not args.meta_path:
        if not os.path.exists(args.dataset_info_path):
            print(f"Error: dataset_info.json not found: {args.dataset_info_path}")
            sys.exit(1)
        if not args.save_path:
            print("Error: --save_path is required for batch mode.")
            sys.exit(1)

        with open(args.dataset_info_path) as f:
            dataset = json.load(f)

        end_idx = args.end_idx if args.end_idx != -1 else len(dataset)
        subset = dataset[args.start_idx:end_idx]
        print(f"[INFO] dataset_info.json: total={len(dataset)}, "
              f"running [{args.start_idx}, {end_idx}) = {len(subset)} items")

        # Skip if result file already exists with all items
        if os.path.exists(args.save_path):
            existing = []
            with open(args.save_path, encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line:
                        try:
                            existing.append(json.loads(_line))
                        except json.JSONDecodeError:
                            pass
            if len(existing) >= len(subset):
                print(f"[INFO] Result file already exists and complete ({len(existing)} items), skipping inference.")
                print(f"[INFO] Run evaluation script --result_path {args.save_path} to see accuracy.")
                sys.exit(0)

        # Truncate file to avoid leftover results from previous runs
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True) \
            if os.path.dirname(args.save_path) else None
        open(args.save_path, "w").close()
        print(f"[INFO] Truncated output file: {args.save_path}")

        results = []
        for item in tqdm(subset, desc="Batch inference"):
            try:
                video_path = os.path.join(args.videos_root, item["video_path"])
                frame_path = item.get("frame_path", "") or ""
                question = _strip_eval_instruction(item["question"])

                # VideoReferSuit: video_path is folder, other datasets are video files
                is_video = os.path.isdir(video_path) if args.dataset == "VideoReferSuit" else os.path.isfile(video_path)
                if not is_video:
                    print(f"[WARN] Video path not found, skipping: {video_path}")
                    out_item = {**item, "response": None, "predict": None, "error": "video_not_found"}
                    results.append(out_item)
                    with open(args.save_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                    continue

                result = run_inference(
                    video_path=video_path,
                    question=question,
                    frame_path=frame_path,
                    tools=tool_schemas,
                    **_common,
                )

                # Extract model predicted option letter
                resp_text = result["response"]
                predict = ""
                m = re.search(r"<answer>\s*([A-Da-d])", resp_text)
                if m:
                    predict = m.group(1).upper()

                out_item = {
                    **item,
                    "response": resp_text,
                    "raw_output": result["raw_output"],
                    "user_text": result.get("user_text"),
                    "predict": predict,
                    "tool_calls": result["tool_calls"],
                    "num_rounds": result["num_rounds"],
                    "correct": (predict == item["answer"]) if predict and "answer" in item else None,
                }
                results.append(out_item)

                with open(args.save_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(out_item, ensure_ascii=False) + "\n")

                if not args.quiet:
                    _print_result(result)
                    print(f"[EVAL] predict={predict}  answer={item.get('answer','?')}  "
                          f"correct={out_item['correct']}"
                          " (for reference only, run eval script for accurate metrics)")

            except Exception as e:
                import traceback
                print(f"[ERROR] video_id={item.get('video_id','?')}  {e}")
                traceback.print_exc()
                out_item = {**item, "response": None, "predict": None, "error": str(e)}
                results.append(out_item)
                with open(args.save_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                continue

        print(f"[INFO] Results saved to: {args.save_path} ({len(results)} items)")
        print(f"[INFO] Run `python calu_acc.py --result_path {args.save_path}` to see accuracy.")

    else:
        print(f"Error: meta_path not found: {args.meta_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
