# Copyright 2025 Individual Contributor: Kaichen Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from uuid import uuid4

import cv2
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


from verl.utils.dataset.vision_utils import process_image
from qwen_vl_utils import fetch_video


# Thread pool for blocking video decode operations
_decode_executor = ThreadPoolExecutor(max_workers=8)


def create_image_grid(images: list[Image.Image], padding: int = 10, bg_color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    if not images:
        raise ValueError("images list cannot be empty")
    if len(images) == 1:
        return images[0]
    import math
    img_width, img_height = images[0].size

    n = len(images)
    images_per_row = max(1, math.ceil(math.sqrt(n)))
    nrows = (n + images_per_row - 1) // images_per_row
    ncols = min(images_per_row, n)

    total_width = ncols * img_width + (ncols + 1) * padding
    total_height = nrows * img_height + (nrows + 1) * padding

    grid_img = Image.new("RGB", (total_width, total_height), bg_color)

    for idx, img in enumerate(images):
        row = idx // images_per_row
        col = idx % images_per_row
        x = padding + col * (img_width + padding)
        y = padding + row * (img_height + padding)
        grid_img.paste(img, (x, y))

    return grid_img


def _fetch_video_sync(video_ele, image_patch_size=16):
    """Synchronous wrapper for fetch_video, to be called in thread pool."""
    return fetch_video(video_ele, image_patch_size=image_patch_size)


async def _fetch_video_async(video_ele, image_patch_size=16):
    """Run blocking fetch_video in a thread pool to avoid blocking the async event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _decode_executor, _fetch_video_sync, video_ele, image_patch_size
    )


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


VIEW_VISUAL_PROMPT_SCHEMA = OpenAIFunctionToolSchema.model_validate({
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
    }
})


CROP_VIDEO_SCHEMA = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "crop_video",
        "description": "Crop a video to a specified duration.",
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
    }
})


class ViewVisualPromptTool(BaseTool):
    """Local implementation of view_visual_prompt.

    Replaces the MCP-based VideoTools class to eliminate subprocess overhead.
    Directly reads image files from disk instead of going through the MCP server.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None):
        super().__init__(config, tool_schema or VIEW_VISUAL_PROMPT_SCHEMA)
        self._instance_dict: dict[str, dict] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        self._instance_dict[instance_id] = {
            "tool_call_count": 0,
            "image_root": create_kwargs.get("image_root"),
        }
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        try:
            self._instance_dict[instance_id]["tool_call_count"] += 1

            frame_path = parameters.get("frame_path", "")
            image_root = self._instance_dict[instance_id].get("image_root")

            if image_root and not os.path.isabs(frame_path):
                frame_path = os.path.join(image_root, frame_path)

            if not os.path.exists(frame_path):
                error_msg = (
                    f"Generated image path is incorrect. Please call the tool again to regenerate it."
                )
                logger.error(f"[ViewVisualPromptTool] Image not found: {frame_path}")
                return ToolResponse(text=error_msg), 0.0, {"error": f"Image not found: {frame_path}"}

            # Read and decode the image
            im = Image.open(frame_path).convert("RGB")
            print(f"[ViewVisualPromptTool] Raw image - type: {type(im)}, size: {im.size}, mode: {im.mode}")
            processed_image = process_image(im, image_patch_size=16)
            print(f"[ViewVisualPromptTool] Processed image - type: {type(processed_image)}, size: {processed_image.size}, mode: {processed_image.mode}")
            images = [processed_image]

            success_msg = "The tool executed successfully. Here are the processed result: "
            return ToolResponse(image=images, text=success_msg), 0.0, {}

        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            logger.error(f"[ViewVisualPromptTool] Execution failed: {e}")
            return ToolResponse(text=error_msg), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["tool_call_count"]

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)


class CropVideoTool(BaseTool):
    """Local implementation of crop_video.

    Directly processes video files using qwen_vl_utils instead of going through MCP.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None):
        super().__init__(config, tool_schema or CROP_VIDEO_SCHEMA)
        self._instance_dict: dict[str, dict] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        self._instance_dict[instance_id] = {
            "tool_call_count": 0,
            "video_root": create_kwargs.get("video_root", ""),
        }
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        try:
            self._instance_dict[instance_id]["tool_call_count"] += 1

            video_path = parameters.get("video_path", "")
            start_time = parameters.get("start_time")
            end_time = parameters.get("end_time")

            video_root = self._instance_dict[instance_id].get("video_root")
            if video_root:
                video_path = os.path.join(video_root, video_path)

            # Validate parameters
            if not video_path or video_path.strip() == "":
                logger.error(f"[CropVideoTool] video_path is empty")
                return ToolResponse(text="Error: video_path parameter is required"), 0.0, {"error": "video_path is empty"}
            if start_time is None or end_time is None:
                logger.error(f"[CropVideoTool] missing time params: start_time={start_time}, end_time={end_time}")
                return ToolResponse(text="Error: start_time and end_time are required"), 0.0, {"error": "missing time params"}
            if start_time < 0:
                logger.error(f"[CropVideoTool] invalid start_time: {start_time} (must be non-negative)")
                return ToolResponse(text=f"Error: start_time ({start_time}) must be non-negative"), 0.0, {"error": "invalid start_time"}
            if end_time <= start_time:
                logger.error(f"[CropVideoTool] invalid time range: end_time={end_time} <= start_time={start_time}")
                return ToolResponse(text=f"Error: end_time ({end_time}) must be greater than start_time ({start_time})"), 0.0, {"error": "invalid time range"}

            if not os.path.exists(video_path):
                logger.error(f"[CropVideoTool] video file not found: {video_path}")
                return ToolResponse(text=f"Error: Video file not found: {video_path}"), 0.0, {"error": f"file not found: {video_path}"}

            # Verify video duration
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"[CropVideoTool] cannot open video file: {video_path}")
                return ToolResponse(text=f"Error: Cannot open video file: {video_path}"), 0.0, {"error": "cannot open video"}
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            if start_time >= duration:
                logger.error(f"[CropVideoTool] start_time ({start_time}s) exceeds video duration ({duration:.2f}s)")
                return ToolResponse(text=f"Error: start_time ({start_time}s) exceeds video duration ({duration:.2f}s)"), 0.0, {"error": "start_time exceeds duration"}
            if end_time > duration:
                logger.error(f"[CropVideoTool] end_time ({end_time}s) exceeds video duration ({duration:.2f}s)")
                return ToolResponse(text=f"Error: end_time ({end_time}s) exceeds video duration ({duration:.2f}s)"), 0.0, {"error": "end_time exceeds duration"}

            # Fetch and process video frames
            video_ele = {
                "type": "video",
                "video": f"{video_path}",
                "fps": 1,
                "min_frames": 4,
                "max_frames": 64,
                "min_pixels": 12544,
                "max_pixels": 336 * 336,
                "video_start": start_time,
                "video_end": end_time,
            }

            logger.info(f"[CropVideoTool] video_path={video_path}, full_path={video_path}, start_time={start_time}, end_time={end_time}, duration={duration:.2f}s, fps={fps}, frame_count={frame_count}")
            logger.info(f"[CropVideoTool] video_ele params: fps={video_ele['fps']}, min_frames={video_ele['min_frames']}, max_frames={video_ele['max_frames']}, min_pixels={video_ele['min_pixels']}, max_pixels={video_ele['max_pixels']}, video_start={video_ele['video_start']}, video_end={video_ele['video_end']}")
            # Run blocking fetch_video in thread pool to avoid blocking async event loop
            video_frames = await _fetch_video_async(video_ele, image_patch_size=16)


            print(f"[CropVideoTool] video_frames - type: {type(video_frames)}, shape: {video_frames.shape}, dtype: {video_frames.dtype}")
            video_frames = video_frames.to(torch.uint8)
            pil_images = [to_pil_image(frame).convert("RGB") for frame in video_frames]
            print(f"[CropVideoTool] pil_images[{0}] - type: {type(pil_images[0])}, size: {pil_images[0].size}, mode: {pil_images[0].mode}")
            processed_images = [process_image(img) for img in pil_images]
            print(f"[CropVideoTool] processed_images[{0}] - type: {type(processed_images[0])}, size: {processed_images[0].size}, mode: {processed_images[0].mode}")

            grid_image = create_image_grid(processed_images, padding=10)
            print(f"[CropVideoTool] grid_image - type: {type(grid_image)}, size: {grid_image.size}, mode: {grid_image.mode}")

            # Save grid image to tmp directory
            save_dir = "/mnt/tidal-alsh01/dataset/zeus/zhaoy/tmp"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"grid_{uuid4().hex}.png")
            grid_image.save(save_path)
            logger.error(f"[CropVideoTool] Grid image saved to: {save_path}")

            success_msg = f"The tool executed successfully. Here are the processed result:"
            return ToolResponse(image=[grid_image], text=success_msg), 0.0, {}

        except Exception as e:
            import traceback
            error_msg = f"Tool execution failed: {e}"
            logger.error(f"[CropVideoTool] Execution failed: {e}\n{traceback.format_exc()}")
            return ToolResponse(text=error_msg), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["tool_call_count"]

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)


if __name__ == "__main__":
    import asyncio

    IMAGE_PATH = "/mnt/tidal-alsh01/dataset/zeus/zhaoy/Thinking_V2P_Videos/data/LLaVA-Video-178K/1_2_m_nextqa/sam3_vp_frames/NextQA/NExTVideo/0002/2828919525/0002-2828919525/frame_000390_triangle_cyan_20260401_151048.png"
    VIDEO_PATH = "/mnt/tidal-alsh01/dataset/zeus/zhaoy/DATA/-YwrMtiqHKg.mp4"

    async def main():
        # Test 1: ViewVisualPromptTool
        print("=" * 60)
        print("Test 1: ViewVisualPromptTool")
        print("=" * 60)
        view_tool = ViewVisualPromptTool(config={})
        instance_id, _ = await view_tool.create()
        try:
            response, cost, info = await view_tool.execute(
                instance_id=instance_id,
                parameters={"frame_path": IMAGE_PATH},
            )
            print(f"  Response text : {response.text}")
            print(f"  Response image list length: {len(response.image) if response.image else 0}")
            if response.image:
                img = response.image[0]
                print(f"  Image type   : {type(img)}")
                print(f"  Image size   : {img.size}")
                print(f"  Image mode   : {img.mode}")
            print(f"  Cost         : {cost}")
            print(f"  Info         : {info}")
        finally:
            await view_tool.release(instance_id)

        print()

        # Test 2: CropVideoTool
        print("=" * 60)
        print("Test 2: CropVideoTool")
        print("=" * 60)
        crop_tool = CropVideoTool(config={})
        instance_id, _ = await crop_tool.create()
        try:
            response, cost, info = await crop_tool.execute(
                instance_id=instance_id,
                parameters={
                    "video_path": VIDEO_PATH,
                    "start_time": 0.0,
                    "end_time": 60.0,
                },
            )
            print(f"  Response text : {response.text}")
            print(f"  Response image list length: {len(response.image) if response.image else 0}")
            if response.image:
                for i, img in enumerate(response.image):
                    print(f"  Image[{i}] type : {type(img)}")
                    print(f"  Image[{i}] size : {img.size}")
                    print(f"  Image[{i}] mode : {img.mode}")
                    save_path = f"/tmp/test_crop_video_grid_{i}.png"
                    img.save(save_path)
                    print(f"  Image[{i}] saved to: {save_path}")
            print(f"  Cost           : {cost}")
            print(f"  Info           : {info}")
        finally:
            await crop_tool.release(instance_id)

        print()
        print("Done.")

    asyncio.run(main())