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

import base64
import logging
import os
from io import BytesIO
from typing import Annotated

import cv2
import torch
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent
from pydantic import Field
from qwen_vl_utils import fetch_video
from torchvision.transforms.functional import to_pil_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MCP_SERVER] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("/mnt/tidal-alsh01/dataset/zeus/zhaoy/Thinking_V2P_Videos/verl/logs/mcp_server.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = FastMCP("Video Tools MCP Server", "0.1.0")

@app.tool(name="crop_video", description="Crop a video to a specified duration.")
def crop_video(
    video_path: Annotated[str, Field(description="Path to the video file")] = None,
    start_time: Annotated[float, Field(description="Start time in seconds")] = None,
    end_time: Annotated[float, Field(description="End time in seconds, must be > start_time")] = None,
) -> list[ImageContent]:
    """
    Crop a video to a specified duration.

    Args:
        video_path (str): Path to the video file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        str: Path to the cropped video file.
    """
    # stdio MCP cannot use print, it will try to parse it as JSON and cause errors
    # print("[log]: Executing crop_video tool...")

    # MCP server cannot use breakpoint()
    # breakpoint()

    # Validate input parameters - now we control all validation logic
    logger.info(f"Validating parameters: video_path={video_path}, start_time={start_time}, end_time={end_time}")

    # Check required parameters
    if video_path is None:
        logger.error("Missing video_path parameter")
        raise ValueError("video_path parameter is required")

    if start_time is None:
        logger.error("Missing start_time parameter")
        raise ValueError("start_time parameter is required")

    if end_time is None:
        logger.error("Missing end_time parameter")
        raise ValueError("end_time parameter is required")

    # Check parameter values
    if not video_path or video_path.strip() == "":
        logger.error(f"Empty video_path parameter: '{video_path}'")
        raise ValueError("video_path cannot be empty")

    if start_time < 0:
        logger.error(f"Invalid start_time: {start_time}")
        raise ValueError(f"start_time must be non-negative, got {start_time}")

    if end_time <= start_time:
        logger.error(f"Invalid time range: start_time={start_time}, end_time={end_time}")
        raise ValueError(f"end_time ({end_time}) must be greater than start_time ({start_time})")

    # Check file existence
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # verify video duration
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    # print(f"video duration: {duration:.2f}s")
    cap.release()

    # validate time range
    if start_time >= duration:
        raise ValueError(f"start_time ({start_time}s) exceeds video duration ({duration:.2f}s)")
    if end_time > duration:
        raise ValueError(f"end_time ({end_time}s) exceeds video duration ({duration:.2f}s)")

    try:
        video_ele = {
            "type": "video",
            "video": f"file://{video_path}",
            "fps": 1,  # 1fps
            "min_frames": 4,
            "max_frames": 64,
            "min_pixels": 28 * 28,
            "max_pixels": 336 * 336,
            "video_start": start_time,
            "video_end": end_time,
        }
        video_frames = fetch_video(video_ele)
        video_frames = video_frames.to(torch.uint8)
        images = [to_pil_image(frame) for frame in video_frames]
        # Encode images to base64
        image_contents = []
        for img in images:
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            image_contents.append(ImageContent(type="image", data=base64_str, mimeType="image/png"))

        return image_contents
    except Exception as e:
        raise RuntimeError(f"Failed to process video {video_path}: {str(e)}") from e


@app.tool(name="view_visual_prompt", description="View the visual prompt image for the current question.")
def view_visual_prompt(
    frame_path: Annotated[str, Field(description="Path to the visual prompt frame image file")],
) -> list[ImageContent]:
    """
    View the visual prompt image for the current question.

    Args:
        frame_path (str): Path to the visual prompt frame image file.

    Returns:
        list[ImageContent]: The visual prompt image encoded as base64.
    """

    logger.info(f"Loading visual prompt image: {frame_path}")

    if not os.path.exists(frame_path):
        logger.error(f"Visual prompt image not found: {frame_path}")
        raise FileNotFoundError(f"Visual prompt image not found: {frame_path}")

    try:
        with open(frame_path, "rb") as f:
            byte_data = f.read()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return [ImageContent(type="image", data=base64_str, mimeType="image/png")]
    except Exception as e:
        raise RuntimeError(f"Failed to load visual prompt image {frame_path}: {str(e)}") from e


if __name__ == "__main__":
    app.run()