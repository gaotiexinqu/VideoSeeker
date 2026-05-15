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
from typing import Optional
from uuid import uuid4

from PIL import Image

from verl.tools.mcp_base_tool import MCPBaseTool
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VideoTools(MCPBaseTool):
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            Tuple of (instance_id, ToolResponse()).
        """
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        self._instance_dict[instance_id] = {
            "tool_call_count": 0,
            "image_root": create_kwargs.get("image_root"),
            "video_root": create_kwargs.get("video_root"),
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        try:
            self._instance_dict[instance_id]["tool_call_count"] += 1  # tool call count
            image_root = self._instance_dict[instance_id].get("image_root")
            if image_root and "frame_path" in parameters:
                frame_path = parameters["frame_path"]
                if not os.path.isabs(frame_path):
                    parameters = {**parameters, "frame_path": os.path.join(image_root, frame_path)}
            video_root = self._instance_dict[instance_id].get("video_root")
            if video_root and "video_path" in parameters:
                video_path = parameters["video_path"]
                if not os.path.isabs(video_path):
                    parameters = {**parameters, "video_path": os.path.join(video_root, video_path)}
            logger.info(f"[VideoTools] execute called: instance_id={instance_id}, parameters={parameters}")
            # Call the MCP tool here, specifically verl/verl/examples_longvt/video_tools/mcp_server.py
            # _call_tool function calls call_tool to mcp_server.py, then calls _parse_tool_result() function
            result_text, metadata = await self._call_tool(instance_id, parameters)

            # Check for API request errors from MCP call
            # Previous bug: api_error was overwritten to empty string, causing always executing the success path below...
            api_error = metadata.get("api_request_error") or ""
            api_error = api_error.strip() if isinstance(api_error, str) else ""
            if api_error:
                # api_error: Visual prompt image not found: /mnt/tidal-alsh01/dataset/zeus/zhaoy/Thinking_V2P_Videos/data/LLaVA-Video-178K...
                error_msg = f"Generated image path is incorrect. Please call the tool again to regenerate it."
                logger.error(f"[VideoTools] MCP call failed: {api_error}")
                return ToolResponse(text=error_msg), 0.0, {"error": api_error}

            image_list = metadata["images"]
            from verl.utils.dataset.vision_utils import process_image

            images = [process_image(image) for image in image_list]

            # Generate dynamic success message with frame count
            success_msg = "The tool executed successfully. Here are the processed result: "
            logger.error(success_msg)
            return ToolResponse(image=images, text=success_msg), 0.0, {}

        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            logger.error(f"[VideoTools] Execution failed: {e}")
            return ToolResponse(text=error_msg), 0.0, {"error": str(e)}

    # tool call count
    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["tool_call_count"]

    def _parse_tool_result(self, content):
        # Check for text content that might contain error messages
        text_parts = [part.text for part in filter(lambda x: x.type == "text", content)]

        # Look for error indicators in text content
        api_error = ""
        for text in text_parts:
            # Previously wrote the wrong logic here, didn't match the keyword...
            error_keywords = ["Visual prompt image not found", "Video file not found"]
            if any(error_keyword.lower() in text.lower() for error_keyword in error_keywords):
                api_error = text
                logger.error(f"[VideoTools] MCP response contains error: {api_error}")
                return "", {"images": [], "api_request_error": api_error}

        # Parse image content
        image_contents = [part.data for part in filter(lambda x: x.type == "image", content)]

        # Convert base64 string to PIL image
        image_lists = []
        for image_content in image_contents:
            im = Image.open(BytesIO(base64.b64decode(image_content)))
            image_lists.append(im)

        return "", {"images": image_lists, "api_request_error": ""}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
