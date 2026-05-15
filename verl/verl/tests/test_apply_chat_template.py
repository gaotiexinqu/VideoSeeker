"""
模拟训练过程中的 apply_chat_template 流程。

对应代码路径：
  sglang_rollout.py::_initialize_tools          → Step 2：加载工具并建立 tool_map
  sglang_rollout.py::_preprocess_prompt_to_...  → Step 3：按样本 tools_kwargs 过滤 tool schemas
  rl_dataset.py::_build_messages                → Step 4：把 <video> 占位符替换为 {"type": "video"}
  schemas.py::AsyncRolloutRequest.__init__       → Step 5：apply_chat_template(tools=...) 生成 prompt
"""

import json
import os
import re
import sys

PROJECT_ROOT = "/mnt/tidal-alsh01/dataset/zeus/zhaoy/Thinking_V2P_Videos/verl"
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoProcessor

from verl.tools.utils.tool_registry import initialize_tools_from_config

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
MODEL_PATH = (
    "/mnt/tidal-alsh01/dataset/zeus/zhaoy/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-VL-4B-Instruct"
    "/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)
MCP_TOOL_CONFIG = os.path.join(
    PROJECT_ROOT, "examples_longvt/video_tools/config/mcp_tool_config.yaml"
)

# ──────────────────────────────────────────────
# 示例数据（来自 zzz.json，代表 parquet 中一行）
# tools_kwargs 的 key 必须与 MCP_TOOL_CONFIG 中注册的工具名一致
# ──────────────────────────────────────────────
SAMPLE = {
    "data_source": "hacs",
    "prompt": [
        {
            "content": (
                "<video>After the man in a pink shirt and white cap drops his piece of paper "
                "at the starting platform, what does he wave to start the kayak race? "
                "Think first, call **crop_video** if needed, then answer. "
                "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> "
                "(if tools needed)  <answer>...</answer>. "
                "The Video path for this video is: -0NUlZvrYY4.mp4"
            ),
            "role": "user",
        }
    ],
    "videos": [
        {
            "fps": 1,
            "max_frames": 512,
            "max_pixels": 50176,
            "min_frames": 1,
            "type": "video",
            "video": "file://-0NUlZvrYY4.mp4",
        }
    ],
    "extra_info": {
        "answer": "A red flag.",
        "index": 1,
        "need_tools_kwargs": True,
        # key 对应 MCP_TOOL_CONFIG 里注册的工具名（此处与问题文本保持一致）
        "tools_kwargs": {
            "crop_video": {"create_kwargs": {"dummy": "dummy"}}
        },
    },
}


# ──────────────────────────────────────────────
# Step 1：加载 Processor
# ──────────────────────────────────────────────
def load_processor(model_path: str) -> AutoProcessor:
    print(f"[1] 加载 Processor: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"    Processor 类型: {type(processor).__name__}")
    return processor


# ──────────────────────────────────────────────
# Step 2：初始化 tool_map
# 对应 sglang_rollout.py::_initialize_tools
#   tool_map = {tool.name: tool for tool in tool_list}
# 注意：使用 get_openai_tool_schema()，与 sglang_rollout.py 保持一致
# ──────────────────────────────────────────────
def load_tool_map(tool_config_path: str) -> dict:
    print(f"\n[2] 初始化 tool_map: {tool_config_path}")
    tool_list = initialize_tools_from_config(tool_config_path)
    tool_map = {tool.name: tool for tool in tool_list}
    print(f"    共加载 {len(tool_map)} 个工具: {list(tool_map.keys())}")
    return tool_map


# ──────────────────────────────────────────────
# Step 3：按样本 tools_kwargs 过滤出当前样本所需的 tool schemas
# 对应 sglang_rollout.py::_preprocess_prompt_to_async_rollout_requests
#   _tool_schemas = [self._tool_map[k].get_openai_tool_schema() for k in _tools_kwargs.keys()]
# ──────────────────────────────────────────────
def get_sample_tool_schemas(tool_map: dict, tools_kwargs: dict) -> list[dict]:
    schemas = []
    for tool_name in tools_kwargs.keys():
        if tool_name not in tool_map:
            print(f"    [警告] tools_kwargs 中的工具 '{tool_name}' 未在 tool_map 中注册，已跳过")
            continue
        schemas.append(tool_map[tool_name].get_openai_tool_schema().model_dump())
    print(f"\n[3] 当前样本使用的工具 schemas ({len(schemas)} 个):")
    for s in schemas:
        print(f"      - {s['function']['name']}: {s['function'].get('description', '')}")
    return schemas


# ──────────────────────────────────────────────
# Step 4：构建 messages
# 对应 rl_dataset.py::_build_messages
# <video> 占位符仅替换为 {"type": "video"}，不携带任何视频元数据
# （视频元数据由后续 process_video() 单独处理）
# ──────────────────────────────────────────────
def build_messages(sample: dict) -> list[dict]:
    messages = [dict(m) for m in sample["prompt"]]

    for message in messages:
        content = message["content"]
        if not isinstance(content, str):
            continue

        segments = re.split(r"(<image>|<video>)", content)
        segments = [s for s in segments if s != ""]
        content_list = []
        for seg in segments:
            if seg == "<image>":
                content_list.append({"type": "image"})
            elif seg == "<video>":
                content_list.append({"type": "video"})
            else:
                content_list.append({"type": "text", "text": seg})
        message["content"] = content_list

    return messages


# ──────────────────────────────────────────────
# Step 5：apply_chat_template
# 对应 schemas.py::AsyncRolloutRequest.__init__ 中：
#   tools = [tool.model_dump() for tool in tool_schemas]
#   raw_prompt = processing_class.apply_chat_template(
#       messages, tools=tools, add_generation_prompt=True, tokenize=False
#   )
# ──────────────────────────────────────────────
def apply_chat_template(
    processor: AutoProcessor,
    messages: list[dict],
    tool_schemas: list[dict],
) -> str:
    raw_prompt = processor.apply_chat_template(
        messages,
        tools=tool_schemas,
        add_generation_prompt=True,
        tokenize=False,
    )
    return raw_prompt


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────
def main():
    processor = load_processor(MODEL_PATH)

    tool_map = load_tool_map(MCP_TOOL_CONFIG)

    tools_kwargs = SAMPLE["extra_info"]["tools_kwargs"]
    tool_schemas = get_sample_tool_schemas(tool_map, tools_kwargs)

    print("\n[4] 构建 messages（<video>/<image> 占位符替换为 {type} dict）")
    messages = build_messages(SAMPLE)
    print("    messages 结构:")
    for i, m in enumerate(messages):
        print(f"      [{i}] role={m['role']}, content={json.dumps(m['content'], ensure_ascii=False)[:120]}...")

    print("\n[5] 执行 apply_chat_template (tokenize=False)")
    raw_prompt = apply_chat_template(processor, messages, tool_schemas)

    print("\n" + "=" * 70)
    print("【apply_chat_template 结果（渲染后字符串）】")
    print("=" * 70)
    print(raw_prompt)
    print("=" * 70)

    # 统计 token 数（不含视觉 token，仅文本占位符）
    text_token_ids = processor.tokenizer(raw_prompt, return_tensors=None)["input_ids"]
    print(f"\n文本 token 数（不含视觉 token 展开）: {len(text_token_ids)}")
    print("\n注意：实际训练时视觉 token 会通过 processor() 进一步展开，")
    print("      导致 input_ids 远大于上述数字（视频帧数越多，token 越多）。")


if __name__ == "__main__":
    main()
