# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""
自定义 test reward 函数

这是一个示例 reward function，展示如何实现自定义的奖励计算逻辑。
您可以根据自己的需求修改此函数。
"""


def compute_score(solution_str: str, ground_truth: str, extra_info: dict = None) -> dict:
    """
    计算 test reward 分数
    
    Args:
        solution_str: 模型生成的解决方案字符串
        ground_truth: 真实答案
        extra_info: 额外信息字典（可选）
    
    Returns:
        dict: 包含 score 和其他评估指标的字典
            - score: 主要分数（会被用作训练的 reward）
            - 其他自定义字段：可以包含任何额外的评估信息
    """
    # 示例实现：基于长度和包含关键词来计算分数
    score = 0.0
    
    # 1. 基础分数：检查是否生成了内容
    if len(solution_str.strip()) > 0:
        score += 1.0
    
    # 2. 长度检查：奖励适当长度的回答
    solution_length = len(solution_str.strip())
    if 10 <= solution_length <= 500:
        score += 1.0
    elif solution_length > 500:
        score += 0.5  # 过长的回答给予部分分数
    
    # 3. 关键词匹配：检查是否包含关键内容
    if ground_truth and ground_truth.lower() in solution_str.lower():
        score += 3.0
    
    # 4. 质量评估：检查一些基本的质量指标
    quality_score = 0.0
    
    # 检查是否有完整的句子结构
    if '.' in solution_str or '。' in solution_str:
        quality_score += 1.0
    
    # 检查是否避免了重复
    words = solution_str.split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        quality_score += unique_ratio * 2.0
    
    score += quality_score
    
    # 归一化分数到 0-10 范围
    max_possible_score = 7.0
    normalized_score = (score / max_possible_score) * 10.0
    normalized_score = min(10.0, max(0.0, normalized_score))
    
    # 返回详细结果
    result = {
        'score': normalized_score,  # 主要分数
        'length': solution_length,  # 回答长度
        'has_ground_truth': ground_truth.lower() in solution_str.lower() if ground_truth else False,  # 是否包含答案
        'quality_score': quality_score,  # 质量分数
        'raw_score': score,  # 原始分数
    }
    
    # 如果有额外信息，可以在这里使用
    if extra_info:
        result['extra_info'] = extra_info
    
    return result


# 您也可以添加其他辅助函数
def compute_score_simple(solution_str: str, ground_truth: str, **kwargs) -> float:
    """
    简化版本：只返回一个浮点数分数
    
    Args:
        solution_str: 模型生成的解决方案字符串
        ground_truth: 真实答案
        **kwargs: 其他可选参数
    
    Returns:
        float: 分数（0.0 - 10.0）
    """
    result = compute_score(solution_str, ground_truth)
    return result['score']


# 异步版本示例（如果需要调用外部 API）
async def compute_score_async(
    solution_str: str, 
    ground_truth: str, 
    extra_info: dict = None,
    **kwargs
) -> dict:
    """
    异步版本的 compute_score
    
    当您需要调用外部 API（如 GPT-4 作为评判器）时使用此版本
    
    Args:
        solution_str: 模型生成的解决方案字符串
        ground_truth: 真实答案
        extra_info: 额外信息字典（可选）
        **kwargs: 其他可选参数
    
    Returns:
        dict: 包含 score 和其他评估指标的字典
    """
    # 示例：这里可以调用外部 API
    # import aiohttp
    # async with aiohttp.ClientSession() as session:
    #     async with session.post(api_url, json={...}) as response:
    #         result = await response.json()
    
    # 目前使用同步版本
    return compute_score(solution_str, ground_truth, extra_info)
