import re

def extract_thinking_answer(response: str):
    """
    从 <answer>...</answer> 中提取首个大写字母 (A/B/C/D)。
    未找到时返回 None。
    """
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    inner = m.group(1).strip()
    letter = re.match(r"^([A-Za-z])", inner)
    if not letter:
        return None
    return letter.group(1).upper()