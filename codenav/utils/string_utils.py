import re
from typing import Optional


def get_tag_content_from_text(
    text: str,
    tag: str,
) -> Optional[str]:
    pattern = f"<{tag}>" + r"\s*(.*?)\s*" + f"</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    content = match.group(1) if match else None
    if content == "":
        return None
    return content


def str2bool(s: str):
    s = s.lower().strip()
    if s in ["yes", "true", "t", "y", "1"]:
        return True
    elif s in ["no", "false", "f", "n", "0"]:
        return False
    else:
        raise NotImplementedError
