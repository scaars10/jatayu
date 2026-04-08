import re

text = """
Here is a **bold** text and an *italic* one.
---
### Headers
## Head 2
# Head 1
"""

def _format_for_telegram(t: str) -> str:
    # Bold **text** -> *text*
    t = re.sub(r'\*\*([^\*]+)\*\*', r'*\1*', t)
    # Headers -> Bold
    t = re.sub(r'^#+\s+(.+)$', r'*\1*', t, flags=re.MULTILINE)
    # Horizontal rules -> unicode line
    t = re.sub(r'^\s*[-_]{3,}\s*$', '──────────────', t, flags=re.MULTILINE)
    return t

print(_format_for_telegram(text))
