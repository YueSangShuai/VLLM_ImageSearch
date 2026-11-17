# 校验 prompts.json 与 w8a8_common.structured_prompt() 逐字一致
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from w8a8_common import structured_prompt

prompts = json.load(open(os.path.join(HERE, "prompts.json")))
s = [p for p in prompts if p["tag"] == "structured"][0]["prompt"]
print("structured identical:", s == structured_prompt())
if s != structured_prompt():
    for i, (a, b) in enumerate(zip(s, structured_prompt())):
        if a != b:
            print("first diff @", i, repr(a), repr(b), s[max(0,i-30):i+30])
            break
    print("len:", len(s), len(structured_prompt()))
