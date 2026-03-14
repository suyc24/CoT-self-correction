from __future__ import annotations

ABLATION_POSITION = (
    "Mask one head slice on self_attn.o_proj input tensor "
    "(post head-concatenation, pre o_proj linear)."
)

DEFAULT_KEYWORDS = [
    # English
    "wait",
    "no",
    "mistake",
    "incorrect",
    "however",
    "hold on",
    "recheck",
    "let me check",
    "not right",
    "this is wrong",
    # Chinese
    "等一下",
    "等等",
    "不对",
    "有误",
    "错误",
    "这里不对",
    "重新检查",
    "重新计算",
    "重算",
]
