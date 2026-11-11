"""LLM 可调用的 Tool 集合（第 5 阶段）。

Tool 只做「参数校验 → 调用 Service → 转成 LLM 可读结果」，
不承载业务逻辑；业务规则一律留在 Service/Router 层。
"""
