"""Travel Agent 后端包。

分层约定（后续阶段不要打破）：
    api/        只做 HTTP：参数校验、异常翻译、组装响应
    services/   业务逻辑：筛选、排序、计算
    providers/  外部数据源：只取数，不做偏好判断
    schemas/    Pydantic 数据结构：跨层共用的唯一契约
"""
