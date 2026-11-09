"""业务异常定义。

约定：
    TravelAgentError  业务可预期错误，带 HTTP 状态码与机器可读 code
    其余未被捕获的异常   统一由 main.py 记为 500，prod 下不暴露细节
"""


class TravelAgentError(Exception):
    """所有业务异常的基类。

    code 给前端做分支判断，message 直接展示给用户，
    所以 message 要写成人能看懂的话，不要把堆栈信息塞进来。
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "travel_agent_error",
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class ProviderUnavailableError(TravelAgentError):
    """外部数据源故障（超时、限流、鉴权失败）。"""

    def __init__(self, provider: str, message: str = "数据源暂时不可用") -> None:
        super().__init__(
            f"{message}：{provider}",
            code="provider_unavailable",
            status_code=503,
        )
