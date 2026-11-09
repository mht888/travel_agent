"""测试包。

约定：
- 单元测试直接测 Service / Schema（不启 HTTP）
- 冒烟测试用 TestClient 走完整 HTTP 链路
- 每个阶段的验收标准都要在测试里可断言，不要只靠手点 /docs
"""
