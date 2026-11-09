"""预算数据结构。

职责边界：
    本文件   只负责「怎么记账」：分类、币种一致、汇总、超支判断
    Service  负责「一天吃多少钱」「机票按什么价估」这类业务估算（第 8 阶段）

关键约束：所有数字都带 DataStatus。估算值不允许被当成真实报价展示。
"""

from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, computed_field, model_validator

from backend.schemas.common import DataStatus, Money

ZERO = Decimal("0")


class BudgetCategory(str, Enum):
    """预算分类，直接对应需求文档的六个拆分项。"""

    FLIGHTS = "flights"
    ACCOMMODATION = "accommodation"
    TRANSPORT = "transport"
    TICKETS = "tickets"
    FOOD = "food"
    OTHER = "other"


class BudgetLineItem(BaseModel):
    """一条预算条目。label 是给人看的，category 是给统计和饼图用的。"""

    category: BudgetCategory
    label: str = Field(..., min_length=1, description="如「往返机票（估算）」")
    amount: Money
    note: str | None = Field(None, description="估算依据，如「按 900 美元/晚 × 4 晚」")


class BudgetBreakdown(BaseModel):
    """整趟旅行的费用账本。

    total / remaining / over_budget / over_budget_amount 都是推导出来的，
    不允许外部传入：否则会出现「条目加起来 980，total 却写 1000」的脏数据。

    之所以不直接给 remaining 一个负数：
        Money 约定 amount >= 0，负数金额会破坏「金额永远是正值」这条全局约定。
        因此超出部分单独放在 over_budget_amount。
    """

    lines: list[BudgetLineItem] = Field(default_factory=list)
    budget_limit: Money | None = Field(
        None, description="用户给的预算上限；None 表示用户没给，不做超支判断"
    )

    @model_validator(mode="after")
    def _validate_currency(self) -> "BudgetBreakdown":
        currencies = {line.amount.currency for line in self.lines}
        if len(currencies) > 1:
            raise ValueError(
                f"预算条目存在多种币种 {sorted(currencies)}，请先统一换算"
            )
        if (
            self.budget_limit is not None
            and currencies
            and self.budget_limit.currency != next(iter(currencies))
        ):
            raise ValueError(
                f"预算上限币种 {self.budget_limit.currency} "
                f"与条目币种 {next(iter(currencies))} 不一致"
            )
        return self

    @property
    def currency(self) -> str | None:
        """账本币种。没有条目时退回预算上限币种；两者都没有则为 None。"""
        if self.lines:
            return self.lines[0].amount.currency
        if self.budget_limit is not None:
            return self.budget_limit.currency
        return None

    @computed_field
    @property
    def total(self) -> Money | None:
        """条目合计。没有条目时返回 None，而不是 0——「没算」和「算了是 0」不是一回事。"""
        if not self.lines:
            return None

        statuses = {line.amount.status for line in self.lines}
        # 来源一致保留原状态；混合来源按最保守处理（有任何 Mock 就标 Mock）
        if len(statuses) == 1:
            status = next(iter(statuses))
        elif DataStatus.MOCK in statuses:
            status = DataStatus.MOCK
        else:
            status = DataStatus.ESTIMATED

        return Money(
            amount=sum((line.amount.amount for line in self.lines), ZERO),
            currency=self.lines[0].amount.currency,
            status=status,
        )

    @computed_field
    @property
    def remaining(self) -> Money | None:
        """预算剩余（只在没超支时有意义）。"""
        total = self.total
        if total is None or self.budget_limit is None:
            return None
        diff = self.budget_limit.amount - total.amount
        if diff <= ZERO:
            return None
        return Money(
            amount=diff,
            currency=self.budget_limit.currency,
            status=total.status,
        )

    @computed_field
    @property
    def over_budget(self) -> bool:
        """是否超出预算。用户没给预算时恒为 False，不能替他假设一个上限。"""
        total = self.total
        if total is None or self.budget_limit is None:
            return False
        return total.amount > self.budget_limit.amount

    @computed_field
    @property
    def over_budget_amount(self) -> Money | None:
        """超出金额。没超支返回 None，与 remaining 互斥。"""
        total = self.total
        if total is None or self.budget_limit is None:
            return None
        diff = total.amount - self.budget_limit.amount
        if diff <= ZERO:
            return None
        return Money(
            amount=diff,
            currency=total.currency,
            status=total.status,
        )
