from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Category(str, Enum):
    FOOD_DRINK = "food_drink"
    GROCERIES = "groceries"
    TRANSPORT = "transport"
    TRAVEL = "travel"
    ENTERTAINMENT = "entertainment"
    SHOPPING = "shopping"
    BILLS = "bills"
    HEALTH = "health"
    SUBSCRIPTIONS = "subscriptions"
    OTHER = "other"


class Expense(BaseModel):
    amount: Optional[float] = Field(default=None)
    currency: Optional[str] = Field(default="USD", max_length=3)
    category: Category = Field(default=Category.OTHER)
    merchant: Optional[str] = None
    description: Optional[str] = None
    date: Optional[str] = None

    @field_validator("currency")
    @classmethod
    def _upper(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if isinstance(v, str) else v


SCHEMA_JSON = Expense.model_json_schema()
