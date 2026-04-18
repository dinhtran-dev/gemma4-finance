from __future__ import annotations

import re
from typing import Optional

from .schema import Category, Expense

_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}

_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}

_CATEGORY_KEYWORDS: dict[Category, tuple[str, ...]] = {
    Category.FOOD_DRINK: ("coffee", "lunch", "dinner", "breakfast", "brunch",
                          "restaurant", "cafe", "bar", "drinks", "chipotle",
                          "starbucks", "mcdonald", "pizza", "burger", "sushi"),
    Category.GROCERIES: ("groceries", "grocery", "supermarket", "whole foods",
                         "trader joe", "safeway", "kroger"),
    Category.TRANSPORT: ("uber", "lyft", "taxi", "cab", "bus", "subway",
                         "metro", "gas", "parking", "toll"),
    Category.TRAVEL: ("flight", "airfare", "airbnb", "hotel", "motel",
                      "booking", "expedia"),
    Category.ENTERTAINMENT: ("movie", "cinema", "concert", "show", "game",
                             "tickets", "museum"),
    Category.SHOPPING: ("amazon", "target", "walmart", "shirt", "shoes",
                        "clothes", "apparel"),
    Category.BILLS: ("electric", "electricity", "water bill", "internet",
                     "rent", "gas bill", "utility"),
    Category.HEALTH: ("pharmacy", "cvs", "walgreens", "doctor", "dentist",
                      "gym", "medicine"),
    Category.SUBSCRIPTIONS: ("netflix", "spotify", "hulu", "subscription",
                             "icloud", "youtube premium", "disney"),
}


def _parse_amount(text: str) -> tuple[Optional[float], Optional[str]]:
    m = re.search(r"([$€£¥])\s*(\d+(?:[.,]\d+)?)", text)
    if m:
        return float(m.group(2).replace(",", ".")), _CURRENCY_SYMBOLS[m.group(1)]

    m = re.search(r"(\d+(?:\.\d+)?)\s*(usd|eur|gbp|jpy|dollars?|bucks?|euros?|yen)",
                  text, re.IGNORECASE)
    if m:
        unit = m.group(2).lower()
        currency = "USD"
        if unit.startswith("eur"):
            currency = "EUR"
        elif unit == "gbp":
            currency = "GBP"
        elif unit in ("jpy", "yen"):
            currency = "JPY"
        return float(m.group(1)), currency

    m = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if m:
        return float(m.group(1)), "USD"

    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    total, current = 0, 0
    matched = False
    for t in tokens:
        if t in _WORD_NUMS:
            matched = True
            v = _WORD_NUMS[t]
            if v == 100:
                current = max(current, 1) * 100
            else:
                current += v
    if matched:
        total += current
        return float(total), "USD"

    return None, None


def _guess_category(text: str) -> Category:
    lower = text.lower()
    for cat, kws in _CATEGORY_KEYWORDS.items():
        if any(k in lower for k in kws):
            return cat
    return Category.OTHER


def _guess_date(text: str) -> Optional[str]:
    lower = text.lower()
    for phrase in ("today", "yesterday", "tomorrow", "last night", "this morning"):
        if phrase in lower:
            return phrase
    m = re.search(r"\b(last|this)\s+(mon|tues|wednes|thurs|fri|satur|sun)day\b",
                  lower)
    if m:
        return m.group(0)
    m = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if m:
        return m.group(0)
    return None


def parse_fallback(user_text: str) -> Expense:
    amount, currency = _parse_amount(user_text)
    return Expense(
        amount=amount,
        currency=currency or "USD",
        category=_guess_category(user_text),
        merchant=None,
        description=None,
        date=_guess_date(user_text),
    )
