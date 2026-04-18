import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expense_parser.fallback import parse_fallback
from expense_parser.prompt import format_training_example
from expense_parser.schema import Category, Expense


def test_schema_defaults():
    e = Expense()
    assert e.currency == "USD"
    assert e.category == Category.OTHER


def test_schema_currency_upper():
    e = Expense(currency="usd")
    assert e.currency == "USD"


def test_format_training_example_round_trip():
    target = {"amount": 15.0, "currency": "USD", "category": "food_drink",
              "merchant": "Starbucks", "description": "coffee", "date": "today"}
    out = format_training_example("spent $15 on coffee at Starbucks", target)
    assert "<start_of_turn>user" in out["text"]
    assert "<start_of_turn>model" in out["text"]
    assert json.dumps(target, ensure_ascii=False, separators=(", ", ": ")) in out["text"]


def test_fallback_parses_dollar_amount():
    e = parse_fallback("spent $15 on coffee at Starbucks today")
    assert e.amount == 15.0
    assert e.currency == "USD"
    assert e.category == Category.FOOD_DRINK
    assert e.date == "today"


def test_fallback_parses_euro():
    e = parse_fallback("€20 for a pint")
    assert e.amount == 20.0
    assert e.currency == "EUR"


def test_fallback_uber():
    e = parse_fallback("uber home 22 yesterday")
    assert e.amount == 22.0
    assert e.category == Category.TRANSPORT
    assert e.date == "yesterday"
