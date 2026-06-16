"""
test_tools.py - Isolation tests for the four FitFindr tools.

search_listings() and price_compare() are pure (no network) and are tested
directly against the real dataset. suggest_outfit() and create_fit_card() call
the Groq LLM, so tests monkeypatch tools._chat with a canned-JSON recorder. This
keeps the suite fast, deterministic, and runnable without a GROQ_API_KEY.

Coverage includes at least one test per documented failure mode: empty search
results, empty wardrobe, apparel-size cross-system pass-through, whole-word
keyword matching, incomplete outfit input, None outfit, malformed JSON from the
model, and price_compare's insufficient-data / verdict-band paths.

Run from the project root:

    pytest tests/
"""

import json
import pytest
import tools
from tools import create_fit_card, price_compare, search_listings, suggest_outfit

# A minimal listing used as `new_item` in the LLM-tool tests.
SAMPLE_ITEM = {
    "id": "lst_test",
    "title": "Vintage Band Tee — Faded Grey",
    "price": 19.0,
    "platform": "depop",
    "style_tags": ["vintage", "grunge"],
    "colors": ["grey"],
    "category": "tops",
}

EXAMPLE_WARDROBE = {
    "items": [
        {
            "id": "w_001",
            "name": "Baggy straight-leg jeans, dark wash",
            "category": "bottoms",
            "colors": ["dark blue"],
            "style_tags": ["denim", "baggy"],
            "notes": None,
        }
    ]
}
EMPTY_WARDROBE = {"items": []}


# ── search_listings (Pure, No Mocking) ──────────────────────────────────────

def test_search_returns_results():
    """Verify search_listings returns a non-empty list of dicts with title and price fields for a broad query."""
    results = search_listings("vintage graphic tee", size=None, max_price=50)
    assert isinstance(results, list)
    assert len(results) > 0
    # Every result is a dict carrying the expected listing fields.
    assert all("title" in item and "price" in item for item in results)


def test_search_empty_results():
    """Verify search_listings returns an empty list (not an exception) when no listings match."""
    results = search_listings("designer ballgown", size="XXS", max_price=5)
    assert results == []


def test_search_price_filter():
    """Verify search_listings excludes every result with a price above max_price."""
    results = search_listings("jacket", size=None, max_price=10)
    assert all(item["price"] <= 10 for item in results)


def test_search_size_filter():
    """Verify apparel size filtering only excludes alpha-sized items that don't include the requested size token.

    Listings with numeric sizes (US 8, W29) or "One Size" are not comparable to
    an apparel size and must pass through; relevance ranking decides those.
    """
    import re

    _ALPHA = {"XS", "S", "M", "L", "XL", "XXL", "XXS", "XXXL"}
    results = search_listings("tee", size="M", max_price=200)
    for item in results:
        size = item["size"].upper()
        tokens = set(re.split(r"[^A-Z0-9]+", size)) - {""}
        alpha_tokens = tokens & _ALPHA
        if alpha_tokens:                      # apparel-sized → must include M
            assert "M" in alpha_tokens, f"{item['title']} ({size}) leaked into size M"
        # Else: numeric / One Size → not comparable, allowed through.


def test_search_whole_word_match_no_substring_leak():
    """Regression: conversational filler words must not score via substring overlap.

    "we" (from "...we could keep...") formerly matched "western" style tags via
    substring, surfacing an unrelated belt as the top result for a boots query.
    """
    results = search_listings(
        "new pair of combat boots we could keep cheap", size=None, max_price=200
    )
    titles = [r["title"].lower() for r in results]
    assert not any("belt" in t for t in titles), (
        "noise word matched a western-tagged belt via substring overlap"
    )


def test_search_apparel_size_does_not_exclude_shoes():
    """Regression: an apparel-size request must not exclude numeric sized shoes since the two systems are not comparable."""
    boots = search_listings("boots", size="Medium", max_price=200)
    assert any("boots" in r["title"].lower() for r in boots), (
        "apparel-size request excluded numeric-sized shoes"
    )


# ── suggest_outfit (LLM mocked) ─────────────────────────────────────────────

@pytest.fixture
def fake_llm(monkeypatch):
    """
    Replace tools._chat with a canned-JSON recorder for the duration of a test.

    The stub appends each call's messages and temperature to a `calls` list,
    which is returned so tests can inspect the prompt that was actually sent
    (e.g. asserting the empty-wardrobe branch was taken). The canned response
    satisfies both the suggest_outfit and create_fit_card contract shapes.

    Returns:
        list[dict]: The calls list, updated in place as the stub is invoked.
    """
    calls = []

    def _fake(messages, temperature, json_mode=False):
        calls.append({"messages": messages, "temperature": temperature})
        return json.dumps(
            {
                "outfit_description": "Pair the tee with the baggy jeans.",
                "matching_items": ["Baggy straight-leg jeans, dark wash"],
                "style_reasoning": "Tonal grunge layering.",
                "style_category": "grunge",
                "fit_card_text": "thrifted this band tee off depop for $19 🖤",
                "style_tags": ["vintage", "grunge"],
                "caption_tone": "casual",
            }
        )

    monkeypatch.setattr(tools, "_chat", _fake)
    return calls


def test_suggest_outfit_returns_contract_dict(fake_llm):
    """Verify suggest_outfit returns a dict with all four contract keys and a non-empty outfit_description."""
    result = suggest_outfit(SAMPLE_ITEM, EXAMPLE_WARDROBE)
    assert set(result.keys()) == {
        "outfit_description",
        "matching_items",
        "style_reasoning",
        "style_category",
    }
    assert isinstance(result["matching_items"], list)
    assert result["outfit_description"]


def test_suggest_outfit_empty_wardrobe(fake_llm):
    """Verify suggest_outfit handles an empty wardrobe without crashing and uses the generic-staples fallback branch."""
    result = suggest_outfit(SAMPLE_ITEM, EMPTY_WARDROBE)
    assert isinstance(result, dict)
    assert result["outfit_description"]
    sent_prompt = fake_llm[-1]["messages"][-1]["content"].lower()
    assert "have not entered any wardrobe" in sent_prompt or "no wardrobe" in sent_prompt


def test_suggest_outfit_malformed_json(monkeypatch):
    """Verify suggest_outfit wraps non-JSON model output in the contract dict rather than raising."""
    monkeypatch.setattr(tools, "_chat", lambda *a, **k: "not valid json")
    result = suggest_outfit(SAMPLE_ITEM, EXAMPLE_WARDROBE)
    assert result["outfit_description"] == "not valid json"
    assert result["matching_items"] == []


# ── create_fit_card (LLM Mocked) ────────────────────────────────────────────

def test_fit_card_returns_contract_dict(fake_llm):
    """Verify create_fit_card returns a dict with all three contract keys and 0–4 style tags."""
    outfit = suggest_outfit(SAMPLE_ITEM, EXAMPLE_WARDROBE)
    card = create_fit_card(outfit, SAMPLE_ITEM)
    assert set(card.keys()) == {"fit_card_text", "style_tags", "caption_tone"}
    assert card["fit_card_text"]
    assert 0 <= len(card["style_tags"]) <= 4


def test_fit_card_missing_outfit_description(fake_llm):
    """Verify create_fit_card falls back to an item-only caption when outfit_description is absent."""
    card = create_fit_card({}, SAMPLE_ITEM)
    assert isinstance(card, dict)
    assert card["fit_card_text"]
    sent_prompt = fake_llm[-1]["messages"][-1]["content"].lower()
    assert "no styling context" in sent_prompt


def test_fit_card_none_outfit(fake_llm):
    """Verify create_fit_card handles a None outfit argument without raising."""
    card = create_fit_card(None, SAMPLE_ITEM)
    assert isinstance(card, dict)
    assert card["fit_card_text"]


def test_fit_card_malformed_json(monkeypatch):
    """Verify create_fit_card wraps garbage model output as fit_card_text rather than raising."""
    monkeypatch.setattr(tools, "_chat", lambda *a, **k: "<<garbage>>")
    card = create_fit_card({"outfit_description": "x"}, SAMPLE_ITEM)
    assert card["fit_card_text"] == "<<garbage>>"
    assert card["style_tags"] == []


# ── price_compare (Pure, No Mocking) ─────────────────────────────────────────

def test_price_compare_returns_contract_dict():
    """Verify price_compare returns a dict with every contract key for a real dataset item."""
    item = search_listings("vintage graphic tee", size=None, max_price=200)[0]
    result = price_compare(item)
    assert set(result.keys()) == {
        "verdict",
        "item_price",
        "comparable_count",
        "comparable_avg",
        "comparable_median",
        "comparable_range",
        "explanation",
    }
    assert result["verdict"] in {"underpriced", "fair", "overpriced", "insufficient_data"}


def test_price_compare_missing_price():
    """Verify a non-numeric price yields an insufficient_data verdict rather than raising."""
    result = price_compare({"id": "x", "category": "tops", "price": None, "style_tags": []})
    assert result["verdict"] == "insufficient_data"
    assert result["item_price"] is None
    assert result["comparable_avg"] is None


def test_price_compare_unknown_category_insufficient_data():
    """Verify an item in a category with no peers reports insufficient_data, not a crash."""
    result = price_compare(
        {"id": "x", "category": "nonexistent-category", "price": 50.0, "style_tags": []}
    )
    assert result["verdict"] == "insufficient_data"
    assert result["comparable_count"] == 0


def test_price_compare_overpriced_verdict():
    """Verify an item priced far above its category comparables is flagged overpriced.

    Uses a real "tops" item id (so it is excluded from its own comparables) but an
    inflated price, which must land above the 1.15x median band.
    """
    base = search_listings("tee", size=None, max_price=200)[0]
    inflated = {**base, "price": 999.0}
    result = price_compare(inflated)
    assert result["verdict"] == "overpriced"
    assert result["comparable_count"] >= 2
    assert result["item_price"] == 999.0


def test_price_compare_underpriced_verdict():
    """Verify an item priced far below its category comparables is flagged underpriced."""
    base = search_listings("tee", size=None, max_price=200)[0]
    cheap = {**base, "price": 1.0}
    result = price_compare(cheap)
    assert result["verdict"] == "underpriced"
    assert result["comparable_count"] >= 2


def test_price_compare_excludes_self():
    """Verify the item being priced is never counted among its own comparables (excluded by id)."""
    listings = tools.load_listings()
    # Pick a category with several members so a comparable set exists.
    item = next(l for l in listings if l.get("category") == "tops")
    result = price_compare(item)
    category_peers = sum(
        1 for l in listings if l.get("category") == "tops" and l.get("id") != item["id"]
    )
    # comparable_count can be <= peers (tag filtering narrows it) but never counts self.
    assert result["comparable_count"] <= category_peers
