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
from tools import (
    create_fit_card,
    load_style_profile,
    price_compare,
    save_style_profile,
    search_listings,
    search_with_fallback,
    suggest_outfit,
    update_style_profile,
)

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


# ── search_with_fallback (Pure, No Mocking) ──────────────────────────────────

def test_fallback_no_adjustment_when_first_search_matches():
    """Verify search_with_fallback reports no adjustments when the constrained search already matches."""
    out = search_with_fallback("vintage graphic tee", size=None, max_price=50)
    assert out["results"]
    assert out["adjustments"] == []
    assert out["size"] is None and out["max_price"] == 50


def test_fallback_drops_size_when_no_size_match():
    """Verify search_with_fallback drops an over-restrictive size and reports it.

    No tee in the dataset is sized XS (they are S/M and L), so a size-XS tee
    search is empty until the size filter is dropped; the helper must then find
    tees and record the removed size filter.
    """
    out = search_with_fallback("graphic tee", size="XS", max_price=200)
    assert out["results"], "expected tees once the size filter was dropped"
    assert out["adjustments"] == ["removed the size XS filter"]
    assert out["size"] is None        # size was relaxed
    assert out["max_price"] == 200     # price was left intact


def test_fallback_lifts_price_after_size():
    """Verify search_with_fallback also lifts the price cap when dropping size alone is not enough."""
    # A graphic tee exists but the cheapest is well above $1, so a $1 cap plus an
    # impossible size forces both relaxations.
    out = search_with_fallback("graphic tee", size="XS", max_price=1)
    assert out["results"]
    assert out["adjustments"] == [
        "removed the size XS filter",
        "lifted the $1 budget cap",
    ]
    assert out["size"] is None and out["max_price"] is None


def test_fallback_genuine_no_match_returns_empty():
    """Verify search_with_fallback returns an empty result (not a crash) when nothing matches even loosened."""
    out = search_with_fallback("nonexistent unicorn garment", size="XXS", max_price=5)
    assert out["results"] == []
    # It still reports every loosening it attempted before giving up.
    assert "removed the size XXS filter" in out["adjustments"]


# ── Style Profile Memory ─────────────────────────────────────────────────────

@pytest.fixture
def temp_profile(tmp_path, monkeypatch):
    """
    Redirect the style profile store to a temp file for the duration of a test.

    Monkeypatches tools._PROFILE_PATH so load/save/update operate on an isolated
    file under pytest's tmp_path, never touching a real style_profiles.json in the
    project root. Returns the path for tests that want to inspect the raw file.

    Returns:
        pathlib.Path: The temp profile-store path now in use.
    """
    path = tmp_path / "style_profiles.json"
    monkeypatch.setattr(tools, "_PROFILE_PATH", str(path))
    return path


def test_load_profile_returns_empty_when_none_saved(temp_profile):
    """Verify load_style_profile returns a fresh empty profile (never None) before anything is saved."""
    profile = load_style_profile("default")
    assert profile == {
        "preferred_size": None,
        "max_price": None,
        "favorite_styles": [],
        "wardrobe": {"items": []},
    }


def test_save_then_load_round_trips(temp_profile):
    """Verify a saved profile is read back identically (after normalization)."""
    save_style_profile(
        {
            "preferred_size": "M",
            "max_price": 30,
            "favorite_styles": ["vintage", "grunge"],
            "wardrobe": {"items": [{"id": "w1", "name": "jeans"}]},
        },
        "default",
    )
    loaded = load_style_profile("default")
    assert loaded["preferred_size"] == "M"
    assert loaded["max_price"] == 30.0
    assert loaded["favorite_styles"] == ["vintage", "grunge"]
    assert loaded["wardrobe"]["items"][0]["name"] == "jeans"


def test_update_profile_merges_without_clobbering(temp_profile):
    """Verify update_style_profile only overwrites provided fields and leaves omitted ones intact."""
    update_style_profile(size="M", max_price=30, styles=["vintage"], profile_id="default")
    # A later search that omits the size must NOT erase the remembered "M".
    update_style_profile(max_price=45, styles=["grunge"], profile_id="default")
    profile = load_style_profile("default")
    assert profile["preferred_size"] == "M"        # preserved
    assert profile["max_price"] == 45.0            # updated
    assert profile["favorite_styles"] == ["vintage", "grunge"]  # appended, deduped


def test_update_profile_dedupes_and_caps_styles(temp_profile):
    """Verify favorite_styles never duplicates a tag and is capped at _MAX_FAVORITE_STYLES."""
    update_style_profile(styles=["vintage", "vintage"], profile_id="default")
    assert load_style_profile("default")["favorite_styles"] == ["vintage"]

    many = [f"style{i}" for i in range(tools._MAX_FAVORITE_STYLES + 5)]
    update_style_profile(styles=many, profile_id="default")
    favorites = load_style_profile("default")["favorite_styles"]
    assert len(favorites) == tools._MAX_FAVORITE_STYLES
    # Newest tags are kept at the cap.
    assert favorites[-1] == many[-1]


def test_profiles_are_isolated_by_id(temp_profile):
    """Verify saving one profile id does not clobber another stored in the same file."""
    save_style_profile({"preferred_size": "S"}, "ana")
    save_style_profile({"preferred_size": "XL"}, "ben")
    assert load_style_profile("ana")["preferred_size"] == "S"
    assert load_style_profile("ben")["preferred_size"] == "XL"


def test_corrupt_profile_file_degrades_to_empty(temp_profile):
    """Verify a corrupt store file is treated as no memory rather than raising."""
    temp_profile.write_text("{ not valid json", encoding="utf-8")
    assert load_style_profile("default") == {
        "preferred_size": None,
        "max_price": None,
        "favorite_styles": [],
        "wardrobe": {"items": []},
    }
