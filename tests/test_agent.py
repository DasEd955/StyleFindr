"""
test_agent.py - Tests for the FitFindr planning loop (agent.py) and the Gradio
query handler (app.py).

Covers query parsing, object-identity state passing between tools, and the four
conditional branches in run_agent(): full success, empty search results, outfit
failure, and partial success when create_fit_card fails. Also covers the three
app.handle_query() paths: empty input, success mapping to all three panels, and
error surfacing in the first panel.

The two LLM-backed tools call Groq, so tests monkeypatch tools._chat with canned
JSON, keeping the suite fast, deterministic, and runnable without a GROQ_API_KEY.
agent imports the tool functions into its own namespace, so failure simulations
patch agent.<tool> directly rather than tools.<tool>.

Run from the project root:

    pytest tests/
"""

import json

import pytest

import agent
import app
import tools
from agent import _parse_query, run_agent


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


@pytest.fixture
def fake_llm(monkeypatch):
    """
    Replace tools._chat with a canned-JSON stub for the duration of a test.

    The stub returns a single JSON blob that satisfies both the suggest_outfit
    and create_fit_card contract shapes, so both tools return valid dicts without
    making any network calls. Patch target is tools._chat (not agent._chat) because
    suggest_outfit and create_fit_card call it directly through the tools module.
    """

    def _fake(messages, temperature, json_mode=False):
        return json.dumps(
            {
                "outfit_description": "Pair the tee with the baggy jeans.",
                "matching_items": ["Baggy straight-leg jeans, dark wash"],
                "style_reasoning": "Tonal grunge layering.",
                "style_category": "grunge",
                "fit_card_text": "thrifted this for $24 off depop 🖤",
                "style_tags": ["vintage", "grunge"],
                "caption_tone": "casual",
            }
        )

    monkeypatch.setattr(tools, "_chat", _fake)


# ── Query Parsing ────────────────────────────────────────────────────────────

def test_parse_query_extracts_all_fields():
    """Verify _parse_query correctly extracts description, size, and max_price from a full query."""
    parsed = _parse_query("vintage cashmere sweater under $200, size L")
    assert parsed["description"] == "vintage cashmere sweater"
    assert parsed["size"] == "L"
    assert parsed["max_price"] == 200.0


def test_parse_query_no_size():
    """Verify _parse_query returns size=None and still extracts max_price and description keywords when no size is present."""
    parsed = _parse_query("looking for a vintage graphic tee under $30")
    assert parsed["size"] is None
    assert parsed["max_price"] == 30.0
    assert "vintage" in parsed["description"].lower()


def test_parse_query_multichar_size_token():
    """Verify _parse_query matches "XXS" before "XS" or "S" due to longest-token-first ordering."""
    parsed = _parse_query("designer ballgown size XXS under $5")
    assert parsed["size"] == "XXS"
    assert parsed["max_price"] == 5.0


# ── Planning Loop: Happy Path + State Passing ────────────────────────────────

def test_full_success_path(fake_llm):
    """Verify run_agent completes all steps with no error and populates selected_item, outfit_suggestion, and fit_card."""
    session = run_agent("vintage graphic tee under $30", EXAMPLE_WARDROBE)
    assert session["error"] is None
    assert session["selected_item"] is not None
    assert session["outfit_suggestion"]["outfit_description"]
    assert session["fit_card"]["fit_card_text"]


def test_state_passes_by_identity(monkeypatch, fake_llm):
    """Verify run_agent passes the exact same objects between steps rather than rebuilding them."""
    captured = {}
    orig_suggest, orig_fc = agent.suggest_outfit, agent.create_fit_card

    def spy_suggest(new_item, wardrobe):
        captured["suggest_new_item"] = new_item
        return orig_suggest(new_item, wardrobe)

    def spy_fc(outfit, new_item):
        captured["fc_outfit"] = outfit
        captured["fc_new_item"] = new_item
        return orig_fc(outfit, new_item)

    monkeypatch.setattr(agent, "suggest_outfit", spy_suggest)
    monkeypatch.setattr(agent, "create_fit_card", spy_fc)

    session = run_agent("vintage graphic tee under $30", EXAMPLE_WARDROBE)

    assert captured["suggest_new_item"] is session["selected_item"]
    assert captured["fc_outfit"] is session["outfit_suggestion"]
    assert captured["fc_new_item"] is session["selected_item"]


# ── Planning Loop: Conditional Branches ──────────────────────────────────────

def test_empty_results_stops_loop(monkeypatch):
    """Verify that empty search results set session["error"] and prevent suggest_outfit from being called."""
    called = {"suggest": False}

    def must_not_run(*a, **k):
        called["suggest"] = True
        raise AssertionError("suggest_outfit called on empty search results")

    monkeypatch.setattr(agent, "suggest_outfit", must_not_run)

    session = run_agent("designer ballgown size XXS under $5", EXAMPLE_WARDROBE)

    assert session["error"] is not None
    assert session["search_results"] == []
    assert session["outfit_suggestion"] is None
    assert session["fit_card"] is None
    assert called["suggest"] is False


def test_empty_wardrobe_continues_to_fit_card(fake_llm):
    """Verify that an empty wardrobe triggers the generic wardrobe staples fallback and the loop still completes."""
    session = run_agent("vintage graphic tee under $30", EMPTY_WARDROBE)
    assert session["error"] is None
    assert session["outfit_suggestion"]["outfit_description"]
    assert session["fit_card"]["fit_card_text"]


def test_suggest_outfit_failure_stops_loop(monkeypatch):
    """Verify that a suggest_outfit exception sets session["error"] and prevents create_fit_card from being called."""
    called = {"fit_card": False}

    def boom(*a, **k):
        raise RuntimeError("simulated API error")

    def must_not_run(*a, **k):
        called["fit_card"] = True
        raise AssertionError("create_fit_card called after outfit failure")

    monkeypatch.setattr(agent, "suggest_outfit", boom)
    monkeypatch.setattr(agent, "create_fit_card", must_not_run)

    session = run_agent("vintage graphic tee under $30", EXAMPLE_WARDROBE)

    assert session["error"] == "Outfit generation failed. Please try again."
    assert session["selected_item"] is not None  # State up to step 4 is kept
    assert session["outfit_suggestion"] is None
    assert session["fit_card"] is None
    assert called["fit_card"] is False


def test_fit_card_failure_is_partial_success(monkeypatch, fake_llm):
    """Verify that a create_fit_card exception leaves session["error"] as None and preserves outfit_suggestion."""
    def boom(*a, **k):
        raise RuntimeError("simulated caption error")

    monkeypatch.setattr(agent, "create_fit_card", boom)

    session = run_agent("vintage graphic tee under $30", EXAMPLE_WARDROBE)

    assert session["error"] is None
    assert session["outfit_suggestion"]["outfit_description"]
    assert session["fit_card"] is None


# ── app.handle_query Mapping ─────────────────────────────────────────────────

def test_handle_query_empty_input():
    """Verify app.handle_query returns an error prompt in the first panel and empty strings for the other two on blank input."""
    listing, outfit, fit_card = app.handle_query("   ", "Example wardrobe")
    assert "enter" in listing.lower()
    assert outfit == "" and fit_card == ""


def test_handle_query_success_maps_three_panels(fake_llm):
    """Verify app.handle_query populates all three output panels on a successful run."""
    listing, outfit, fit_card = app.handle_query(
        "vintage graphic tee under $30", "Example wardrobe"
    )
    assert listing and outfit and fit_card
    # Listing panel surfaces the item facts; fit-card panel surfaces the caption.
    assert "$" in listing
    assert "thrifted" in fit_card.lower()


def test_handle_query_error_goes_to_first_panel():
    """Verify app.handle_query surfaces session["error"] in the first panel and leaves the other two empty."""
    listing, outfit, fit_card = app.handle_query(
        "designer ballgown size XXS under $5", "Example wardrobe"
    )
    assert "no listings matched" in listing.lower()
    assert outfit == "" and fit_card == ""
