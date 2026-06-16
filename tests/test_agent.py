"""
Tests for the FitFindr planning loop (agent.py) and the Gradio query handler
(app.py).

Run from the project root:

    pytest tests/

These cover the Milestone 4 behaviours: query parsing, state passing between
tools, and the conditional branches in the planning loop. The two LLM-backed
tools call Groq, so we monkeypatch `tools._chat` — that keeps the suite fast,
deterministic, and runnable without a GROQ_API_KEY. `agent` imports the tool
functions into its own namespace, so failure simulations patch `agent.<tool>`
directly.
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
    """Replace tools._chat with canned JSON so suggest_outfit/create_fit_card
    return valid contract dicts without hitting the network."""

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


# ── query parsing ────────────────────────────────────────────────────────────

def test_parse_query_extracts_all_fields():
    parsed = _parse_query("vintage cashmere sweater under $200, size L")
    assert parsed["description"] == "vintage cashmere sweater"
    assert parsed["size"] == "L"
    assert parsed["max_price"] == 200.0


def test_parse_query_no_size():
    parsed = _parse_query("looking for a vintage graphic tee under $30")
    assert parsed["size"] is None
    assert parsed["max_price"] == 30.0
    assert "vintage" in parsed["description"].lower()


def test_parse_query_multichar_size_token():
    # "XXS" must win over "XS"/"S" — longest-token-first matching.
    parsed = _parse_query("designer ballgown size XXS under $5")
    assert parsed["size"] == "XXS"
    assert parsed["max_price"] == 5.0


# ── planning loop: happy path + state passing ────────────────────────────────

def test_full_success_path(fake_llm):
    session = run_agent("vintage graphic tee under $30", EXAMPLE_WARDROBE)
    assert session["error"] is None
    assert session["selected_item"] is not None
    assert session["outfit_suggestion"]["outfit_description"]
    assert session["fit_card"]["fit_card_text"]


def test_state_passes_by_identity(monkeypatch, fake_llm):
    # The EXACT objects must flow between steps — no re-prompting or rebuilding.
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


# ── planning loop: conditional branches ──────────────────────────────────────

def test_empty_results_stops_loop(monkeypatch):
    # No matches → error set, fit_card stays None, suggest_outfit NEVER called.
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
    # Empty wardrobe is a fallback, NOT an error: the loop runs end to end.
    session = run_agent("vintage graphic tee under $30", EMPTY_WARDROBE)
    assert session["error"] is None
    assert session["outfit_suggestion"]["outfit_description"]
    assert session["fit_card"]["fit_card_text"]


def test_suggest_outfit_failure_stops_loop(monkeypatch):
    # suggest_outfit failure → error set, loop stops, create_fit_card NOT called.
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
    assert session["selected_item"] is not None  # state up to step 4 is kept
    assert session["outfit_suggestion"] is None
    assert session["fit_card"] is None
    assert called["fit_card"] is False


def test_fit_card_failure_is_partial_success(monkeypatch, fake_llm):
    # create_fit_card failure must NOT error the run — outfit is still kept.
    def boom(*a, **k):
        raise RuntimeError("simulated caption error")

    monkeypatch.setattr(agent, "create_fit_card", boom)

    session = run_agent("vintage graphic tee under $30", EXAMPLE_WARDROBE)

    assert session["error"] is None
    assert session["outfit_suggestion"]["outfit_description"]
    assert session["fit_card"] is None


# ── app.handle_query mapping ─────────────────────────────────────────────────

def test_handle_query_empty_input():
    listing, outfit, fit_card = app.handle_query("   ", "Example wardrobe")
    assert "enter" in listing.lower()
    assert outfit == "" and fit_card == ""


def test_handle_query_success_maps_three_panels(fake_llm):
    listing, outfit, fit_card = app.handle_query(
        "vintage graphic tee under $30", "Example wardrobe"
    )
    assert listing and outfit and fit_card
    # Listing panel surfaces the item facts; fit-card panel surfaces the caption.
    assert "$" in listing
    assert "thrifted" in fit_card.lower()


def test_handle_query_error_goes_to_first_panel():
    listing, outfit, fit_card = app.handle_query(
        "designer ballgown size XXS under $5", "Example wardrobe"
    )
    assert "no listings matched" in listing.lower()
    assert outfit == "" and fit_card == ""
