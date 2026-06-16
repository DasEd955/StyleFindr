"""
agent.py

The FitFindr planning loop. Orchestrates the three tools in response to a
natural language user query, passing state between them via a session dict.

Complete tools.py and test each tool in isolation before implementing this file.

Usage (once implemented):
    from agent import run_agent
    from utils.data_loader import get_example_wardrobe

    result = run_agent(
        query="vintage graphic tee under $30, size M",
        wardrobe=get_example_wardrobe(),
    )
    print(result["fit_card"])
    print(result["error"])   # None on success
"""

import re

from tools import search_listings, suggest_outfit, create_fit_card


# ── query parsing ─────────────────────────────────────────────────────────────

# Standalone size tokens, longest-first so "XXS" is matched before "XS"/"S".
_SIZE_TOKENS = ["XXS", "XXL", "XS", "XL", "S", "M", "L"]

# Phrases that introduce a price ceiling (e.g. "under $30", "less than 30").
_PRICE_PREFIX = r"(?:under|below|less than|cheaper than|max(?:imum)?|up to)"

# Conversational lead-ins that carry no search signal.
_LEAD_INS = r"\b(?:i'?m\s+)?(?:looking for|searching for|search for|find me|show me|want|need|looking)\b"


def _parse_query(query: str) -> dict:
    """
    Extract a search description, size, and max_price from a free-text query.

    Choice (per planning.md): regex/string parsing rather than an LLM call —
    the fields we need (a price number, a size token, the leftover keywords) are
    cheap and deterministic to pull with patterns, so we avoid an extra API hop
    and the latency/failure surface that comes with it.

    Returns a dict: {"description": str, "size": str | None, "max_price": float | None}
    matching the search_listings() parameter names.
    """
    text = query or ""
    lower = text.lower()

    # max_price: prefer an explicit "under $30" style phrase, else any "$30".
    max_price = None
    m = re.search(rf"{_PRICE_PREFIX}\s*\$?\s*(\d+(?:\.\d+)?)", lower)
    if not m:
        m = re.search(r"\$\s*(\d+(?:\.\d+)?)", lower)
    if m:
        max_price = float(m.group(1))

    # size: prefer "size M", else a standalone size token as a whole word.
    size = None
    sm = re.search(r"size\s+([a-z0-9]+)", lower)
    if sm:
        size = sm.group(1).upper()
    else:
        for tok in _SIZE_TOKENS:
            if re.search(rf"\b{tok.lower()}\b", lower):
                size = tok
                break

    # description: strip the price phrase, size phrase, and lead-ins, leaving the
    # item keywords for search_listings to score (it drops stopwords itself).
    desc = re.sub(rf"{_PRICE_PREFIX}\s*\$?\s*\d+(?:\.\d+)?(?:\s*dollars)?", "", text, flags=re.I)
    desc = re.sub(r"\$\s*\d+(?:\.\d+)?", "", desc)
    desc = re.sub(r"size\s+[a-z0-9]+", "", desc, flags=re.I)
    desc = re.sub(_LEAD_INS, "", desc, flags=re.I)
    desc = re.sub(r"[,]", " ", desc)
    desc = re.sub(r"\s+", " ", desc).strip()

    return {"description": desc, "size": size, "max_price": max_price}


# ── session state ─────────────────────────────────────────────────────────────

def _new_session(query: str, wardrobe: dict) -> dict:
    """
    Initialize and return a fresh session dict for one user interaction.

    The session dict is the single source of truth for everything that happens
    during a run — it stores the original query, parsed parameters, tool results,
    and any error that caused early termination.

    You may add fields to this dict as needed for your implementation.
    """
    return {
        "query": query,              # original user query
        "parsed": {},                # extracted description / size / max_price
        "search_results": [],        # list of matching listing dicts
        "selected_item": None,       # top result, passed into suggest_outfit
        "wardrobe": wardrobe,        # user's wardrobe dict
        "outfit_suggestion": None,   # string returned by suggest_outfit
        "fit_card": None,            # string returned by create_fit_card
        "error": None,               # set if the interaction ended early
    }


# ── planning loop ─────────────────────────────────────────────────────────────

def run_agent(query: str, wardrobe: dict) -> dict:
    """
    Main agent entry point. Runs the FitFindr planning loop for a single
    user interaction and returns the completed session dict.

    Args:
        query:    Natural language user request
                  (e.g., "vintage graphic tee under $30, size M")
        wardrobe: User's wardrobe dict — use get_example_wardrobe() or
                  get_empty_wardrobe() from utils/data_loader.py

    Returns:
        The session dict after the interaction completes. Check session["error"]
        first — if it is not None, the interaction ended early and the other
        output fields (outfit_suggestion, fit_card) will be None.

    TODO — implement this function using the planning loop you designed in planning.md:

        Step 1: Initialize the session with _new_session().

        Step 2: Parse the user's query to extract a description, size, and
                max_price. You can use regex, string splitting, or ask the LLM
                to parse it — document your choice in planning.md.
                Store the result in session["parsed"].

        Step 3: Call search_listings() with the parsed parameters.
                Store results in session["search_results"].
                If no results: set session["error"] to a helpful message and
                return the session early. Do NOT proceed to suggest_outfit
                with empty input.

        Step 4: Select the item to use (e.g., the top result).
                Store it in session["selected_item"].

        Step 5: Call suggest_outfit() with the selected item and wardrobe.
                Store the result in session["outfit_suggestion"].

        Step 6: Call create_fit_card() with the outfit suggestion and selected item.
                Store the result in session["fit_card"].

        Step 7: Return the session.

    Before writing code, complete the Planning Loop and State Management sections
    of planning.md — your implementation should match what you described there.
    """
    # Step 1 — initialize the single source of truth for this interaction.
    session = _new_session(query, wardrobe)

    # Step 2 — parse the query into search parameters (regex, see _parse_query).
    session["parsed"] = _parse_query(query)
    parsed = session["parsed"]

    # Step 3 — search. An unexpected exception (e.g. corrupt data file) is
    # distinct from "no matches": the former stops with a generic error, the
    # latter stops with a helpful "broaden your search" message.
    try:
        results = search_listings(
            description=parsed["description"],
            size=parsed["size"],
            max_price=parsed["max_price"],
        )
    except Exception:
        session["error"] = "Search failed due to an unexpected error. Please try again."
        return session

    session["search_results"] = results

    # Branch on the search result — this is the key conditional in the loop.
    # An empty list ends the interaction; we do NOT proceed to suggest_outfit.
    if not results:
        size_txt = parsed["size"] or "any size"
        price_txt = (
            f"${parsed['max_price']:.0f}"
            if parsed["max_price"] is not None
            else "your budget"
        )
        session["error"] = (
            f"No listings matched '{parsed['description']}' in size {size_txt} "
            f"under {price_txt}. Try a broader description, a higher budget, or "
            f"a different size."
        )
        return session

    # Step 4 — select the top-ranked listing as the item to style.
    session["selected_item"] = results[0]

    # Step 5 — suggest an outfit. An empty wardrobe is handled inside the tool
    # (generic-staples fallback) and is NOT an error, so the loop continues.
    # A genuine LLM/API failure raises; we catch it, set the error, and stop —
    # create_fit_card is never called with empty input.
    try:
        outfit = suggest_outfit(session["selected_item"], wardrobe)
    except Exception:
        session["error"] = "Outfit generation failed. Please try again."
        return session

    if not outfit or not outfit.get("outfit_description"):
        session["error"] = "Outfit generation failed. Please try again."
        return session

    session["outfit_suggestion"] = outfit

    # Step 6 — create the fit card. This is partial-failure tolerant: if the
    # LLM call fails, we keep the outfit suggestion and simply leave fit_card
    # as None (a degraded success), rather than erroring out the whole run.
    try:
        session["fit_card"] = create_fit_card(outfit, session["selected_item"])
    except Exception:
        session["fit_card"] = None

    # Step 7 — return the completed session.
    return session


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.data_loader import get_example_wardrobe, get_empty_wardrobe

    print("=== Happy path: graphic tee ===\n")
    session = run_agent(
        query="looking for a vintage graphic tee under $30",
        wardrobe=get_example_wardrobe(),
    )
    if session["error"]:
        print(f"Error: {session['error']}")
    else:
        print(f"Found: {session['selected_item']['title']}")
        print(f"\nOutfit: {session['outfit_suggestion']}")
        print(f"\nFit card: {session['fit_card']}")

    print("\n\n=== No-results path ===\n")
    session2 = run_agent(
        query="designer ballgown size XXS under $5",
        wardrobe=get_example_wardrobe(),
    )
    print(f"Error message: {session2['error']}")
