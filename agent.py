"""
agent.py - The FitFindr planning loop.

Orchestrates the three tools (search_listings, suggest_outfit, create_fit_card)
in response to a natural language user query, passing state between them through
a single session dict. The loop has two conditional branches: empty search
results terminate early with an error, while suggest_outfit failure also
terminates early; create_fit_card failure is treated as a partial success and
does not set the error field.

Query parsing uses regex rather than an LLM call; the fields needed (a price
number, a size token, leftover keywords) are cheap and deterministic to extract
with patterns, avoiding an extra API hop and its latency/failure surface.

Usage:
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


# ── Query Parsing ─────────────────────────────────────────────────────────────

# Standalone size tokens, longest-first so "XXS" is matched before "XS"/"S".
_SIZE_TOKENS = ["XXS", "XXL", "XS", "XL", "S", "M", "L"]

# Phrases that introduce a price ceiling (e.g. "under $30", "less than 30").
_PRICE_PREFIX = r"(?:under|below|less than|cheaper than|max(?:imum)?|up to)"

# Conversational lead-ins that carry no search signal.
_LEAD_INS = r"\b(?:i'?m\s+)?(?:looking for|searching for|search for|find me|show me|want|need|looking)\b"


def _parse_query(query: str) -> dict:
    """
    Extract a search description, size, and max_price from a free-text query.

    Uses regex rather than an LLM call: price numbers, size tokens, and leftover
    keywords are cheap and deterministic to pull with patterns. The description
    is produced by stripping out the price phrase, size phrase, and conversational
    lead-ins, leaving only the item keywords for search_listings() to score.
    Size tokens are matched longest-first so "XXS" wins over "XS"/"S".

    Args:
        query (str): The raw natural language query from the user.

    Returns:
        dict: Keys matching search_listings() parameters —
            description (str): cleaned item keywords,
            size (str | None): normalized size token (e.g. "M", "XXS"), or None,
            max_price (float | None): price ceiling extracted from the query, or None.
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
    # Item keywords for search_listings to score (it drops stopwords itself).
    desc = re.sub(rf"{_PRICE_PREFIX}\s*\$?\s*\d+(?:\.\d+)?(?:\s*dollars)?", "", text, flags=re.I)
    desc = re.sub(r"\$\s*\d+(?:\.\d+)?", "", desc)
    desc = re.sub(r"size\s+[a-z0-9]+", "", desc, flags=re.I)
    desc = re.sub(_LEAD_INS, "", desc, flags=re.I)
    desc = re.sub(r"[,]", " ", desc)
    desc = re.sub(r"\s+", " ", desc).strip()

    return {"description": desc, "size": size, "max_price": max_price}


# ── Session State ─────────────────────────────────────────────────────────────

def _new_session(query: str, wardrobe: dict) -> dict:
    """
    Initialize and return a fresh session dict for one user interaction.

    The session dict is the single source of truth for a run; it carries the
    original query, parsed parameters, every tool's output, and any error that
    caused early termination. run_agent() mutates this dict in place at each
    step rather than threading individual return values between calls.

    Args:
        query (str): The raw user query as submitted.
        wardrobe (dict): The user's wardrobe dict, passed through to suggest_outfit().

    Returns:
        dict: A zeroed-out session with keys: query, parsed, search_results,
            selected_item, wardrobe, outfit_suggestion, fit_card, error.
    """
    return {
        "query": query,              # Original user query
        "parsed": {},                # Extracted description / size / max_price
        "search_results": [],        # List of matching listing dicts
        "selected_item": None,       # Top result, passed into suggest_outfit
        "wardrobe": wardrobe,        # User's wardrobe dict
        "outfit_suggestion": None,   # String returned by suggest_outfit
        "fit_card": None,            # String returned by create_fit_card
        "error": None,               # Set if the interaction ended early
    }


# ── Planning Loop ─────────────────────────────────────────────────────────────

def run_agent(query: str, wardrobe: dict) -> dict:
    """
    Run the FitFindr planning loop for a single user interaction.

    Parses the query, searches listings, selects the top result, generates an
    outfit suggestion, and produces a fit card; each step stored in the session
    dict. Two branches terminate early: empty search results set session["error"]
    and return before suggest_outfit is called; a suggest_outfit failure also sets
    the error and stops the loop. A create_fit_card failure is partial-success
    only; session["fit_card"] is left as None but session["error"] stays None.

    Args:
        query (str): Natural language user request
            (e.g., "vintage graphic tee under $30, size M").
        wardrobe (dict): User's wardrobe dict — pass get_example_wardrobe() or
            get_empty_wardrobe() from utils/data_loader.py.

    Returns:
        dict: The completed session dict. Check session["error"] first. If it
            is not None the interaction ended early and outfit_suggestion and
            fit_card will be None. On partial success, fit_card may be None
            while outfit_suggestion is populated.
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

    # Branch on the search result; this is the key conditional in the loop.
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
    # (generic wardrobe staples fallback) and is NOT an error, so the loop continues.
    # A genuine LLM/API failure raises; we catch it, set the error, and stop
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


# ── CLI Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.data_loader import get_example_wardrobe, get_empty_wardrobe

    print("=== Happy Path: Graphic Tee ===\n")
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

    print("\n\n=== No Results Path ===\n")
    session2 = run_agent(
        query="designer ballgown size XXS under $5",
        wardrobe=get_example_wardrobe(),
    )
    print(f"Error message: {session2['error']}")
