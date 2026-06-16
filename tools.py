"""
tools.py - The four core FitFindr tools and their private helpers.

Each public tool is a standalone function that can be called and tested
independently before being wired into the agent loop in agent.py.

search_listings() and price_compare() are pure (no network): they load the
dataset and reason over it deterministically — search_listings ranks by
keyword-overlap score, price_compare judges an item's price against comparable
listings. suggest_outfit() and create_fit_card() call the Groq LLM via the thin
_chat() wrapper; both return validated contract dicts and handle malformed JSON
defensively rather than raising. Private helpers (_tokens, _size_matches,
_relevance_score, _find_comparables, etc.) are module-internal and not part of
the public API.

search_with_fallback() wraps search_listings(), retrying with progressively
looser filters when nothing matches and reporting what it adjusted. The
load_style_profile / save_style_profile / update_style_profile helpers persist a
user's style preferences (size, budget, favorite styles, wardrobe) to a small
JSON file so the agent can remember them across sessions.

Tools:
    search_listings(description, size, max_price)      → list[dict]
    search_with_fallback(description, size, max_price) → dict
    suggest_outfit(new_item, wardrobe)                 → dict
    create_fit_card(outfit, new_item)                  → dict
    price_compare(item)                                → dict

Style profile memory:
    load_style_profile(profile_id)                  → dict
    save_style_profile(profile, profile_id)         → dict
    update_style_profile(size, max_price, styles, wardrobe, profile_id) → dict
"""

import json
import os
import re
import statistics
from dotenv import load_dotenv
from groq import Groq
from utils.data_loader import load_listings

load_dotenv()

# Free-tier model shared across the LLM-backed tools.
_MODEL = "llama-3.3-70b-versatile"


# ── Groq Client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """
    Initialize and return a Groq client authenticated with GROQ_API_KEY.

    Reads the key from the environment (populated by load_dotenv() at module
    level). Raises ValueError immediately if the key is absent so callers
    receive a clear message rather than a cryptic auth failure from the SDK.

    Returns:
        Groq: An authenticated Groq client instance.

    Raises:
        ValueError: If GROQ_API_KEY is not set in the environment.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


def _chat(messages: list[dict], temperature: float, json_mode: bool = False) -> str:
    """
    Send a chat completion request to Groq and return the response text.

    Thin wrapper that keeps the LLM call in one place so tests can monkeypatch
    it without touching each tool. Raises on API or network failure (it does
    not swallow exceptions) so the agent layer can catch and surface the error.

    Args:
        messages (list[dict]): OpenAI-format message dicts (role/content).
        temperature (float): Sampling temperature passed to the model.
        json_mode (bool): If True, sets response_format to json_object so the
            model is constrained to emit valid JSON.

    Returns:
        str: The raw text content of the model's first completion choice.

    Raises:
        groq.APIError: On any API or network failure.
    """
    client = _get_groq_client()
    kwargs = {"model": _MODEL, "messages": messages, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ── Tool 1 Helpers ──────────────────────────────────────────────────────────

# Common words that carry no search signal: dropped from query keywords so they
# don't inflate relevance scores (e.g. "a vintage tee for me" → ["vintage", "tee"]).
# Includes conversational filler ("looking for a new pair under my budget, thanks!")
# so polite, full sentence queries don't leak noise words into the relevance score.
_STOPWORDS = {
    # articles, prepositions, conjunctions
    "a", "an", "the", "for", "with", "in", "of", "and", "or", "to", "on", "at",
    "by", "from", "as", "is", "are", "be", "than", "then", "so", "but",
    # pronouns / contractions
    "my", "i", "im", "me", "we", "us", "you", "your", "it", "its",
    # search / request filler
    "some", "looking", "look", "want", "need", "find", "show", "get", "size",
    "new", "pair", "most", "prefer", "could", "would", "should", "keep",
    "budget", "less", "more", "possible", "thank", "thanks", "if", "please",
    "just", "really", "also", "like", "can", "this", "that", "these", "those",
    "maybe", "around", "about", "under", "over",
}


def _tokens(text: str) -> set[str]:
    """
    Tokenize a string into a set of lowercase word tokens for whole word matching.

    Splitting on non-alphanumeric boundaries and returning a set ensures that
    keyword comparisons in _relevance_score() never match substrings; e.g., "we"
    cannot match "western" because the tokens are compared with `in`, not `find`.

    Args:
        text (str): Any string; None-safe (treats None as empty).

    Returns:
        set[str]: Lower-case alphanumeric tokens, empty strings excluded.
    """
    return {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t}


def _extract_keywords(description: str) -> list[str]:
    """
    Extract meaningful search keywords from a description string.

    Lowercases, strips punctuation, and drops tokens that are in the _STOPWORDS
    set or shorter than two characters. The result is passed to _relevance_score()
    to compute per-listing keyword overlap.

    Args:
        description (str): The cleaned search description from _parse_query().

    Returns:
        list[str]: Lower-case alphanumeric tokens with stopwords removed.
    """
    if not description:
        return []
    tokens = re.split(r"[^a-z0-9]+", description.lower())
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]


# Alpha clothing sizes, in one set so we can tell an apparel size ("M") apart from
# a shoe/waist size ("8", "W30"). Word forms are normalized to these letters.
_ALPHA_SIZES = {"XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL"}
_SIZE_WORDS = {
    "XSMALL": "XS", "SMALL": "S", "MEDIUM": "M", "LARGE": "L",
    "XLARGE": "XL", "EXTRALARGE": "XL",
}


def _normalize_size_token(token: str) -> str:
    """
    Map a word-form size to its canonical letter abbreviation.

    Converts tokens like "MEDIUM" or "LARGE" to their single/two-letter forms
    ("M", "L") so they can be compared against the _ALPHA_SIZES set. Tokens
    that are not in _SIZE_WORDS (e.g. "XL", "8") are returned unchanged.

    Args:
        token (str): An uppercase size token (e.g. "MEDIUM", "XL", "8.5").

    Returns:
        str: The canonical form if recognized, otherwise token unchanged.
    """
    return _SIZE_WORDS.get(token, token)


def _size_matches(requested: str, listing_size: str) -> bool:
    """
    Return True if a listing's size satisfies the requested size.

    The dataset uses two incompatible sizing systems: alpha apparel sizes
    ("S/M", "XL (oversized)", "M/L") and numeric shoe/waist sizes ("US 7",
    "W30 L30"). A request only filters within its own system; e.g., asking for
    "Medium" (apparel) must NOT exclude shoes sized "US 8" because the two
    systems are not comparable. This cross-system pass-through prevents a
    clothing size request from silently wiping out the shoe category.

    Args:
        requested (str): The size token from the parsed query (e.g. "M", "8").
        listing_size (str): The size field of a listing (e.g. "S/M", "US 7").

    Returns:
        bool: True when the listing satisfies the request or when the systems
            are not comparable; False only when both sizes are in the same system
            and the listing does not match.
    """
    if not requested:
        return True

    req = _normalize_size_token(requested.strip().upper())
    listing = (listing_size or "").upper()

    # Universal fit items satisfy every request.
    if "ONE SIZE" in listing:
        return True

    # `listing` is already upper-cased; split directly (do NOT use _tokens, which
    # lower-cases and would never match the upper-case size sets).
    raw_tokens = [t for t in re.split(r"[^A-Z0-9]+", listing) if t]
    listing_tokens = {_normalize_size_token(t) for t in raw_tokens}
    listing_alpha = listing_tokens & _ALPHA_SIZES
    listing_numbers = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", listing)}

    if req in _ALPHA_SIZES:
        # Apparel request: filter only against apparel listings.
        if listing_alpha:
            return req in listing_alpha
        return True  # Listing has no apparel size (e.g. shoes) → not comparable
    if re.fullmatch(r"\d+(?:\.\d+)?", req):
        # Numeric request: filter only against numeric listings.
        if listing_numbers:
            return float(req) in listing_numbers
        return True  # Listing has no numeric size → not comparable
    # Unrecognized request format → don't exclude on size.
    return True


def _relevance_score(listing: dict, keywords: list[str]) -> int:
    """
    Compute a weighted keyword overlap score for a single listing.

    Matching is whole word (token-based via _tokens()), not substring, so
    conversational filler like "we" cannot score against "western" tags.
    Weights: style_tags (+3), title/colors/category/brand (+2 each), description (+1).

    Args:
        listing (dict): A listing dict from the dataset.
        keywords (list[str]): Lower-case keyword tokens from _extract_keywords().

    Returns:
        int: Total relevance score; 0 means no keyword overlap.
    """
    title_tokens = _tokens(listing.get("title", ""))
    desc_tokens = _tokens(listing.get("description", ""))
    category = listing.get("category", "").lower()
    brand_tokens = _tokens(listing.get("brand") or "")
    tag_tokens = _tokens(" ".join(listing.get("style_tags", [])))
    color_tokens = _tokens(" ".join(listing.get("colors", [])))

    score = 0
    for kw in keywords:
        if kw in tag_tokens:
            score += 3
        if kw in title_tokens:
            score += 2
        if kw in color_tokens:
            score += 2
        if kw == category:
            score += 2
        if kw in brand_tokens:
            score += 2
        if kw in desc_tokens:
            score += 1
    return score


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching a description, with
    optional size and price ceiling filters.

    Applies hard filters (price, size) first, then scores each surviving listing
    by keyword overlap using _relevance_score(). Listings with a score of zero are
    dropped. When no keywords can be extracted from the description (e.g. blank
    input), falls back to returning all price/size-filtered items sorted by price.
    An empty result is a normal outcome, the function never raises on no matches.

    Args:
        description (str): Keywords describing the item (e.g. "vintage graphic tee").
        size (str | None): Size token to filter by (e.g. "M", "8"), or None to skip.
            Filtering is cross-system-safe: apparel sizes do not exclude numeric-sized
            items such as shoes.
        max_price (float | None): Maximum price inclusive, or None to skip.

    Returns:
        list[dict]: Matching listing dicts sorted by relevance descending, price
            ascending as a tiebreaker. Empty list if nothing matches.
    """
    # 1. Load the full dataset. A failure here (missing/corrupt data file) is a
    #    genuinely unexpected error; we let it propagate so the agent layer can
    #    report "search failed", distinct from the normal "no matches" case.
    listings = load_listings()

    # 2. Hard filters: price ceiling (inclusive) and size, when provided.
    filtered = []
    for item in listings:
        if max_price is not None and item.get("price", float("inf")) > max_price:
            continue
        if size and not _size_matches(size, item.get("size", "")):
            continue
        filtered.append(item)

    # 3/4. Score by keyword relevance and drop listings with no overlap.
    keywords = _extract_keywords(description)
    if not keywords:
        # No usable keywords (e.g. blank description): fall back to a
        # price/size-only search, cheapest first.
        return sorted(filtered, key=lambda x: x.get("price", 0.0))

    scored = []
    for item in filtered:
        score = _relevance_score(item, keywords)
        if score > 0:
            scored.append((score, item))

    # 5. Sort by relevance (desc), tie-break by price (ascending). Return dicts only.
    scored.sort(key=lambda pair: (-pair[0], pair[1].get("price", 0.0)))
    return [item for _, item in scored]


# ── Tool 1b: search_with_fallback ─────────────────────────────────────────────

def search_with_fallback(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> dict:
    """
    Search listings, automatically loosening constraints when nothing matches.

    Calls search_listings() with the full constraint set first. If that returns
    no results, it retries with progressively looser filters: dropping the size
    filter first, then the price ceiling, and stops at the first relaxation that
    yields any results. Each relaxation is recorded in human-readable form so the
    agent can tell the user exactly what was adjusted (e.g. "removed the size M
    filter"). Size is relaxed before price because an exact size is usually less
    central to intent than staying within budget; both are dropped if needed.

    A genuinely empty result (nothing matches even with every filter removed) is
    a normal outcome: the function returns an empty list rather than raising, with
    `adjustments` listing every loosening that was attempted so the caller can
    still explain what it tried.

    Args:
        description (str): Keywords describing the item (passed through unchanged).
        size (str | None): Size token to filter by, or None to skip.
        max_price (float | None): Maximum price inclusive, or None to skip.

    Returns:
        dict: A result dict with keys:
            results (list[dict]): Matching listings (possibly empty), ranked as
                search_listings() ranks them.
            adjustments (list[str]): Human-readable descriptions of each filter
                that was loosened to obtain `results`; empty when the first,
                fully-constrained search already matched.
            size (str | None): The size filter actually used for `results`
                (None if it was dropped).
            max_price (float | None): The price ceiling actually used for
                `results` (None if it was dropped).
    """
    # First pass: honor every constraint the user gave.
    results = search_listings(description, size, max_price)
    if results:
        return {"results": results, "adjustments": [], "size": size, "max_price": max_price}

    adjustments = []
    eff_size, eff_max_price = size, max_price

    # Relaxation 1 → drop the size filter (usually the most restrictive).
    if eff_size is not None:
        adjustments.append(f"removed the size {eff_size} filter")
        eff_size = None
        results = search_listings(description, eff_size, eff_max_price)
        if results:
            return {
                "results": results,
                "adjustments": adjustments,
                "size": eff_size,
                "max_price": eff_max_price,
            }

    # Relaxation 2 → lift the price ceiling.
    if eff_max_price is not None:
        adjustments.append(f"lifted the ${eff_max_price:.0f} budget cap")
        eff_max_price = None
        results = search_listings(description, eff_size, eff_max_price)
        if results:
            return {
                "results": results,
                "adjustments": adjustments,
                "size": eff_size,
                "max_price": eff_max_price,
            }

    # Nothing matched even with every filter relaxed → a normal empty result.
    return {
        "results": results,
        "adjustments": adjustments,
        "size": eff_size,
        "max_price": eff_max_price,
    }


# ── Tool 2 Helpers ────────────────────────────────────────────────────────────

def _format_item(item: dict) -> str:
    """
    Render a listing dict as a single line summary for inclusion in an LLM prompt.

    Args:
        item (dict): A listing dict (title, category, colors, style_tags).

    Returns:
        str: A compact human-readable line, e.g.
            "Vintage Band Tee (category: tops; colors: grey; style: vintage, grunge)".
    """
    tags = ", ".join(item.get("style_tags", []))
    colors = ", ".join(item.get("colors", []))
    return (
        f"{item.get('title', 'Unknown item')} "
        f"(category: {item.get('category', 'n/a')}; "
        f"colors: {colors or 'n/a'}; style: {tags or 'n/a'})"
    )


def _format_wardrobe(wardrobe: dict) -> str:
    """
    Render a wardrobe dict as a bullet list for inclusion in an LLM prompt.

    Formats each wardrobe item with its name, category, colors, style tags, and
    optional notes. The structure gives the model enough context to reference
    specific owned pieces by name when composing an outfit.

    Args:
        wardrobe (dict): A wardrobe dict with an 'items' list of wardrobe item dicts.

    Returns:
        str: Newline-separated bullet lines, one per wardrobe item.
    """
    lines = []
    for w in wardrobe.get("items", []):
        tags = ", ".join(w.get("style_tags", []))
        colors = ", ".join(w.get("colors", []))
        note = f" — {w['notes']}" if w.get("notes") else ""
        lines.append(
            f"- {w.get('name', 'Unnamed')} [{w.get('category', 'n/a')}; "
            f"{colors}; {tags}]{note}"
        )
    return "\n".join(lines)


def _normalize_outfit(parsed: dict) -> dict:
    """
    Normalize a parsed outfit dict to guarantee the four contract fields exist
    with the correct types.

    Coerces outfit_description, style_reasoning, and style_category to str, and
    normalizes matching_items to a list[str] even if the model returned a bare
    string. Missing fields default to empty string / empty list.

    Args:
        parsed (dict): A dict parsed from the model's JSON response.

    Returns:
        dict: Guaranteed keys — outfit_description (str), matching_items (list[str]),
            style_reasoning (str), style_category (str).
    """
    matching = parsed.get("matching_items", [])
    if isinstance(matching, str):
        matching = [matching]
    return {
        "outfit_description": str(parsed.get("outfit_description", "")).strip(),
        "matching_items": [str(m) for m in matching] if isinstance(matching, list) else [],
        "style_reasoning": str(parsed.get("style_reasoning", "")).strip(),
        "style_category": str(parsed.get("style_category", "")).strip(),
    }


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def suggest_outfit(new_item: dict, wardrobe: dict) -> dict:
    """
    Given a thrifted item and the user's wardrobe, suggest a complete outfit.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty; handled gracefully via a
                  generic-staples fallback.

    Returns:
        A dict (per planning.md Tool 2 spec) with:
            outfit_description (str): the recommended outfit in plain language
            matching_items (list[str]): named wardrobe pieces used (or, when the
                wardrobe is empty, the generic staples suggested)
            style_reasoning (str): why the combination works
            style_category (str): aesthetic label (e.g. "streetwear")

    Failure modes:
        - Empty wardrobe → LLM is asked for generic staples; matching_items
          holds those staples and outfit_description flags they are suggestions,
          not owned pieces. This is NOT an error; the loop continues.
        - LLM/API failure → the underlying exception propagates so the agent can
          catch it, set error_message, and stop (it never returns None silently).
        - Malformed JSON from the LLM → falls back to wrapping the raw text in a
          well-formed dict rather than crashing.
    """
    items = wardrobe.get("items", []) if isinstance(wardrobe, dict) else []
    item_summary = _format_item(new_item)

    if not items:
        # Empty-wardrobe fallback: recommend generic staples to build around.
        user_prompt = (
            f"A shopper is considering this secondhand item:\n{item_summary}\n\n"
            "They have NOT entered any wardrobe yet, so you cannot reference items "
            "they own. Suggest one complete outfit built around this piece using "
            "generic staple items (e.g. straight-leg jeans, white sneakers). Make "
            "clear in the description that these are general suggestions, not items "
            "they already own.\n\n"
            "Respond with ONLY a JSON object with these keys:\n"
            '  "outfit_description" (string),\n'
            '  "matching_items" (array of the generic staple names you suggested),\n'
            '  "style_reasoning" (string, 1-2 sentences),\n'
            '  "style_category" (string, e.g. "streetwear", "cottagecore").'
        )
    else:
        user_prompt = (
            f"A shopper is considering this secondhand item:\n{item_summary}\n\n"
            f"Here is their current wardrobe:\n{_format_wardrobe(wardrobe)}\n\n"
            "Suggest ONE complete, wearable outfit that pairs the new item with "
            "specific pieces they already own. Reference the wardrobe pieces by "
            "their exact names.\n\n"
            "Respond with ONLY a JSON object with these keys:\n"
            '  "outfit_description" (string referencing the new item and the '
            'wardrobe pieces by name),\n'
            '  "matching_items" (array of the exact wardrobe item names you used),\n'
            '  "style_reasoning" (string, 1-2 sentences on why it works),\n'
            '  "style_category" (string, e.g. "streetwear", "quiet luxury").'
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are FitFindr, a sharp, practical personal stylist for "
                "secondhand fashion. You always reply with valid JSON only."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    content = _chat(messages, temperature=0.7, json_mode=True)

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        # Defensive: model returned non-JSON despite json_mode. Wrap it so the
        # caller still receives the contract shape instead of a crash.
        return {
            "outfit_description": (content or "").strip(),
            "matching_items": [],
            "style_reasoning": "",
            "style_category": "",
        }

    return _normalize_outfit(parsed)


# ── Tool 3 Helpers ────────────────────────────────────────────────────────────

def _normalize_fit_card(parsed: dict) -> dict:
    """
    Normalize a parsed fit-card dict to guarantee the three contract fields exist
    with the correct types.

    Coerces fit_card_text and caption_tone to str, normalizes style_tags to a
    list[str] capped at four entries, and handles the case where the model returned
    style_tags as a bare string rather than an array.

    Args:
        parsed (dict): A dict parsed from the model's JSON response.

    Returns:
        dict: Guaranteed keys — fit_card_text (str), style_tags (list[str], max 4),
            caption_tone (str).
    """
    tags = parsed.get("style_tags", [])
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        tags = []
    return {
        "fit_card_text": str(parsed.get("fit_card_text", "")).strip(),
        "style_tags": [str(t).strip() for t in tags][:4],
        "caption_tone": str(parsed.get("caption_tone", "")).strip(),
    }


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: dict, new_item: dict) -> dict:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Args:
        outfit:   The outfit dict from suggest_outfit() — uses outfit_description
                  and style_category. May be None/empty or missing
                  outfit_description; handled via a simplified caption fallback.
        new_item: The listing dict for the thrifted item (title, price, platform).

    Returns:
        A dict (per planning.md Tool 3 spec) with:
            fit_card_text (str): casual 1-3 sentence caption (may have one emoji)
            style_tags (list[str]): 2-4 hashtag-style aesthetic keywords
            caption_tone (str): detected tone (e.g. "casual", "confident")

    Failure modes:
        - outfit missing/empty outfit_description → build a simplified caption
          from new_item fields only (does NOT crash, does NOT raise).
        - LLM/API failure → the exception propagates so the agent can skip the
          fit card and still display the outfit (partial success).
        - Malformed JSON → wrap the raw text as the caption rather than crashing.

    Different inputs (and even repeat calls) produce different captions: the LLM
    runs at high temperature and no fixed seed.
    """
    title = new_item.get("title", "this piece")
    price = new_item.get("price")
    platform = new_item.get("platform", "secondhand")
    price_str = f"${price:.0f}" if isinstance(price, (int, float)) else "a steal"

    outfit = outfit if isinstance(outfit, dict) else {}
    description = (outfit.get("outfit_description") or "").strip()
    style_category = (outfit.get("style_category") or "").strip()

    if not description:
        # Fallback: no outfit context — caption the find from the item alone.
        user_prompt = (
            f"Write a short, authentic social-media caption for a secondhand "
            f"find. Item: {title}, bought for {price_str} on {platform}. There "
            f"is no styling context, so focus on the excitement of the find "
            f"itself.\n\n"
        )
    else:
        category_line = f"Overall vibe: {style_category}.\n" if style_category else ""
        user_prompt = (
            f"Write a short, authentic social-media caption for a secondhand "
            f"find styled into an outfit.\n"
            f"Item: {title}, bought for {price_str} on {platform}.\n"
            f"Outfit: {description}\n"
            f"{category_line}\n"
        )

    user_prompt += (
        "Style rules: sound like a real person posting an OOTD, NOT a product "
        "description. 1-3 sentences. Mention the item name, price, and platform "
        "naturally (once each). At most one emoji.\n\n"
        "Respond with ONLY a JSON object with these keys:\n"
        '  "fit_card_text" (string, the caption),\n'
        '  "style_tags" (array of 2-4 short hashtag-style aesthetic keywords, '
        "no # symbol),\n"
        '  "caption_tone" (string, e.g. "casual", "confident", "nostalgic").'
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are FitFindr's caption writer. You write punchy, authentic "
                "thrift-haul captions and always reply with valid JSON only."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    # High temperature so repeat calls on the same input still vary.
    content = _chat(messages, temperature=1.0, json_mode=True)

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {
            "fit_card_text": (content or "").strip(),
            "style_tags": [],
            "caption_tone": "",
        }

    return _normalize_fit_card(parsed)


# ── Tool 4 Helpers ────────────────────────────────────────────────────────────

# Fewer than this many comparables is too thin a sample to call a price fair or
# not, so price_compare() returns an "insufficient_data" verdict instead.
_MIN_COMPARABLES = 2

# Fairness bands on the ratio of the item's price to the comparable median.
# A price within ±15% of the median is "fair"; outside that it reads as a deal
# or a markup. The asymmetry-free band heuristic keeps the verdict easy to explain.
_UNDERPRICED_RATIO = 0.85
_OVERPRICED_RATIO = 1.15


def _find_comparables(item: dict, listings: list[dict]) -> list[dict]:
    """
    Select the listings that are comparable to a given item for pricing.

    A peer is comparable when it is in the same category AND shares at least one
    style tag with the item. That pairing keeps "a vintage denim jacket" from
    being priced against an unrelated plain blazer in the same category. To avoid
    a verdict built on too thin a sample, falls back to all same-category listings
    when the tag overlap set has fewer than _MIN_COMPARABLES entries. The item
    itself is always excluded by id so a listing never prices against itself.

    Args:
        item (dict): The listing being evaluated (uses id, category, style_tags).
        listings (list[dict]): The full dataset to draw comparables from.

    Returns:
        list[dict]: Comparable listing dicts (may be empty when the category has
            no other members).
    """
    item_id = item.get("id")
    category = (item.get("category") or "").lower()
    item_tags = _tokens(" ".join(item.get("style_tags", [])))

    same_category = [
        other
        for other in listings
        if other.get("id") != item_id
        and (other.get("category") or "").lower() == category
    ]

    # Prefer peers that also share a style tag; fall back to the whole category
    # when that narrower set is too small to judge against.
    tag_overlap = [
        other
        for other in same_category
        if _tokens(" ".join(other.get("style_tags", []))) & item_tags
    ]
    if len(tag_overlap) >= _MIN_COMPARABLES:
        return tag_overlap
    return same_category


def _price_verdict(item_price: float, median_price: float) -> str:
    """
    Classify an item's price against the comparable median into a fairness band.

    Compares the ratio item_price / median_price against the _UNDERPRICED_RATIO
    and _OVERPRICED_RATIO thresholds. A median of zero (free comparables) cannot
    yield a meaningful ratio, so it is reported as "insufficient_data".

    Args:
        item_price (float): The item's own asking price.
        median_price (float): Median price of the comparable listings.

    Returns:
        str: One of "underpriced", "fair", "overpriced", or "insufficient_data".
    """
    if median_price <= 0:
        return "insufficient_data"
    ratio = item_price / median_price
    if ratio <= _UNDERPRICED_RATIO:
        return "underpriced"
    if ratio >= _OVERPRICED_RATIO:
        return "overpriced"
    return "fair"


# ── Tool 4: price_compare ─────────────────────────────────────────────────────

def price_compare(item: dict) -> dict:
    """
    Estimate whether an item's asking price is fair against comparable listings.

    Pure and deterministic (no LLM call), mirroring search_listings(): it loads
    the dataset, gathers comparable listings via _find_comparables() (same
    category + shared style tag, with a same-category fallback), then judges the
    item's price against the median of those peers using the _UNDERPRICED_RATIO /
    _OVERPRICED_RATIO bands. Thin data is a normal outcome: when fewer than
    _MIN_COMPARABLES peers exist, or the item has no usable price, it returns a
    "insufficient_data" verdict instead of raising.

    Args:
        item (dict): The listing being evaluated (uses id, price, category,
            style_tags).

    Returns:
        dict: A verdict dict with keys:
            verdict (str): "underpriced", "fair", "overpriced", or
                "insufficient_data".
            item_price (float | None): The item's asking price, echoed back.
            comparable_count (int): Number of comparable listings found.
            comparable_avg (float | None): Mean comparable price, or None when
                the sample is too thin.
            comparable_median (float | None): Median comparable price, or None.
            comparable_range (list[float] | None): [min, max] comparable price,
                or None.
            explanation (str): One-sentence plain-language summary.
    """
    # Load the full dataset. A failure here (missing/corrupt data file) is a
    # genuinely unexpected error; we let it propagate so the agent layer can
    # report "comparison failed", distinct from the normal "not enough
    # comparables" outcome below.
    listings = load_listings()

    price = item.get("price")
    if not isinstance(price, (int, float)):
        # No usable price → nothing to compare against.
        return {
            "verdict": "insufficient_data",
            "item_price": None,
            "comparable_count": 0,
            "comparable_avg": None,
            "comparable_median": None,
            "comparable_range": None,
            "explanation": "This item has no listed price, so its fairness can't be assessed.",
        }
    price = float(price)

    comparables = _find_comparables(item, listings)
    prices = [
        float(c["price"])
        for c in comparables
        if isinstance(c.get("price"), (int, float))
    ]

    if len(prices) < _MIN_COMPARABLES:
        return {
            "verdict": "insufficient_data",
            "item_price": price,
            "comparable_count": len(prices),
            "comparable_avg": None,
            "comparable_median": None,
            "comparable_range": None,
            "explanation": (
                f"Only {len(prices)} comparable listing(s) found — not enough to "
                "judge whether this price is fair."
            ),
        }

    avg_price = round(statistics.fmean(prices), 2)
    median_price = round(statistics.median(prices), 2)
    low, high = round(min(prices), 2), round(max(prices), 2)
    verdict = _price_verdict(price, median_price)

    explanations = {
        "underpriced": (
            f"At ${price:.0f}, this is a deal — comparable items typically go for "
            f"around ${median_price:.0f} (${low:.0f}–${high:.0f})."
        ),
        "fair": (
            f"At ${price:.0f}, this is priced fairly — comparable items typically "
            f"go for around ${median_price:.0f} (${low:.0f}–${high:.0f})."
        ),
        "overpriced": (
            f"At ${price:.0f}, this is on the high side — comparable items typically "
            f"go for around ${median_price:.0f} (${low:.0f}–${high:.0f})."
        ),
    }

    return {
        "verdict": verdict,
        "item_price": price,
        "comparable_count": len(prices),
        "comparable_avg": avg_price,
        "comparable_median": median_price,
        "comparable_range": [low, high],
        "explanation": explanations[verdict],
    }


# ── Style Profile Memory ──────────────────────────────────────────────────────

# Where the cross-session style profile is persisted. It lives at the project
# root (next to this file) and is git-ignored: it holds per-user preferences,
# not fixtures, so it must never be committed alongside the mock data in data/.
# Tests monkeypatch this constant to redirect writes to a temp file.
_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "style_profiles.json")

# Cap on remembered favorite style tags so the profile can't grow without bound
# as the user runs more searches; the most recently seen tags win at the cap.
_MAX_FAVORITE_STYLES = 12


def _empty_profile() -> dict:
    """
    Return a fresh, empty style profile with every contract field present.

    Used both as the value returned by load_style_profile() when no profile has
    been saved yet and as the normalization fallback when a stored profile is
    missing or malformed. Keeping the empty shape in one place means callers can
    always rely on the four keys existing regardless of disk state.

    Args:
        None

    Returns:
        dict: A profile dict with keys — preferred_size (None), max_price (None),
            favorite_styles (empty list), wardrobe (dict with empty items list).
    """
    return {
        "preferred_size": None,
        "max_price": None,
        "favorite_styles": [],
        "wardrobe": {"items": []},
    }


def _normalize_profile(parsed: dict) -> dict:
    """
    Coerce a raw profile dict (from disk or a caller) into the contract shape.

    Guarantees the four profile fields exist with the correct types so the agent
    never has to defend against a half-populated or hand edited profile file:
    preferred_size becomes a str or None, max_price a float or None,
    favorite_styles a capped list[str], and wardrobe a dict with an items list.
    A non-dict input (e.g. a corrupt entry) degrades to a fresh empty profile.

    Args:
        parsed (dict): A dict read from the profile store, or any value to coerce.

    Returns:
        dict: A profile dict guaranteed to match the _empty_profile() shape.
    """
    if not isinstance(parsed, dict):
        return _empty_profile()

    styles = parsed.get("favorite_styles", [])
    if not isinstance(styles, list):
        styles = []

    wardrobe = parsed.get("wardrobe")
    if not isinstance(wardrobe, dict) or not isinstance(wardrobe.get("items"), list):
        wardrobe = {"items": []}

    size = parsed.get("preferred_size")
    price = parsed.get("max_price")
    return {
        "preferred_size": str(size) if size else None,
        "max_price": float(price) if isinstance(price, (int, float)) else None,
        "favorite_styles": [str(s) for s in styles][-_MAX_FAVORITE_STYLES:],
        "wardrobe": wardrobe,
    }


def _read_profile_store() -> dict:
    """
    Read the whole profile store (profile_id → profile) from disk.

    The store is a single JSON object mapping profile ids to profile dicts. A
    missing file (no profile saved yet) or a corrupt/unreadable one is treated as
    an empty store rather than an error, so the memory feature degrades to "no
    memory" instead of breaking the agent on a first run or a bad file.

    Args:
        None

    Returns:
        dict: The raw store mapping ids to profile dicts; empty dict when the
            file is absent or cannot be parsed.
    """
    try:
        with open(_PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_style_profile(profile_id: str = "default") -> dict:
    """
    Load a saved style profile so a returning user need not re-describe everything.

    Reads the profile store from disk and returns the normalized profile for the
    given id. When no profile has been saved under that id yet, returns a fresh
    empty profile (never None and never raises), so the agent can treat first-time
    and returning users through the same code path.

    Args:
        profile_id (str): Identifier for the profile to load. Defaults to
            "default", the single-user profile used by the Gradio UI.

    Returns:
        dict: A profile dict with keys preferred_size (str | None), max_price
            (float | None), favorite_styles (list[str]), and wardrobe (dict).
    """
    store = _read_profile_store()
    return _normalize_profile(store.get(profile_id, {}))


def save_style_profile(profile: dict, profile_id: str = "default") -> dict:
    """
    Persist a style profile to disk under the given id, preserving other profiles.

    Normalizes the profile to the contract shape, merges it into the existing
    store (so saving one profile id never clobbers another), and writes the whole
    store back as indented JSON. Returns the normalized profile that was written
    so callers can reuse it without re-reading the file.

    Args:
        profile (dict): The profile dict to save; normalized before writing.
        profile_id (str): Identifier to store the profile under. Defaults to
            "default".

    Returns:
        dict: The normalized profile dict as written to disk.
    """
    normalized = _normalize_profile(profile)
    store = _read_profile_store()
    store[profile_id] = normalized
    with open(_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)
    return normalized


def update_style_profile(
    size: str | None = None,
    max_price: float | None = None,
    styles: list[str] | None = None,
    wardrobe: dict | None = None,
    profile_id: str = "default",
) -> dict:
    """
    Merge newly observed preferences into the saved profile and persist it.

    Called after a successful interaction to remember what the user searched for:
    the size and budget they used, the style tags of the item they landed on, and
    the wardrobe in play. Only arguments that are provided overwrite existing
    values (a None/empty argument leaves that field untouched), so a single search
    that omits a size does not erase a previously remembered one. New style tags
    are appended without duplicates and capped at _MAX_FAVORITE_STYLES, newest kept.

    Args:
        size (str | None): Size to remember as preferred, or None to leave as-is.
        max_price (float | None): Budget to remember, or None to leave as-is.
        styles (list[str] | None): Style tags to fold into favorite_styles, or None.
        wardrobe (dict | None): Wardrobe to remember, or None to leave as-is.
        profile_id (str): Profile to update. Defaults to "default".

    Returns:
        dict: The updated, normalized profile dict as written to disk.
    """
    profile = load_style_profile(profile_id)
    if size:
        profile["preferred_size"] = size
    if max_price is not None:
        profile["max_price"] = float(max_price)
    if styles:
        existing = profile["favorite_styles"]
        for tag in styles:
            if tag and tag not in existing:
                existing.append(tag)
        profile["favorite_styles"] = existing[-_MAX_FAVORITE_STYLES:]
    if wardrobe is not None and isinstance(wardrobe, dict):
        profile["wardrobe"] = wardrobe
    return save_style_profile(profile, profile_id)
