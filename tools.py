"""
tools.py - The three core FitFindr tools and their private helpers.

Each public tool is a standalone function that can be called and tested
independently before being wired into the agent loop in agent.py.

search_listings() is pure (no network): it loads the dataset, applies hard
price/size filters, then ranks by keyword-overlap score. suggest_outfit() and
create_fit_card() call the Groq LLM via the thin _chat() wrapper; both return
validated contract dicts and handle malformed JSON defensively rather than
raising. Private helpers (_tokens, _size_matches, _relevance_score, etc.) are
module-internal and not part of the public API.

Tools:
    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)              → dict
    create_fit_card(outfit, new_item)               → dict
"""

import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()

# Free-tier model shared across the LLM-backed tools.
_MODEL = "llama-3.3-70b-versatile"


# ── Groq client ───────────────────────────────────────────────────────────────

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
    it without touching each tool. Raises on API or network failure — it does
    not swallow exceptions — so the agent layer can catch and surface the error.

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


# ── Tool 1 helpers ──────────────────────────────────────────────────────────

# Common words that carry no search signal — dropped from query keywords so they
# don't inflate relevance scores (e.g. "a vintage tee for me" → ["vintage", "tee"]).
# Includes conversational filler ("looking for a new pair under my budget, thanks!")
# so polite, full-sentence queries don't leak noise words into the relevance score.
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
    Tokenize a string into a set of lower-case word tokens for whole-word matching.

    Splitting on non-alphanumeric boundaries and returning a set ensures that
    keyword comparisons in _relevance_score() never match substrings — "we"
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


def _normalize_size_token(tok: str) -> str:
    """
    Map a word-form size to its canonical letter abbreviation.

    Converts tokens like "MEDIUM" or "LARGE" to their single/two-letter forms
    ("M", "L") so they can be compared against the _ALPHA_SIZES set. Tokens
    that are not in _SIZE_WORDS (e.g. "XL", "8") are returned unchanged.

    Args:
        tok (str): An upper-case size token (e.g. "MEDIUM", "XL", "8.5").

    Returns:
        str: The canonical form if recognized, otherwise tok unchanged.
    """
    return _SIZE_WORDS.get(tok, tok)


def _size_matches(requested: str, listing_size: str) -> bool:
    """
    Return True if a listing's size satisfies the requested size.

    The dataset uses two incompatible sizing systems: alpha apparel sizes
    ("S/M", "XL (oversized)", "M/L") and numeric shoe/waist sizes ("US 7",
    "W30 L30"). A request only filters within its own system — asking for
    "Medium" (apparel) must NOT exclude shoes sized "US 8" because the two
    systems are not comparable. This cross-system pass-through prevents a
    clothing-size request from silently wiping out the shoe category.

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

    # Universal-fit items satisfy every request.
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
        return True  # listing has no apparel size (e.g. shoes) — not comparable
    if re.fullmatch(r"\d+(?:\.\d+)?", req):
        # Numeric request: filter only against numeric listings.
        if listing_numbers:
            return float(req) in listing_numbers
        return True  # listing has no numeric size — not comparable
    # Unrecognized request format → don't exclude on size.
    return True


def _relevance_score(listing: dict, keywords: list[str]) -> int:
    """
    Compute a weighted keyword-overlap score for a single listing.

    Matching is whole-word (token-based via _tokens()), not substring, so
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
    An empty result is a normal outcome — the function never raises on no matches.

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
    #    genuinely unexpected error — we let it propagate so the agent layer can
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

    # 5. Sort by relevance (desc), tie-break by price (asc). Return dicts only.
    scored.sort(key=lambda pair: (-pair[0], pair[1].get("price", 0.0)))
    return [item for _, item in scored]


# ── Tool 2 helpers ────────────────────────────────────────────────────────────

def _format_item(item: dict) -> str:
    """
    Render a listing dict as a single-line summary for inclusion in an LLM prompt.

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
                  wardrobe item dicts. May be empty — handled gracefully via a
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


# ── Tool 3 helpers ────────────────────────────────────────────────────────────

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
