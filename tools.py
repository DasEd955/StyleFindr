"""
tools.py

The three required FitFindr tools. Each tool is a standalone function that
can be called and tested independently before being wired into the agent loop.

Complete and test each tool before moving to agent.py.

Tools:
    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)              → str
    create_fit_card(outfit, new_item)               → str
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
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


def _chat(messages: list[dict], temperature: float, json_mode: bool = False) -> str:
    """
    Thin wrapper around a Groq chat completion. Raises on API/network failure so
    the agent layer can catch it and report the error (per the Error Handling
    table in planning.md) — it does not swallow exceptions.
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
_STOPWORDS = {
    "a", "an", "the", "for", "with", "in", "of", "and", "or", "to",
    "my", "i", "me", "some", "looking", "want", "need", "size",
}


def _extract_keywords(description: str) -> list[str]:
    """Lower-case, strip punctuation, and drop stopwords/short tokens."""
    if not description:
        return []
    tokens = re.split(r"[^a-z0-9]+", description.lower())
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]


def _size_matches(requested: str, listing_size: str) -> bool:
    """
    True if a listing's size satisfies the requested size.

    Sizes in the dataset are inconsistent ("S/M", "XL (oversized)", "M/L",
    "One Size / Oversized", "W30 L30", "US 7"). We tokenize the listing size
    and check for an exact token match so "M" matches "S/M" and "M/L" but NOT
    "XL" (which a naive substring check would wrongly catch). "One Size" items
    fit anyone, so they match any requested size.
    """
    if not requested:
        return True
    req = requested.strip().upper()
    listing = (listing_size or "").upper()

    # Universal-fit items satisfy every request.
    if "ONE SIZE" in listing:
        return True

    tokens = [t for t in re.split(r"[^A-Z0-9]+", listing) if t]
    return req in tokens


def _relevance_score(listing: dict, keywords: list[str]) -> int:
    """
    Weighted keyword-overlap score. Style tags are the strongest signal,
    then title/colors/category/brand, then the free-text description.
    """
    title = listing.get("title", "").lower()
    desc = listing.get("description", "").lower()
    category = listing.get("category", "").lower()
    brand = (listing.get("brand") or "").lower()
    tag_text = " ".join(listing.get("style_tags", [])).lower()
    color_text = " ".join(listing.get("colors", [])).lower()

    score = 0
    for kw in keywords:
        if kw in tag_text:
            score += 3
        if kw in title:
            score += 2
        if kw in color_text:
            score += 2
        if kw == category:
            score += 2
        if brand and kw in brand:
            score += 2
        if kw in desc:
            score += 1
    return score


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Args:
        description: Keywords describing what the user is looking for
                     (e.g., "vintage graphic tee").
        size:        Size string to filter by, or None to skip size filtering.
                     Matching is case-insensitive (e.g., "M" matches "S/M").
        max_price:   Maximum price (inclusive), or None to skip price filtering.

    Returns:
        A list of matching listing dicts, sorted by relevance (best match first).
        Returns an empty list if nothing matches — does NOT raise an exception.

    Each listing dict has the following fields:
        id, title, description, category, style_tags (list), size,
        condition, price (float), colors (list), brand, platform

    TODO:
        1. Load all listings with load_listings().
        2. Filter by max_price and size (if provided).
        3. Score each remaining listing by keyword overlap with `description`.
        4. Drop any listings with a score of 0 (no relevant matches).
        5. Sort by score, highest first, and return the listing dicts.

    Before writing code, fill in the Tool 1 section of planning.md.
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
    """One-line summary of a listing for the LLM prompt."""
    tags = ", ".join(item.get("style_tags", []))
    colors = ", ".join(item.get("colors", []))
    return (
        f"{item.get('title', 'Unknown item')} "
        f"(category: {item.get('category', 'n/a')}; "
        f"colors: {colors or 'n/a'}; style: {tags or 'n/a'})"
    )


def _format_wardrobe(wardrobe: dict) -> str:
    """Bullet list of wardrobe pieces, grouped enough for the LLM to reason over."""
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
    """Guarantee all four contract fields exist with the right types."""
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
    """Guarantee the fit-card contract fields exist with the right types."""
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
