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

import os
import re

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()


# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


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


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty — handle this gracefully.

    Returns:
        A non-empty string with outfit suggestions.
        If the wardrobe is empty, offer general styling advice for the item
        rather than raising an exception or returning an empty string.

    TODO:
        1. Check whether wardrobe['items'] is empty.
        2. If empty: call the LLM with a prompt for general styling ideas
           (what kinds of items pair well, what vibe it suits, etc.).
        3. If not empty: format the wardrobe items into a prompt and ask
           the LLM to suggest specific outfit combinations using the new item
           and named pieces from the wardrobe.
        4. Return the LLM's response as a string.

    Before writing code, fill in the Tool 2 section of planning.md.
    """
    # Replace this with your implementation
    return ""


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Args:
        outfit:   The outfit suggestion string from suggest_outfit().
        new_item: The listing dict for the thrifted item.

    Returns:
        A 2–4 sentence string usable as an Instagram/TikTok caption.
        If outfit is empty or missing, return a descriptive error message
        string — do NOT raise an exception.

    The caption should:
    - Feel casual and authentic (like a real OOTD post, not a product description)
    - Mention the item name, price, and platform naturally (once each)
    - Capture the outfit vibe in specific terms
    - Sound different each time for different inputs (use higher LLM temperature)

    TODO:
        1. Guard against an empty or whitespace-only outfit string.
        2. Build a prompt that gives the LLM the item details and the outfit,
           and asks for a caption matching the style guidelines above.
        3. Call the LLM and return the response.

    Before writing code, fill in the Tool 3 section of planning.md.
    """
    # Replace this with your implementation
    return ""
