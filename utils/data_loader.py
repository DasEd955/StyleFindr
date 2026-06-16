"""
data_loader.py - Utility helpers for loading the mock listings dataset and wardrobe schema.

All data lives in the sibling `data/` directory as JSON files. The helpers
here resolve paths relative to this file so they work regardless of the
working directory. `load_listings` and `load_wardrobe_schema` are the primary
loaders; `get_example_wardrobe` and `get_empty_wardrobe` are thin convenience
wrappers that spare callers from navigating the schema dict keys.
"""

import json
import os
from typing import Optional

# Resolve the path to the data directory relative to this file
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_listings() -> list[dict]:
    """
    Load and return all mock listings from the dataset JSON file.

    Reads data/listings.json relative to this file's directory. Raises on a
    missing or corrupt file so callers (e.g. search_listings) can surface a
    meaningful "search failed" error rather than an empty result.

    Args:
        None

    Returns:
        list[dict]: All listing dicts. Each dict carries:
            id (str), title (str), description (str),
            category (str — tops/bottoms/outerwear/shoes/accessories),
            style_tags (list[str]), size (str), condition (str — excellent/good/fair),
            price (float), colors (list[str]), brand (str | None),
            platform (str — depop/thredUp/poshmark).
    """
    path = os.path.join(_DATA_DIR, "listings.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_wardrobe_schema() -> dict:
    """
    Load and return the full wardrobe schema file, including field definitions,
    an example wardrobe, and an empty template.

    Args:
        None

    Returns:
        dict: A top-level dict with three keys:
            schema (dict): field definitions for a single wardrobe item,
            example_wardrobe (dict): a sample wardrobe with 10 pre-filled items,
            empty_wardrobe (dict): a starting template with an empty items list.
    """
    path = os.path.join(_DATA_DIR, "wardrobe_schema.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_example_wardrobe() -> dict:
    """
    Return the example wardrobe from the schema file.

    Convenience wrapper around load_wardrobe_schema() that spares callers from
    navigating the top-level schema dict. Pass the result directly to run_agent()
    or suggest_outfit() as the `wardrobe` argument.

    Args:
        None

    Returns:
        dict: A wardrobe dict with an 'items' key holding 10 pre-filled wardrobe items.
    """
    schema = load_wardrobe_schema()
    return schema["example_wardrobe"]


def get_empty_wardrobe() -> dict:
    """
    Return the empty wardrobe template from the schema file.

    Convenience wrapper around load_wardrobe_schema() for the new-user flow.
    suggest_outfit() handles an empty items list gracefully via a generic-staples
    fallback, so passing this is safe at every stage of the pipeline.

    Args:
        None

    Returns:
        dict: A wardrobe dict with an empty 'items' list.
    """
    schema = load_wardrobe_schema()
    return schema["empty_wardrobe"]


# --- Quick Sanity Check ---
if __name__ == "__main__":
    listings = load_listings()
    print(f"Loaded {len(listings)} listings.")
    print(f"First listing: {listings[0]['title']} — ${listings[0]['price']}")

    wardrobe = get_example_wardrobe()
    print(f"\nExample wardrobe has {len(wardrobe['items'])} items.")
    print(f"First item: {wardrobe['items'][0]['name']}")
