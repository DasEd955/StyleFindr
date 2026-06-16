# StyleFindr — FitFindr Agent

A multi-tool AI agent that helps users find secondhand clothing listings, suggests outfits intelligently using the user's wardrobe, and generates shareable fit captions for social media: all from a single natural-language processing query.

---

## Tool Inventory & Outline

### Tool 1: `search_listings`

**File:** [tools.py:190](tools.py#L190)

| | |
|---|---|
| **Input: `description`** | `str` — Keywords describing the item (e.g., `"vintage graphic tee"`). Stopwords and filler are stripped; whole-word token matching prevents substring noise. |
| **Input: `size`** | `str \| None` — Size filter (e.g., `"M"`, `"XL"`, `"8"`). Apparel sizes and numeric shoe/waist sizes are treated as separate systems so requesting size `"M"` does not exclude shoes. `None` skips size filtering. |
| **Input: `max_price`** | `float \| None` — Price ceiling (inclusive). `None` skips price filtering. |
| **Output** | `list[dict]` — Matching listing dicts sorted by weighted relevance (style tags > title/colors/category/brand > description). Empty list `[]` if nothing matches — never raises. |

**Purpose:** Pure keyword + hard-filter search over the 40-item mock dataset (`data/listings.json`). No LLM call; fully deterministic. The agent's entry point — if this returns empty, the loop stops here.

---

### Tool 2: `suggest_outfit`

**File:** [tools.py:297](tools.py#L297)

| | |
|---|---|
| **Input: `new_item`** | `dict` — Full listing dict from `search_listings` (uses `title`, `category`, `colors`, `style_tags`). |
| **Input: `wardrobe`** | `dict` — Wardrobe dict with an `"items"` key. Accepts empty wardrobe — triggers a generic-staples fallback without crashing. |
| **Output** | `dict` with keys: `outfit_description` (str), `matching_items` (list[str]), `style_reasoning` (str), `style_category` (str). |

**Purpose:** Uses the Groq LLM (`llama-3.3-70b-versatile`) to reason about style compatibility and suggest one complete, wearable outfit pairing the thrifted item with pieces the user already owns. If the wardrobe is empty it suggests generic staples and flags them as such.

---

### Tool 3: `create_fit_card`

**File:** [tools.py:403](tools.py#L403)

| | |
|---|---|
| **Input: `outfit`** | `dict` — Outfit dict from `suggest_outfit` (uses `outfit_description`, `style_category`). Accepts `None` or missing `outfit_description` — falls back to item-only caption. |
| **Input: `new_item`** | `dict` — Listing dict from `search_listings` (uses `title`, `price`, `platform`). |
| **Output** | `dict` with keys: `fit_card_text` (str, 1–3 sentence caption), `style_tags` (list[str], 2–4 hashtag-style keywords), `caption_tone` (str, e.g. `"casual"`, `"confident"`). |

**Purpose:** Generates an authentic, casual social-media OOTD caption for the thrifted find. Runs at `temperature=1.0` so repeat calls on the same input still vary. This is the only tool whose failure is treated as a partial success — the agent displays the outfit without a fit card rather than stopping.

---

## Planning Loop

**File:** [agent.py:110](agent.py#L110)

The agent uses a **sequential conditional loop** — each step checks the previous output before deciding whether to continue. Tools are not called unconditionally.

```
Step 1 — Parse query (regex, no LLM)
         Extract: description, size, max_price
         Store in session["parsed"]

Step 2 — search_listings(description, size, max_price)
         → Empty list?  Set session["error"], STOP (no results message)
         → Results?     session["selected_item"] = results[0], continue

Step 3 — suggest_outfit(selected_item, wardrobe)
         → LLM failure? Set session["error"], STOP (outfit error message)
         → Empty outfit? Set session["error"], STOP
         → Empty wardrobe? Fallback to staples, continue (NOT an error)
         → Success?     session["outfit_suggestion"] = outfit, continue

Step 4 — create_fit_card(outfit_suggestion, selected_item)
         → LLM failure? session["fit_card"] = None, continue (partial success)
         → Success?     session["fit_card"] = fit_card, continue

Step 5 — Return completed session dict
```

**Query parsing choice:** Step 1 uses regex (`_parse_query()` in [agent.py:38](agent.py#L38)), not an LLM call. Price numbers (`under $30`, `$30`), size tokens (`size M`, standalone `XS`/`S`/`M`/`L`/`XL`), and leftover keywords are cheap and deterministic to extract with patterns. Skipping an LLM hop here reduces latency and removes a failure surface before any tool runs.

**Key behavioral rule:** An empty `search_listings` result stops the loop entirely. An empty wardrobe in `suggest_outfit` triggers a fallback path but does NOT stop the loop. A failed `create_fit_card` degrades gracefully — the outfit is still shown.

---

## State Management

**File:** [agent.py:86](agent.py#L86)

A single **session dict** is initialized at the start of each `run_agent()` call and serves as the sole source of truth for the interaction. No tool re-receives information the user already provided.

```python
session = {
    "query":             str,          # original user query (unchanged throughout)
    "parsed":            dict,         # {description, size, max_price} from _parse_query
    "search_results":    list[dict],   # full list from search_listings
    "selected_item":     dict | None,  # search_results[0] — top-ranked listing
    "wardrobe":          dict,         # loaded once, passed directly to suggest_outfit
    "outfit_suggestion": dict | None,  # full dict from suggest_outfit
    "fit_card":          dict | None,  # full dict from create_fit_card (None = partial success)
    "error":             str | None,   # set on early termination; None on success
}
```

**Data flow:** `selected_item` passes directly from `search_listings` into `suggest_outfit` without user re-entry. Both `outfit_suggestion` and `selected_item` pass directly into `create_fit_card`. The Gradio UI ([app.py:23](app.py#L23)) reads `session["selected_item"]`, `session["outfit_suggestion"]`, and `session["fit_card"]` to populate the three output panels.

---

## Error Handling

| Tool | Failure Mode | Agent Response |
|------|-------------|----------------|
| `search_listings` | Returns `[]` | Sets `session["error"]`: *"No listings matched '[desc]' in size [size] under [price]. Try a broader description, a higher budget, or a different size."* Loop stops. |
| `search_listings` | Unexpected exception (e.g., missing data file) | Sets `session["error"]`: *"Search failed due to an unexpected error. Please try again."* Loop stops. |
| `suggest_outfit` | Empty wardrobe | LLM is prompted for generic staples; `matching_items` holds those staples; `outfit_description` flags them as suggestions. Loop **continues** — this is not an error. |
| `suggest_outfit` | LLM/API failure | Sets `session["error"]`: *"Outfit generation failed. Please try again."* Loop stops. `create_fit_card` is never called with empty input. |
| `suggest_outfit` | Malformed JSON from LLM | Wraps raw text as `outfit_description`; other fields default to empty. Caller receives the contract shape instead of a crash. |
| `create_fit_card` | `outfit_description` missing or empty | Falls back to item-only caption using `title`, `price`, `platform` only. |
| `create_fit_card` | LLM/API failure | `session["fit_card"]` is set to `None`. Gradio shows *"Fit card generation failed — see your outfit suggestion instead."* Session is still returned (partial success). |
| `create_fit_card` | Malformed JSON from LLM | Wraps raw text as `fit_card_text`; `style_tags` defaults to `[]`. |

**Concrete examples from testing:**

- **`test_search_whole_word_match_no_substring_leak`** ([tests/test_tools.py:88](tests/test_tools.py#L88)): The query `"new pair of combat boots we could keep cheap"` previously surfaced a western-tagged belt as the top hit because `"we"` substring-matched `"western"`. Whole-word token matching fixed this — the test asserts no belt appears in results.

- **`test_search_apparel_size_does_not_exclude_shoes`** ([tests/test_tools.py:101](tests/test_tools.py#L101)): Requesting size `"Medium"` (an alpha apparel size) previously excluded all numeric-sized shoes (`"US 8"`). The fix treats the two sizing systems as non-comparable — the test asserts boots still appear in results.

- **`test_suggest_outfit_empty_wardrobe`** ([tests/test_tools.py:150](tests/test_tools.py#L150)): Passing `{"items": []}` to `suggest_outfit` must not crash and must route to the generic-staples branch. The test confirms the returned dict matches the contract shape and that the LLM prompt contains the no-wardrobe fallback language.

- **`test_fit_card_malformed_json`** ([tests/test_tools.py:194](tests/test_tools.py#L194)): When the LLM returns `"<<garbage>>"` instead of JSON, `create_fit_card` wraps the raw string as `fit_card_text` and returns `style_tags: []` rather than raising.

---

## Spec Reflection

**What matched the plan:**

The implementation follows the planning.md spec closely. The three tools cover exactly the planned inputs and outputs. The sequential conditional loop in `run_agent()` matches the pseudocode step-for-step: empty search results stop the loop, empty wardrobe triggers a fallback but continues, and a failed fit card is treated as a partial success rather than an error. The session dict field names and data-flow arrows in the Mermaid diagram map directly to the code.

**What changed during implementation:**

1. **Size-filter logic became significantly more complex.** The original plan described a simple case-insensitive string match. Testing revealed that apparel sizes (`"M"`, `"S/M"`) and numeric shoe/waist sizes (`"US 8"`, `"W30 L30"`) are incompatible systems — filtering across them silently excluded entire categories. The fix introduced `_size_matches()` with cross-system pass-through logic, and a regression test to lock it down.

2. **Keyword matching switched from substring to whole-word tokens.** The original relevance scorer used `in` string containment. During testing, stopwords like `"we"` matched style tags like `"western"`, surfacing irrelevant results. Changing to token-set intersection fixed this and required adding `_STOPWORDS` and `_tokens()`.

3. **`suggest_outfit` return type changed from `str` to `dict`.** The planning.md spec listed the return as a dict, but an earlier implementation draft had it return a plain string. The dict shape (with `outfit_description`, `matching_items`, `style_reasoning`, `style_category`) was formalized to give `create_fit_card` and the Gradio formatter structured fields to work with.

4. **`create_fit_card` failure became partial success, not a hard stop.** The original error table said to display the outfit and note the failure. The implementation encodes this as `session["fit_card"] = None` and lets the Gradio formatter handle the display — which keeps the agent loop simple and the UI consistent.

---

<!--
## Original Starter Kit README (commented out — preserved for reference)

# FitFindr — Starter Kit

This starter kit contains everything you need to begin Project 2.

## What's Included

```
ai201-project2-fitfindr-starter/
├── data/
│   ├── listings.json          # 40 mock secondhand listings
│   └── wardrobe_schema.json   # Wardrobe format + example wardrobe
├── utils/
│   └── data_loader.py         # Helper functions for loading the data
├── planning.md                # Your planning template — fill this out first
└── requirements.txt           # Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

Set your Groq API key in a `.env` file (get a free key at [console.groq.com](https://console.groq.com)):
```
GROQ_API_KEY=your_key_here
```

## The Mock Listings Dataset

`data/listings.json` contains 40 mock secondhand listings across categories (tops, bottoms, outerwear, shoes, accessories) and styles (vintage, y2k, grunge, cottagecore, streetwear, and more).

Each listing has: `id`, `title`, `description`, `category`, `style_tags`, `size`, `condition`, `price`, `colors`, `brand`, and `platform`.

Load it with:
```python
from utils.data_loader import load_listings
listings = load_listings()
```

## The Wardrobe Schema

`data/wardrobe_schema.json` defines the format your agent uses to represent a user's existing wardrobe. It includes:

- `schema`: field definitions for a wardrobe item
- `example_wardrobe`: a sample wardrobe with 10 items you can use for testing
- `empty_wardrobe`: a starting template for a new user

Load an example wardrobe with:
```python
from utils.data_loader import get_example_wardrobe
wardrobe = get_example_wardrobe()
```

## Where to Start

1. **Read `planning.md` and fill it out before writing any code.**
2. Verify the data loads correctly by running `python utils/data_loader.py`.
3. Build and test each tool individually before connecting them through your planning loop.

Your implementation files go in this same directory. There's no required file structure for your agent code — organize it however makes sense for your design.
-->
