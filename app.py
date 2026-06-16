"""
app.py - Gradio interface for the FitFindr styling pipeline.

The UI is built intentionally thin. All search, price-check, outfit generation,
and fit card logic lives in agent.run_agent() and the four tools in tools.py;
this module only renders the results across three output panels. handle_query()
is the single Gradio callback; it guards against empty input, selects the
wardrobe, delegates to run_agent(), and maps the session dict to (listing_text,
outfit_text, fit_card_text). The price verdict is folded into listing_text.

The five private _format_* helpers convert raw dicts from the session into
human-readable strings for the Textbox panels. A per-session wardrobe is
chosen at submit time via a radio button (example, empty, or the saved profile),
so `new-user`, `existing-wardrobe`, and `returning-user` flows are exercised
through the same code path. A "Remember my preferences" checkbox turns on the
cross-session style profile: when set, run_agent() applies any previously saved
size/budget to the query and saves the preferences observed this run. The
profile-applied and search-fallback notes are surfaced by _format_notes() above
the listing panel.

Run with:
    python app.py

Then open the localhost URL shown in your terminal (usually http://localhost:7860).
"""

import gradio as gr
from agent import run_agent
from utils.data_loader import get_example_wardrobe, get_empty_wardrobe


# ── Query Handler ─────────────────────────────────────────────────────────────

# Profile id used by the single-user Gradio UI for cross-session memory.
_PROFILE_ID = "default"


def handle_query(user_query: str, 
                 wardrobe_choice: str, 
                 remember_prefs: bool = False,) -> tuple[str, str, str]:
    """
    Handle one Gradio submit event and return the three output panel strings.

    Guards against an empty query, selects the wardrobe based on the radio
    choice, wires up cross-session memory from the "remember" checkbox, delegates
    the full pipeline to run_agent(), and maps the resulting session dict to the
    three Textbox panels. On early termination (empty results or outfit failure)
    the error surfaces in the first panel and the other two are returned as empty
    strings.

    The wardrobe radio also drives the style profile: choosing "Saved profile"
    loads the wardrobe (and size/budget defaults) from the saved profile, while
    the "remember" checkbox enables saving this run's preferences. Either one
    activates the profile so the profile applied / saved notes can appear.

    Args:
        user_query (str): The text the user typed into the search box.
        wardrobe_choice (str): Radio value — "Example wardrobe",
            "Empty wardrobe (new user)", or "Saved profile".
        remember_prefs (bool): Whether to persist this run's preferences to the
            saved profile for next time.

    Returns:
        tuple[str, str, str]: (listing_text, outfit_text, fit_card_text).
            Each string maps to one of the three output Textbox panels.
            On error, only the first string is non-empty.
    """
    # 1. Guard against an empty query.
    if not user_query or not user_query.strip():
        return "Please enter what you're looking for.", "", ""

    # 2. Resolve the wardrobe and profile wiring from the controls. "Saved
    #    profile" loads the wardrobe from memory (wardrobe=None lets run_agent
    #    source it from the profile); the other choices pick a fixture wardrobe.
    profile_id = None
    wardrobe = None
    if wardrobe_choice == "Saved profile":
        profile_id = _PROFILE_ID
    elif wardrobe_choice == "Empty wardrobe (new user)":
        wardrobe = get_empty_wardrobe()
    else:
        wardrobe = get_example_wardrobe()

    # The "remember" checkbox both saves and (by activating the profile) lets any
    # previously saved size/budget defaults be applied to this query.
    save_profile = bool(remember_prefs)
    if save_profile:
        profile_id = _PROFILE_ID

    # 3. Run the planning loop.
    session = run_agent(
        user_query.strip(),
        wardrobe,
        profile_id=profile_id,
        save_profile=save_profile,
    )

    # 4. Early termination paths (empty results, search/outfit failure) surface
    #    the error in the first panel and leave the other two empty.
    if session["error"]:
        return session["error"], "", ""

    # 5. Map the session dict to the three panel strings. The price verdict is
    #    folded into the listing panel since it qualifies the item's price, and
    #    any memory / fallback notes are prepended above it.
    listing_text = _format_listing(session["selected_item"])
    price_text = _format_price(session["price_check"])
    if price_text:
        listing_text = f"{listing_text}\n\n{price_text}"

    notes_text = _format_notes(session)
    if notes_text:
        listing_text = f"{notes_text}\n\n{listing_text}"

    return (
        listing_text,
        _format_outfit(session["outfit_suggestion"]),
        _format_fit_card(session["fit_card"]),
    )


# ── Output Formatting ───────────────────────────────────────────────────────

def _format_listing(item: dict) -> str:
    """
    Render a listing dict into a human-readable string for the top listing panel.

    Args:
        item (dict): A listing dict from session["selected_item"].

    Returns:
        str: Multi-line text with title, price, platform, condition, size, and
            optional brand and style tags.
    """
    price = item.get("price")
    price_str = f"${price:.2f}" if isinstance(price, (int, float)) else "n/a"
    tags = ", ".join(item.get("style_tags", []))
    lines = [
        item.get("title", "Untitled listing"),
        f"{price_str} · {item.get('platform', 'n/a')} · {item.get('condition', 'n/a')} condition",
        f"Size: {item.get('size', 'n/a')}",
    ]
    if item.get("brand"):
        lines.append(f"Brand: {item['brand']}")
    if tags:
        lines.append(f"Style: {tags}")
    return "\n".join(lines)


def _format_notes(session: dict) -> str:
    """
    Render the cross-session memory and search-fallback notes for the listing panel.

    Surfaces three optional, non-error status lines, in order: preferences pulled
    from the saved profile and applied to this query (profile_applied), filters the
    fallback search had to loosen to find any results (search_adjustments), and a
    confirmation that this run's preferences were saved (profile_saved). Any subset
    may be empty; the helper returns "" when there is nothing to report so the
    listing panel is unchanged for the common case.

    Args:
        session (dict): The completed session dict from run_agent(). Reads
            profile_applied, search_adjustments, and profile_saved.

    Returns:
        str: Newline-separated note lines, or an empty string when there are none.
    """
    lines = []
    if session.get("profile_applied"):
        lines.append(
            "🧠 Applied your saved preferences: "
            + ", ".join(session["profile_applied"])
        )
    if session.get("search_adjustments"):
        lines.append(
            "🔁 No exact matches, so we "
            + " and ".join(session["search_adjustments"])
            + " to find these."
        )
    if session.get("profile_saved"):
        lines.append("💾 Saved your preferences for next time.")
    return "\n".join(lines)


# Verdict → short labelled headline for the price panel. Keys mirror the
# price_compare() verdict strings exactly; an unknown verdict yields no label.
_PRICE_LABELS = {
    "underpriced": "💰 Great Price",
    "fair": "✅ Fairly Priced",
    "overpriced": "⚠️ Priced High",
    "insufficient_data": "🤷 Not Enough Comparables",
}


def _format_price(price_check: dict | None) -> str:
    """
    Render a price_compare result dict into a labelled line for the listing panel.

    price_check is None when the price step failed (partial success): the rest of
    the listing panel is still shown, and this helper contributes nothing rather
    than crashing. The verdict maps to a short headline via _PRICE_LABELS, followed
    by price_compare's own one-sentence explanation.

    Args:
        price_check (dict | None): The dict from session["price_check"], or None on
            partial success. Expected keys: verdict, explanation.

    Returns:
        str: A two-line "label\\nexplanation" block, or an empty string when
            price_check is None.
    """
    if not price_check:
        return ""
    label = _PRICE_LABELS.get(price_check.get("verdict"), "")
    explanation = price_check.get("explanation", "")
    return f"{label}\n{explanation}".strip()


def _format_outfit(outfit: dict) -> str:
    """
    Render a suggest_outfit result dict into a human-readable string for the outfit panel.

    Args:
        outfit (dict): The outfit dict from session["outfit_suggestion"], containing
            outfit_description, matching_items, style_reasoning, and style_category.

    Returns:
        str: Multi-line text combining all four outfit fields, with blank lines
            separating the pieces, reasoning, and vibe label.
    """
    lines = [outfit.get("outfit_description", "")]
    if outfit.get("matching_items"):
        lines.append("\nPieces: " + ", ".join(outfit["matching_items"]))
    if outfit.get("style_reasoning"):
        lines.append(f"\nWhy it works: {outfit['style_reasoning']}")
    if outfit.get("style_category"):
        lines.append(f"\nVibe: {outfit['style_category']}")
    return "\n".join(lines).strip()


def _format_fit_card(fit_card: dict | None) -> str:
    """
    Render a create_fit_card result dict into a human-readable string for the fit-card panel.

    fit_card is None when the caption step failed (partial success): the outfit
    panel is still populated, and this panel displays a degraded-result notice
    rather than crashing or hiding the rest of the output.

    Args:
        fit_card (dict | None): The fit card dict from session["fit_card"], or None
            on partial success. Expected keys: fit_card_text, style_tags, caption_tone.

    Returns:
        str: The caption text, hashtag-style tags, and tone label joined as
            multi-line text; or a fallback message when fit_card is None.
    """
    if not fit_card:
        return "Fit card generation failed — see your outfit suggestion instead."
    lines = [fit_card.get("fit_card_text", "")]
    tags = fit_card.get("style_tags", [])
    if tags:
        lines.append("\n" + " ".join(f"#{t.replace(' ', '')}" for t in tags))
    if fit_card.get("caption_tone"):
        lines.append(f"\nTone: {fit_card['caption_tone']}")
    return "\n".join(lines).strip()


# ── Interface ─────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "vintage graphic tee under $30",
    "90s track jacket in size M",
    "flowy midi skirt under $40",
    "black combat boots size 8",
    "butterfly print size XS under $40",      # Triggers the size-drop fallback
    "designer ballgown size XXS under $5",    # Deliberate no-results test
]

def build_interface():
    """
    Build and return the Gradio Blocks interface for FitFindr.

    Constructs the full UI: a query textbox, wardrobe radio selector, submit
    button, three output Textbox panels, and a set of example queries. Both the
    button click and the textbox Enter key are wired to handle_query(). The
    returned demo object is launched by the __main__ guard.

    Returns:
        gr.Blocks: The assembled Gradio interface, ready for .launch().
    """
    with gr.Blocks(title="FitFindr") as demo:
        gr.Markdown("""
# FitFindr 🛍️
Find secondhand pieces & get outfit ideas based on your wardrobe.
Describe what you're looking for. Include size and price if you want to filter.
If nothing matches, FitFindr automatically loosens your filters and tells you
what it adjusted. Tick **Remember my preferences** to save your size, budget,
and wardrobe across sessions, or pick **Saved profile** to reuse them.
        """)

        with gr.Row():
            query_input = gr.Textbox(
                label="What are you looking for?",
                placeholder="e.g. vintage graphic tee under $30, size M",
                lines=2,
                scale=3,
            )
            wardrobe_choice = gr.Radio(
                choices=[
                    "Example wardrobe",
                    "Empty wardrobe (new user)",
                    "Saved profile",
                ],
                value="Example wardrobe",
                label="Wardrobe",
                scale=1,
            )

        remember_check = gr.Checkbox(
            value=False,
            label="💾 Remember my preferences (size, budget & wardrobe) for next time",
        )

        submit_btn = gr.Button("Find it", variant="primary")

        with gr.Row():
            listing_output = gr.Textbox(
                label="🛍️ Top listing found",
                lines=8,
                interactive=False,
            )
            outfit_output = gr.Textbox(
                label="👗 Outfit idea",
                lines=8,
                interactive=False,
            )
            fitcard_output = gr.Textbox(
                label="✨ Your fit card",
                lines=8,
                interactive=False,
            )

        gr.Examples(
            examples=[[q, "Example wardrobe", False] for q in EXAMPLE_QUERIES],
            inputs=[query_input, wardrobe_choice, remember_check],
            label="Try these queries",
        )

        submit_btn.click(
            fn=handle_query,
            inputs=[query_input, wardrobe_choice, remember_check],
            outputs=[listing_output, outfit_output, fitcard_output],
        )
        query_input.submit(
            fn=handle_query,
            inputs=[query_input, wardrobe_choice, remember_check],
            outputs=[listing_output, outfit_output, fitcard_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
