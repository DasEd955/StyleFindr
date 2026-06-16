"""
app.py - Gradio interface for the FitFindr styling pipeline.

The UI is intentionally thin. All search, outfit-generation, and fit-card logic
lives in agent.run_agent() and the three tools in tools.py; this module only
renders the results across three output panels. handle_query() is the single
Gradio callback — it guards against empty input, selects the wardrobe, delegates
to run_agent(), and maps the session dict to (listing_text, outfit_text, fit_card_text).

The three private _format_* helpers convert raw dicts from the session into
human-readable strings for the Textbox panels. A per-session wardrobe is
chosen at submit time via a radio button (example vs. empty), so new-user and
existing-wardrobe flows are exercised through the same code path.

Run with:
    python app.py

Then open the localhost URL shown in your terminal (usually http://localhost:7860).
"""

import gradio as gr

from agent import run_agent
from utils.data_loader import get_example_wardrobe, get_empty_wardrobe


# ── query handler ─────────────────────────────────────────────────────────────

def handle_query(user_query: str, wardrobe_choice: str) -> tuple[str, str, str]:
    """
    Handle one Gradio submit event and return the three output panel strings.

    Guards against an empty query, selects the wardrobe based on the radio
    choice, delegates the full pipeline to run_agent(), and maps the resulting
    session dict to the three Textbox panels. On early termination (empty
    results or outfit failure) the error surfaces in the first panel and the
    other two are returned as empty strings.

    Args:
        user_query (str): The text the user typed into the search box.
        wardrobe_choice (str): Radio value — "Example wardrobe" or
            "Empty wardrobe (new user)".

    Returns:
        tuple[str, str, str]: (listing_text, outfit_text, fit_card_text).
            Each string maps to one of the three output Textbox panels.
            On error, only the first string is non-empty.
    """
    # 1. Guard against an empty query.
    if not user_query or not user_query.strip():
        return "Please enter what you're looking for.", "", ""

    # 2. Select the wardrobe based on the radio choice.
    wardrobe = (
        get_empty_wardrobe()
        if wardrobe_choice == "Empty wardrobe (new user)"
        else get_example_wardrobe()
    )

    # 3. Run the planning loop.
    session = run_agent(user_query.strip(), wardrobe)

    # 4. Early-termination paths (empty results, search/outfit failure) surface
    #    the error in the first panel and leave the other two empty.
    if session["error"]:
        return session["error"], "", ""

    # 5. Map the session dict to the three panel strings.
    return (
        _format_listing(session["selected_item"]),
        _format_outfit(session["outfit_suggestion"]),
        _format_fit_card(session["fit_card"]),
    )


# ── output formatting ───────────────────────────────────────────────────────

def _format_listing(item: dict) -> str:
    """
    Render a listing dict into a human-readable string for the top-listing panel.

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


# ── interface ─────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "vintage graphic tee under $30",
    "90s track jacket in size M",
    "flowy midi skirt under $40",
    "black combat boots size 8",
    "designer ballgown size XXS under $5",   # deliberate no-results test
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
Find secondhand pieces and get outfit ideas based on your wardrobe.
Describe what you're looking for — include size and price if you want to filter.
        """)

        with gr.Row():
            query_input = gr.Textbox(
                label="What are you looking for?",
                placeholder="e.g. vintage graphic tee under $30, size M",
                lines=2,
                scale=3,
            )
            wardrobe_choice = gr.Radio(
                choices=["Example wardrobe", "Empty wardrobe (new user)"],
                value="Example wardrobe",
                label="Wardrobe",
                scale=1,
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
            examples=[[q, "Example wardrobe"] for q in EXAMPLE_QUERIES],
            inputs=[query_input, wardrobe_choice],
            label="Try these queries",
        )

        submit_btn.click(
            fn=handle_query,
            inputs=[query_input, wardrobe_choice],
            outputs=[listing_output, outfit_output, fitcard_output],
        )
        query_input.submit(
            fn=handle_query,
            inputs=[query_input, wardrobe_choice],
            outputs=[listing_output, outfit_output, fitcard_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
