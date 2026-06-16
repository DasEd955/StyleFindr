"""
app.py

Gradio interface for FitFindr. The layout and wiring are already set up —
your job is to fill in handle_query() so it calls run_agent() and maps
the session results to the three output panels.

Run with:
    python app.py

Then open the localhost URL shown in your terminal (usually http://localhost:7860,
but check your terminal — the port may differ).
"""

import gradio as gr

from agent import run_agent
from utils.data_loader import get_example_wardrobe, get_empty_wardrobe


# ── query handler ─────────────────────────────────────────────────────────────

def handle_query(user_query: str, wardrobe_choice: str) -> tuple[str, str, str]:
    """
    Called by Gradio when the user submits a query.

    Args:
        user_query:     The text the user typed into the search box.
        wardrobe_choice: Either "Example wardrobe" or "Empty wardrobe (new user)".

    Returns:
        A tuple of three strings:
            (listing_text, outfit_suggestion, fit_card)
        Each string maps to one of the three output panels in the UI.

    TODO:
        1. Guard against an empty query (return early with an error message).
        2. Select the wardrobe based on wardrobe_choice.
        3. Call run_agent() with the query and selected wardrobe.
        4. If session["error"] is set, return the error in the first panel
           and empty strings for the other two.
        5. Otherwise, format session["selected_item"] into a readable listing_text
           string and return it along with session["outfit_suggestion"] and
           session["fit_card"].
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
    """Render the selected listing dict into a readable panel string."""
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
    """Render the suggest_outfit dict into a readable panel string."""
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
    Render the create_fit_card dict into a readable panel string.

    fit_card is None when the caption step failed (a partial success): the
    outfit is still shown, and this panel notes the degraded result rather
    than crashing or hiding the rest of the output.
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
