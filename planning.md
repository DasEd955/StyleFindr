# FitFindr — planning.md

> Complete this document before writing any implementation code.
> Your spec and agent diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Your planning.md will be reviewed as part of your submission.
> Update it before starting any stretch features.

---

## Tools

List every tool your agent will use. For each tool, fill in all four fields.
You must have at least 3 tools. The three required tools are listed — add any additional tools below them.

### Tool 1: search_listings

**What it does:**
<!-- Describe what this tool does in 1–2 sentences -->
This tool searches the listings dataset for clothing items (secondhand) that match the user's query for description, size, and budget. Results are ranked & filtered by relevance priority, such that the agent can choose the best match. 

**Input parameters:**
<!-- List each parameter, its type, and what it represents -->
- `description` (str): Keywords that describe the desired item (e.g., "Designer Cashmere Sweater")
- `size` (str): The user's preferred clothing size
- `max_price` (float): The highest price that a user is willing to pay

**What it returns:**
<!-- Describe the return value — what fields does a result contain? -->
'search_listings' returns a list of matching listing dictionaries sorted in relevance order. Each listing will contain:
- `id`
- `title`
- `description`
- `category`
- `style_tags`
- `size`
- `condition`
- `price`
- `colors`
- `material`
- `brand`
- `platform`

**What happens if it fails or returns nothing:**
<!-- What should the agent do if no listings match? -->
In the edge case where no listings are found or correlated, the tool will return an empty list & an explanatory message. The agent will inform the user that no matches were found & make a suggestion that the user broadens their search criterion (e.g., higher budget, different size or color, more general description). The planning loop will terminate and does not proceed to the outfit generation stage. 

---

### Tool 2: suggest_outfit

**What it does:**
<!-- Describe what this tool does in 1–2 sentences -->
This tools analyzes a selected secondhand item alongside the user's existing wardrobe & generates >= 1 complete outfit recommendations that match the item's style. 

**Input parameters:**
<!-- List each parameter, its type, and what it represents -->
- `new_item` (dict): The selected most relevant listings returned from `search_listing`
- `wardrobe` (dict): The user's wardrobe data containing available clothing articles & accessories. 

**What it returns:**
<!-- Describe the return value -->
A dictionary that contains:
- `outfit_description` (str): Description of the suggested outfit
- `matching_items` (list): A list of extant items in the user's wardrobe that matches the suggested item
- `style_reasoning` (str): Description of reasoning behind the outfit suggestion
- `style_category` (str): Generalized style category that the outfit falls within 

**What happens if it fails or returns nothing:**
<!-- What should the agent do if the wardrobe is empty or no outfit can be suggested? -->
In the edge case that the wardrobe is empty, the tool will generate styling recommendations using generic staple pieces instead of wardrobe-specific intelligent recommendations. In the event where the output cannot be successfully generated, the agent will explain the issue & ask the user for greater wardrobe context. 

---

### Tool 3: create_fit_card

**What it does:**
<!-- Describe what this tool does in 1–2 sentences -->
This tool creates a short, social media-esque caption based on the selected secondhand item via `search_listings` & the generated outfit recommendation via `suggest_outfit`.

**Input parameters:**
<!-- List each parameter, its type, and what it represents -->
- `outfit` (dict): The outfit recommendation returned by `suggest_outfit`
- `new_item`(dict): The selected listing returned by `search_listings`

**What it returns:**
<!-- Describe the return value -->
A dictionary that contains:
- `fit_card_text` (str): The final social media-esque caption generated for the outfit. Primary output shown to the user & sounds like a natural social media caption post. 
- `style_tags` (list): A list of keywords that describes the overall aesthetic of the outfit, i.e., ["vintage", "luxury"]
- `caption_tone` (str): The tone used when generating the fit card, e.g., "casual", "sophisticated", "confident".

**What happens if it fails or returns nothing:**
<!-- What should the agent do if the outfit data is incomplete? -->
In the edge case where outfit information is incomplete, the tool will create a simplified fit card utilizing only the item information. In the event where the generationf fails entirely, the agent will inform the user & display the given outfit recommendation without the fit card. 

---

### Additional Tools (if any) [TBD]

<!-- Copy the block above for any tools beyond the required three -->

---

## Planning Loop

**How does your agent decide which tool to call next?**
<!-- Describe the logic your planning loop uses. What does it look at? What conditions change its behavior? How does it know when it's done? -->
1. Receive the user's query request & extract the given item description, size, and maximum price as the given input parameters. 
2. Call `search_listings(description, size, max_price)`
3. Check the search results:
- If the results list is empty, return an error message to the user & stop. 
- If results are found (exist), select the highest-ranked listing in terms of relevance & store it as `selected_item`.
4. Retrieve the user's existing wardrobe data & call `suggest_outfit(selected_item, wardrobe)`.
5. Check the outfit result:
- If a valid outfit is returned, store it as `selected_outfit`.
- If the wardrobe is empty, generate a fallback outfit utilizing "common" wardrobe staples. 
- If outfit generation fails entirely, notify the user & stop. 
6. Call `create_fit_card(selected_outfit, selected_item)`.
7. Display the selected item, outfit recommendation, and fit card to the user. 
8. End the session once all outputs have been successfully generated. 

- KEY NOTE: The above is a logic description of the planning loop when a single-pass successfully occurs. In practice, the loop changes behavior based on tool output & does not automatically proceed if a required result is missing. 

---

## State Management

**How does information from one tool get passed to the next?**
<!-- Describe how your agent stores and accesses state within a session. What data is tracked? How is it passed between tool calls? -->
The agent maintains a session state dictionary throughout the interaction. This tracked state includes:
- `user_query`
- `description`
- `size`
- `max_price`
- `search_results`
- `selected_item`
- `wardrobe`
- `selected_outfit`
- `fit_card`
- `error_message`

---

## Error Handling

For each tool, describe the specific failure mode you're handling and what the agent does in response.

| Tool | Failure mode | Agent response |
|------|-------------|----------------|
| search_listings | No results match the query | |
| suggest_outfit | Wardrobe is empty | |
| create_fit_card | Outfit input is missing or incomplete | |

---

## Architecture

<!-- Draw a diagram of your agent showing how the components connect:
     User input → Planning Loop → Tools (search_listings, suggest_outfit, create_fit_card)
                                                                          ↕
                                                                   State / Session
     Show what triggers each tool, how state flows between them, and where error paths branch off.
     ASCII art, a Mermaid diagram (https://mermaid.js.org/syntax/flowchart.html), or an embedded
     sketch are all fine. You'll share this diagram with an AI tool when asking it to implement
     the planning loop and each individual tool. -->

---

## AI Tool Plan

<!-- For each part of the implementation below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, your agent diagram)
     - What you expect it to produce
     - How you'll verify the output matches your spec before moving on

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Tool 1 spec (inputs, return value, failure mode) and ask it to implement
     search_listings() using load_listings() from the data loader — then test it against 3 queries
     before trusting it" is a plan. -->

**Milestone 3 — Individual tool implementations:**

**Milestone 4 — Planning loop and state management:**

---

## A Complete Interaction (Step by Step)

<!-- Write out what a full user interaction looks like from start to finish — tool call by tool call. Use a specific example query. -->

StyleFindr is an agentic AI toolset that helps users find secondhand clothing recommendations, determine how new additions fit in with their existing wardrobe, and generate a comprehensive outfit description. The application utilizes a planning loop to decide which tool to call next based on previous results. The procedure searches listings first, then suggests outfits for a selected item, and creates a fit card, with variable flexibility as aforementioned. If a tool fails or returns a non-useful result, the agent will communicate the issue, and intelligently ask for clarification or retry with a fallback strategy, rather than continuing with invlid data or silent failure. 

**Example user query:** "I'm looking for a vintage graphic tee under $30. I mostly wear baggy jeans and chunky sneakers. What's out there and how would I style it?"

**Step 1:**
<!-- What does the agent do first? Which tool is called? With what input? -->

**Step 2:**
<!-- What happens next? What was returned from step 1? What tool is called now? -->

**Step 3:**
<!-- Continue until the full interaction is complete -->

**Final output to user:**
<!-- What does the user actually see at the end? -->
