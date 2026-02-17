"""
OpenRouter model catalog for translation and simplification tasks.

Central registry of recommended models, organized by tier.
Used by all frontends (CLI, web, GitHub Actions) as the single source of truth.
"""

# Estimated tokens for a typical Chinese novel (~100K characters):
# ~50K input tokens + ~100K output tokens for translation
TYPICAL_BOOK_INPUT_TOKENS = 50_000
TYPICAL_BOOK_OUTPUT_TOKENS = 100_000

MODELS = {
    # === FREE TIER ===
    "deepseek/deepseek-r1:free": {
        "name": "DeepSeek R1",
        "provider": "DeepSeek",
        "tier": "free",
        "input_price": 0,
        "output_price": 0,
        "context_window": 64_000,
        "chinese_quality": "excellent",
        "best_for": ["translation", "simplification"],
        "description": "Best free model for Chinese. Reasoning model by Chinese AI lab.",
        "note": "Rate limited: ~20 req/min. Data training opt-in.",
    },
    "qwen/qwen3-235b-a22b-thinking-2507:free": {
        "name": "Qwen3 235B",
        "provider": "Alibaba",
        "tier": "free",
        "input_price": 0,
        "output_price": 0,
        "context_window": 262_000,
        "chinese_quality": "excellent",
        "best_for": ["translation", "simplification"],
        "description": "Free frontier Chinese model. Native Chinese training.",
        "note": "Rate limited: ~20 req/min. Data training opt-in.",
    },

    # === STANDARD TIER ===
    "deepseek/deepseek-chat": {
        "name": "DeepSeek V3",
        "provider": "DeepSeek",
        "tier": "standard",
        "input_price": 0.19,
        "output_price": 0.87,
        "context_window": 164_000,
        "chinese_quality": "excellent",
        "best_for": ["translation", "simplification"],
        "description": "Best value for Chinese. Near-GPT-4o quality at 1/15th the cost.",
    },
    "qwen/qwen3-235b-a22b-2507": {
        "name": "Qwen3 235B Instruct",
        "provider": "Alibaba",
        "tier": "standard",
        "input_price": 0.07,
        "output_price": 0.10,
        "context_window": 262_000,
        "chinese_quality": "excellent",
        "best_for": ["translation", "simplification"],
        "description": "Cheapest frontier model. Native Chinese, 262K context.",
    },
    "google/gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash",
        "provider": "Google",
        "tier": "standard",
        "input_price": 0.30,
        "output_price": 2.50,
        "context_window": 1_000_000,
        "chinese_quality": "very_good",
        "best_for": ["translation"],
        "description": "1M context window. Great for chapter-level translation coherence.",
    },
    "openai/gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "provider": "OpenAI",
        "tier": "standard",
        "input_price": 0.15,
        "output_price": 0.60,
        "context_window": 128_000,
        "chinese_quality": "good",
        "best_for": ["translation"],
        "description": "Cheap and fast. Good for simple translations.",
    },
    "anthropic/claude-haiku-4-5": {
        "name": "Claude Haiku 4.5",
        "provider": "Anthropic",
        "tier": "standard",
        "input_price": 1.00,
        "output_price": 5.00,
        "context_window": 200_000,
        "chinese_quality": "very_good",
        "best_for": ["translation", "simplification"],
        "description": "Fast Claude model. Good instruction following for HSK tasks.",
    },

    # === PREMIUM TIER ===
    "anthropic/claude-sonnet-4-5": {
        "name": "Claude Sonnet 4.5",
        "provider": "Anthropic",
        "tier": "premium",
        "input_price": 3.00,
        "output_price": 15.00,
        "context_window": 200_000,
        "chinese_quality": "excellent",
        "best_for": ["translation", "simplification"],
        "description": "Top-tier instruction following. Excellent for HSK simplification.",
    },
    "openai/gpt-4o": {
        "name": "GPT-4o",
        "provider": "OpenAI",
        "tier": "premium",
        "input_price": 2.50,
        "output_price": 10.00,
        "context_window": 128_000,
        "chinese_quality": "excellent",
        "best_for": ["translation"],
        "description": "Highest lexical richness in translation benchmarks.",
    },
    "google/gemini-2.5-pro": {
        "name": "Gemini 2.5 Pro",
        "provider": "Google",
        "tier": "premium",
        "input_price": 1.25,
        "output_price": 10.00,
        "context_window": 1_000_000,
        "chinese_quality": "very_good",
        "best_for": ["translation"],
        "description": "1M context. Process entire books with full coherence.",
    },
    "anthropic/claude-opus-4-5": {
        "name": "Claude Opus 4.5",
        "provider": "Anthropic",
        "tier": "premium",
        "input_price": 5.00,
        "output_price": 25.00,
        "context_window": 200_000,
        "chinese_quality": "excellent",
        "best_for": ["translation", "simplification"],
        "description": "Most capable Claude. Best for nuanced literary translation.",
    },
}

# Tier defaults — the recommended model for each tier
TIER_DEFAULTS = {
    "free": "deepseek/deepseek-r1:free",
    "standard": "deepseek/deepseek-chat",
    "premium": "anthropic/claude-sonnet-4-5",
}

TIER_LABELS = {
    "free": "Free (OpenRouter)",
    "standard": "Standard",
    "premium": "Premium",
}


def estimate_book_cost(model_id):
    """Estimate cost to process a typical book (~50K in + 100K out tokens)."""
    model = MODELS.get(model_id)
    if not model:
        return None
    input_cost = model["input_price"] * TYPICAL_BOOK_INPUT_TOKENS / 1_000_000
    output_cost = model["output_price"] * TYPICAL_BOOK_OUTPUT_TOKENS / 1_000_000
    return round(input_cost + output_cost, 2)


def get_models_by_tier(tier):
    """Return list of (model_id, model_info) for a given tier."""
    return [
        (mid, info) for mid, info in MODELS.items()
        if info["tier"] == tier
    ]


def format_model_table():
    """Format the model catalog as a printable text table."""
    lines = []
    lines.append(
        f"{'Tier':<10} {'Model':<24} {'Provider':<10} "
        f"{'In $/M':>8} {'Out $/M':>9} {'Chinese':>10}  {'Best for'}"
    )
    lines.append("-" * 100)

    for tier in ("free", "standard", "premium"):
        for mid, info in get_models_by_tier(tier):
            default_mark = " *" if mid == TIER_DEFAULTS[tier] else ""
            name = info["name"] + default_mark
            lines.append(
                f"{tier:<10} {name:<24} {info['provider']:<10} "
                f"${info['input_price']:>6.2f} ${info['output_price']:>7.2f} "
                f"{info['chinese_quality']:>10}  {', '.join(info['best_for'])}"
            )

    lines.append("")
    lines.append("(* = tier default)")
    lines.append("No --model or --tier: uses Google Translate (free, no API key needed)")
    lines.append("Any OpenRouter model requires: OPENROUTER_API_KEY environment variable")
    return "\n".join(lines)
