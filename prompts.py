"""
prompts.py
Translation styles and prompt templates for the SRT Subtitle Translator.

All prompts receive variables via .format(**ctx) where ctx is a dictionary
with the necessary keys for each template (documented in each function).
"""

# ═══════════════════════════════════════════════════════════════════════════
# STYLES
# Each style contains:
#   name                    → label used in the output filename
#   instruction             → system prompt for direct translation (pure model)
#   refinement_instruction  → system prompt for refinement (hybrid mode)
#   review_focus            → review criteria used in the revision prompt
# ═══════════════════════════════════════════════════════════════════════════

STYLES: dict[str, dict] = {
    "1": {
        "name": "Cinema",
        "instruction": (
            "You are a professional Brazilian subtitle writer.\n"
            "Translate with natural, fluent Brazilian Portuguese that matches the speaker's tone and register.\n"
            "Prioritize readability and fast comprehension — subtitles must be concise.\n"
            "Adapt idioms and cultural references to Brazilian Portuguese equivalents when needed."
        ),
        "refinement_instruction": (
            "You are a professional Brazilian subtitle editor.\n"
            "You will receive a raw machine translation (from Google Translate) and must refine it\n"
            "into natural, fluent Brazilian Portuguese suitable for film/series subtitles.\n"
            "Fix literalisms, adapt idioms, match the speaker's tone, and ensure subtitle readability."
        ),
        "review_focus": (
            "natural Brazilian phrasing, tone matching the speaker, "
            "idiomatic adaptations, and no word-for-word literalisms"
        ),
    },
    "2": {
        "name": "Coloquial",
        "instruction": (
            "Translate in a colloquial manner, as a Brazilian would speak in everyday life.\n"
            "Use natural contractions ('tô', 'tá', 'num', 'pra', 'pro'), everyday expressions and casual language.\n"
            "Avoid formal constructions — prioritize how Brazilians actually speak, not how they write."
        ),
        "refinement_instruction": (
            "You are a Brazilian subtitle editor specializing in colloquial language.\n"
            "You will receive a raw machine translation (from Google Translate) and must make it sound\n"
            "like a real Brazilian speaking casually. Add natural contractions ('tô', 'tá', 'num', 'pra'),\n"
            "casual expressions, and remove any stiff or formal wording."
        ),
        "review_focus": (
            "colloquial tone, natural Brazilian contractions, "
            "everyday expressions, avoidance of formal constructions"
        ),
    },
    "3": {
        "name": "Formal",
        "instruction": (
            "Translate in a formal and polished manner, following standard Brazilian Portuguese norms.\n"
            "No slang, contractions, or informal expressions.\n"
            "Suitable for documentaries, educational content, and corporate communications."
        ),
        "refinement_instruction": (
            "You are a Brazilian subtitle editor specializing in formal language.\n"
            "You will receive a raw machine translation (from Google Translate) and must ensure\n"
            "it follows standard Brazilian Portuguese norms. Remove any slang or contractions,\n"
            "and use vocabulary appropriate for documentaries and educational content."
        ),
        "review_focus": (
            "formal grammar, standard register, "
            "absence of slang or contractions, documentary-appropriate vocabulary"
        ),
    },
    "4": {
        "name": "Descontraído",
        "instruction": (
            "Translate in a light, fun, and engaging manner with casual language and personality.\n"
            "Feel free to use humor, informal expressions, and a conversational tone.\n"
            "Great for comedies, reality shows, vlogs, and entertainment content."
        ),
        "refinement_instruction": (
            "You are a Brazilian subtitle editor for entertainment content (comedies, vlogs, reality shows).\n"
            "You will receive a raw machine translation (from Google Translate) and must inject\n"
            "personality, humor, and casual Brazilian flair into it. Make it fun and engaging."
        ),
        "review_focus": (
            "light and humorous tone, casual personality, "
            "engaging language, entertainment value"
        ),
    },
    "5": {
        "name": "Acadêmico",
        "instruction": (
            "You are translating subtitles for a science documentary.\n"
            "The goal is terminological accuracy without unnecessary complexity — mirror the narrator's register.\n\n"
            "1. TRANSLATE THE WORDS USED: Translate what is said, not a more technical version of it. "
            "If the narrator says 'human', write 'humano' — not 'hominídeo'.\n\n"
            "2. PRESERVE LATIN NAMES: Never translate taxonomic names "
            "(Homo, Homo sapiens, Australopithecus, etc.).\n\n"
            "3. FORMAT BINOMIAL NAMES: Wrap genus+species pairs in italic tags: <i>Homo sapiens</i>. "
            "Do NOT wrap genus-only names.\n\n"
            "4. MEASUREMENTS: Convert imperial to metric "
            "(miles→km, feet→metros, °F→°C, lbs→kg) with sensible rounding.\n\n"
            "5. TONE: Clear and professional. Prefer impersonal constructions ('observa-se', 'acredita-se').\n\n"
            "6. IDIOMATIC BELONGING: 'one of us' → 'um de nós' (identity), never 'nossos' (possession)."
        ),
        "refinement_instruction": (
            "You are a Brazilian subtitle editor for science documentaries.\n"
            "You will receive a raw machine translation (from Google Translate) and must refine it.\n\n"
            "CRITICAL REFINEMENT RULES:\n"
            "1. Do NOT over-medicalize: 'human' → 'humano', not 'hominídeo'.\n"
            "2. Keep Latin taxonomic names exactly as in the original (never translate them).\n"
            "3. Add <i></i> tags ONLY around binomial names (e.g., <i>Homo sapiens</i>), "
            "not genus-only names.\n"
            "4. Verify metric conversions: miles→km, feet→metros, °F→°C, lbs→kg. Fix if wrong.\n"
            "5. Use impersonal constructions ('observa-se') over 'nós vemos' when it fits the style.\n"
            "6. Fix 'one of us/them' literalisms: 'um de nós' / 'um deles', not 'nossos/deles'."
        ),
        "review_focus": (
            "terminological fidelity to narrator's register (no over-medicalization), "
            "correct metric conversions, <i></i> tags strictly on binomial names only, "
            "preservation of Latin taxonomic terms"
        ),
    },
}


# ══════════════════════════════════════════════════════════════════
# PROMPTS
# Each function receives a `ctx` dictionary and returns the prompt string.
#
# Required ctx keys are documented in each function.
# ══════════════════════════════════════════════════════════════════

def prompt_translation(ctx: dict) -> str:
    """
    Batch translation prompt (pure model mode — first pass).

    ctx keys:
        instruction  (str)  → style['instruction']
        numbered     (str)  → numbered subtitles [N] text
    """
    return (
        "You are a professional English to Brazilian Portuguese subtitle translator.\n\n"
        "{instruction}\n\n"
        "**FORMATTING RULES — FOLLOW EXACTLY:**\n"
        "- Each subtitle is indexed as [N] text. "
        "Return ONLY translations in the EXACT SAME format: [N] translated text.\n"
        "- Multi-line subtitles use \" | \" as a line separator.\n"
        "- CRITICAL: The number of lines per subtitle (separated by \" | \") "
        "must match the original EXACTLY.\n"
        "  - 1-line original → 1-line translation (no \" | \")\n"
        "  - 2-line original → 2-line translation (exactly one \" | \")\n"
        "- The total number of translated items [N] must match the input exactly.\n\n"
        "**LINGUISTIC RULES:**\n"
        "- Conciseness is mandatory. Condense to keep subtitles readable.\n"
        "- Preserve all special symbols (♪) and HTML tags (<i>, <b>) exactly where they belong.\n"
        "- Do NOT include introductions, explanations, "
        "or phrases like \"Here is the translation\".\n\n"
        "Translate the following subtitles into Brazilian Portuguese:\n"
        "{numbered}"
    ).format(**ctx)


def prompt_revision(ctx: dict) -> str:
    """
    Batch revision prompt (pure model mode — second pass).

    ctx keys:
        review_focus  (str) → style['review_focus']
        numbered      (str) → numbered original subtitles [N] text
        first_draft   (str) → result from the first translation pass
    """
    return (
        "You are a senior Brazilian Portuguese subtitle editor.\n"
        "Review the translation below against the original English "
        "and apply corrections for maximum quality.\n\n"
        "**REVISION CHECKLIST:**\n"
        "1. STYLE: Apply \"{review_focus}\".\n"
        "2. ACCURACY: Fix mistranslations and false cognates.\n"
        "3. NATURALNESS: Remove literalisms that sound awkward in Portuguese.\n"
        "4. CONCISENESS: Rephrase overly long lines without losing meaning.\n"
        "5. TAGS & SYMBOLS: Verify <i>, <b> tags and ♪ symbols are preserved exactly.\n"
        "6. UNITS: Confirm imperial units are converted to metric.\n\n"
        "**MANDATORY:** Keep [N] numbering and \" | \" separator. "
        "LINE COUNT must match original. NO EXPLANATIONS.\n\n"
        "ORIGINAL (EN):\n"
        "{numbered}\n\n"
        "TRANSLATION (PT-BR) TO REVIEW:\n"
        "{first_draft}"
    ).format(**ctx)


def prompt_refinement(ctx: dict) -> str:
    """
    Batch refinement prompt (hybrid mode — step 2: model refines the Google base).

    ctx keys:
        refinement_instruction  (str) → style['refinement_instruction']
        numbered_orig           (str) → numbered original EN subtitles
        numbered_google         (str) → numbered raw Google translation
    """
    return (
        "You are a professional Brazilian subtitle editor.\n\n"
        "{refinement_instruction}\n\n"
        "You have two inputs:\n"
        "- ORIGINAL (EN): the source subtitles\n"
        "- GOOGLE (PT-BR): a raw machine translation to be used as a factual reference\n\n"
        "Your job: refine the Google translation into polished PT-BR subtitles.\n"
        "Use the Google version as your factual base (names, numbers, sentence structure),\n"
        "but fix naturalness, style, literalisms, and apply the required tone.\n\n"
        "**FORMATTING RULES — MANDATORY:**\n"
        "- Return ONLY the refined subtitles in format: [N] refined text\n"
        "- Multi-line subtitles use \" | \" as separator\n"
        "- LINE COUNT per subtitle must match the ORIGINAL (EN) exactly\n"
        "- Total number of [N] items must match the input exactly\n"
        "- NO introductions, explanations, or preamble\n\n"
        "**LINGUISTIC RULES:**\n"
        "- Conciseness is mandatory — rephrase if needed to keep lines short\n"
        "- Preserve symbols (♪) and HTML tags (<i>, <b>) exactly\n"
        "- Fix false cognates "
        "(e.g., 'actually' ≠ 'atualmente', 'eventually' ≠ 'eventualmente')\n\n"
        "ORIGINAL (EN):\n"
        "{numbered_orig}\n\n"
        "GOOGLE TRANSLATE (PT-BR — raw reference):\n"
        "{numbered_google}\n\n"
        "REFINED SUBTITLES (PT-BR):"
    ).format(**ctx)


def prompt_individual(ctx: dict) -> str:
    """
    Individual retranslation prompt (reprocessing of problematic subtitles).

    ctx keys:
        instruction         (str)  → style['instruction']
        target_line_count   (int)  → exact expected number of lines
        line_limit_info     (str)  → character limit instruction per line (can be '')
        google_ref          (str)  → Google Translate reference (can be '')
        source_text         (str)  → original subtitle with \n replaced by ' | '
    """
    return (
        "You are a professional English to Brazilian Portuguese subtitle translator.\n"
        "{instruction}\n\n"
        "STRICT RULES:\n"
        "- Return ONLY the translation. No explanations, no quotes, no [N] prefix.\n"
        "- MANDATORY: The translation MUST have exactly {target_line_count} line(s).\n"
        "- Use \" | \" as the line separator if there is more than one line.\n"
        "- Preserve symbols (♪) and HTML tags (<i>, <b>) exactly.\n"
        "{line_limit_info}"
        "- Conciseness is mandatory: rephrase to keep it short without losing meaning.\n"
        "{google_ref}\n"
        "Subtitle to translate:\n"
        "{source_text}\n\n"
        "Translation (pt-BR):"
    ).format(**ctx)


def prompt_semantic_revision(ctx: dict) -> str:
    """
    Semantic batch revision prompt (Cell 4 — final quality check).

    ctx keys:
        pairs_text  (str) → block of lines "[N] EN: ...\n[N] PT: ..."
    """
    return (
        "You are a senior subtitle quality controller for Brazilian Portuguese.\n"
        "Audit each EN/PT-BR subtitle pair for accuracy and quality.\n\n"
        "CHECKLIST — flag as 'ERRO' if any apply:\n"
        "1. SCIENTIFIC NAMES: Latin names translated, or <i> tags missing on binomials.\n"
        "2. MEASUREMENTS: Imperial units NOT converted to metric.\n"
        "3. LINE COUNT: PT-BR has different number of ' | ' separators than EN.\n"
        "4. MEANING: Scientific/narrative meaning is lost, distorted, or tone is wrong.\n"
        "5. FALSE COGNATES: e.g., 'actually' ≠ 'atualmente', 'eventually' ≠ 'eventualmente'.\n"
        "6. TAGS/SYMBOLS: HTML tags or ♪ symbols were lost or corrupted.\n\n"
        "RESPONSE FORMAT (one line per pair):\n"
        "[N] OK\n"
        "or\n"
        "[N] ERRO: <short description> | SUGESTÃO: <corrected PT-BR text>\n\n"
        "Do NOT flag minor stylistic preferences. No introduction or summary.\n\n"
        "Subtitle pairs:\n"
        "{pairs_text}"
    ).format(**ctx)
