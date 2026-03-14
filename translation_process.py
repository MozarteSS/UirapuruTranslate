"""
translation_process.py
Helper functions for the SRT Subtitle Translator with Ollama.
Styles and prompt templates are in prompts.py.
"""

import re
import time
import json
import os
import unicodedata
from collections import deque

import ollama
from deep_translator import GoogleTranslator

from prompts import (
    STYLES,
    prompt_translation,
    prompt_revision,
    prompt_refinement,
    prompt_individual,
    prompt_semantic_revision,
)


# ══════════════════════════════════════════════════════════════════
# PARSE / BUILD SRT
# ══════════════════════════════════════════════════════════════════

def parse_srt(content: str) -> list[dict]:
    """Reads SRT content and returns a list of blocks."""
    content = re.sub(r'\r\n|\r', '\n', content)
    blocks = []
    for block in re.split(r'\n{2,}', content.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        timecode = lines[1].strip()
        if '-->' not in timecode:
            continue
        text = '\n'.join(lines[2:])
        blocks.append({'index': index, 'timecode': timecode, 'text': text})
    return blocks


def build_srt(blocks: list[dict]) -> str:
    """Rebuilds SRT content from a list of blocks."""
    parts = [f"{b['index']}\n{b['timecode']}\n{b['text']}" for b in blocks]
    return '\n\n'.join(parts) + '\n'


def read_srt_file(filepath: str) -> str:
    """Reads an SRT file trying multiple encodings."""
    for enc in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252'):
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return re.sub(r'\r\n|\r', '\n', f.read())
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    raise FileNotFoundError(f'Could not read: {filepath}')


def parse_srt_dict(filepath: str) -> dict:
    """Returns a dictionary {index: text} from an SRT file."""
    content = read_srt_file(filepath)
    blocks = {}
    for block in re.split(r'\n{2,}', content.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        if '-->' not in lines[1]:
            continue
        blocks[idx] = '\n'.join(lines[2:])
    return blocks


def parse_srt_full(filepath: str) -> list[dict]:
    """Returns a full list of blocks (index, timecode, text) from an SRT file."""
    content = read_srt_file(filepath)
    blocks = []
    for block in re.split(r'\n{2,}', content.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        if '-->' not in lines[1]:
            continue
        blocks.append({'index': idx, 'timecode': lines[1].strip(), 'text': '\n'.join(lines[2:])})
    return blocks


def build_srt_from_dict(full_blocks: list[dict], corrected: dict) -> str:
    """Rebuilds SRT applying selective corrections from a dict."""
    parts = []
    for b in full_blocks:
        text = corrected.get(b['index'], b['text'])
        parts.append(f"{b['index']}\n{b['timecode']}\n{text}")
    return '\n\n'.join(parts) + '\n'


# ══════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════

def is_empty_block(text: str) -> bool:
    """Returns True if the block has no translatable text (e.g., only ♪ or numbers)."""
    clean = re.sub(r'<[^>]+>', '', text).strip()
    return not clean or bool(re.fullmatch(r'[\W\d\s♪]+', clean))


def clean_result(text: str) -> str:
    """Removes common model response artifacts (e.g., ** → <i>, unwanted prefixes)."""
    text = re.sub(r'\*\*?(.*?)\*\*?', r'<i>\1</i>', text)
    text = text.replace('<i> ', '<i>').replace(' </i>', '</i>')
    text = re.sub(r'^(Tradução|Translation|PT-BR|Result|Revised)[:\s]+', '', text, flags=re.IGNORECASE)
    return text.strip()


def strip_tags(text: str) -> str:
    """Removes HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text).strip()


# ══════════════════════════════════════════════════════════════════
# GOOGLE TRANSLATE
# ══════════════════════════════════════════════════════════════════

_gt = GoogleTranslator(source='en', target='pt')
_GT_CHAR_LIMIT = 4500  # safety margin below the 5000-char limit


def google_translate_batch(texts: list[str]) -> list[str]:
    """
    Translates a list of texts using Google Translate.
    Uses separators to preserve 1-to-1 correspondence with the fewest
    possible calls. Falls back to individual calls if the batch exceeds the limit.
    """
    def _translate_chunk(chunk_texts):
        parts   = [t.replace('\n', ' | ') for t in chunk_texts]
        payload = '|||SEP|||'.join(parts)
        if len(payload) > _GT_CHAR_LIMIT:
            return None
        translated = _gt.translate(payload)
        if not translated:
            return None
        translated_parts = re.split(r'\|{3}SEP\|{3}', translated)
        if len(translated_parts) == len(chunk_texts):
            return [p.strip() for p in translated_parts]
        return None

    result = _translate_chunk(texts)
    if result is not None:
        return result

    # Individual fallback
    translations = []
    for t in texts:
        try:
            trad = _gt.translate(t.replace('\n', ' | ')) or t
            translations.append(trad)
        except Exception:
            translations.append(t)
    return translations


# ══════════════════════════════════════════════════════════════════
# BATCH TRANSLATION — PURE MODEL
# ══════════════════════════════════════════════════════════════════

def translate_batch_model(
    texts: list[str],
    style: dict,
    translation_model: str,
    revision_model: str,
    batch_size: int,
) -> tuple[str, str]:
    """Translation + auto-revision entirely by the local model."""
    numbered = '\n'.join(
        f'[{i+1}] {t.replace(chr(10), " | ")}' for i, t in enumerate(texts)
    )

    resp_trans = ollama.chat(
        model=translation_model,
        messages=[{'role': 'user', 'content': prompt_translation({
            'instruction': style['instruction'],
            'numbered'   : numbered,
        })}],
        options={'temperature': 0.2, 'num_ctx': 8192,
                 'num_predict': 256 * batch_size, 'repeat_penalty': 1.1},
    )
    first_draft = resp_trans.message.content.strip()

    resp_rev = ollama.chat(
        model=revision_model,
        messages=[{'role': 'user', 'content': prompt_revision({
            'review_focus': style['review_focus'],
            'numbered'    : numbered,
            'first_draft' : first_draft,
        })}],
        options={'temperature': 0.1, 'num_ctx': 8192,
                 'num_predict': 256 * batch_size, 'repeat_penalty': 1.1},
    )

    return clean_result(resp_rev.message.content), numbered


# ══════════════════════════════════════════════════════════════════
# BATCH TRANSLATION — HYBRID MODE
# ══════════════════════════════════════════════════════════════════

def translate_batch_hybrid(
    texts: list[str],
    style: dict,
    translation_model: str,
    revision_model: str,
    batch_size: int,
) -> tuple[str, str]:
    """
    Hybrid mode:
      1. Google Translate generates the factual base (names, numbers, structure)
      2. Local model refines to the chosen style and corrects literalisms
    """
    try:
        base_google = google_translate_batch(texts)
    except Exception as e:
        print(f'\n  ⚠️  Google Translate failed ({e}) — using pure model for this batch')
        return translate_batch_model(texts, style, translation_model, revision_model, batch_size)

    numbered_orig   = '\n'.join(f'[{i+1}] {t.replace(chr(10), " | ")}' for i, t in enumerate(texts))
    numbered_google = '\n'.join(f'[{i+1}] {g}' for i, g in enumerate(base_google))

    resp = ollama.chat(
        model=translation_model,
        messages=[{'role': 'user', 'content': prompt_refinement({
            'refinement_instruction': style['refinement_instruction'],
            'numbered_orig'         : numbered_orig,
            'numbered_google'       : numbered_google,
        })}],
        options={'temperature': 0.15, 'num_ctx': 8192,
                 'num_predict': 256 * batch_size, 'repeat_penalty': 1.1},
    )

    return clean_result(resp.message.content), numbered_orig


# ══════════════════════════════════════════════════════════════════
# MAIN DISPATCHER
# ══════════════════════════════════════════════════════════════════

def translate_batch(
    texts: list[str],
    style: dict,
    translation_model: str,
    revision_model: str,
    batch_size: int,
    hybrid_mode: bool,
) -> tuple[dict[int, str], list[int]]:
    """
    Calls the appropriate mode and unpacks the numbered indices from the result.

    Returns:
        translations  → dict {1-based_index: translated_text} with what the model returned
        missing       → list of 1-based indices absent from the response
    """
    if hybrid_mode:
        result_raw, _ = translate_batch_hybrid(texts, style, translation_model, revision_model, batch_size)
    else:
        result_raw, _ = translate_batch_model(texts, style, translation_model, revision_model, batch_size)

    translations: dict[int, str] = {}
    for line in result_raw.splitlines():
        m = re.match(r'^\[(\d+)\]\s*(.*)', line)
        if m:
            translations[int(m.group(1))] = re.sub(r'\s*\|\s*', '\n', m.group(2))

    missing = [i + 1 for i in range(len(texts)) if (i + 1) not in translations]
    return translations, missing


def translate_batch_with_retry(
    batch: list[str],
    style: dict,
    translation_model: str,
    revision_model: str,
    batch_size: int,
    hybrid_mode: bool,
    chars_threshold: int,
    expansion_factor: float,
    max_retries: int = 3,
) -> list[str]:
    """
    Runs translate_batch with smart retries:
      - Accumulates correct translations between attempts.
      - Only re-sends missing indices to the model.
      - After exhausting retries, reprocesses each still-missing subtitle
        individually — the batch never falls back to the original as a whole.
    """
    # Start with originals as a safe fallback
    result = list(batch)
    # 1-based indices still needing translation
    pending = list(range(1, len(batch) + 1))

    for attempt in range(max_retries):
        # Build sub-batch with only pending items
        pending_texts = [batch[i - 1] for i in pending]
        try:
            translations, local_missing = translate_batch(
                pending_texts, style, translation_model,
                revision_model, batch_size, hybrid_mode,
            )
            # Apply received translations (local index → global index)
            for local_idx, trad in translations.items():
                global_idx = pending[local_idx - 1]
                result[global_idx - 1] = trad

            # Update pending to those still missing
            pending = [pending[i - 1] for i in local_missing]

            if not pending:
                return result  # ✅ all translated

            print(f'\n  ⚠️  Missing indices: {pending} — attempt {attempt + 1}/{max_retries}')

        except Exception as e:
            wait = 2 ** attempt
            print(f'\n  ⚠️  Attempt {attempt + 1}/{max_retries} failed: {e}. Waiting {wait}s...')
            time.sleep(wait)

    # After all retries, reprocess persistently missing subtitles individually
    if pending:
        print(f'\n  🔧 {len(pending)} subtitle(s) missing after {max_retries} retries '
              f'— reprocessing individually...')
        for global_idx in pending:
            orig = batch[global_idx - 1]
            if is_empty_block(orig):
                continue  # empty blocks don't need translation
            new = reprocess_individual(
                global_idx - 1, orig, style, translation_model,
                hybrid_mode, chars_threshold, expansion_factor,
            )
            result[global_idx - 1] = new
            status = '✅' if new.strip() != orig.strip() else '⚠️  (original kept)'
            print(f'     ↳ Subtitle #{global_idx} {status}')

    return result


# ══════════════════════════════════════════════════════════════════
# VALIDATION AND INDIVIDUAL REPROCESSING
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# MISALIGNMENT DETECTION — WITHOUT LLM
# Resolves two problems caused when the model "skips" or "merges"
# subtitles at the start of a batch, shifting all the rest:
#   1. Consecutive duplicates  → Jaccard similarity between adjacent translations
#   2. Out-of-position anchors → Latin names, numbers, and proper nouns that
#                                 appear the same in EN and PT serve as GPS
#                                 to detect which slot orig[i]'s content ended up in
# ══════════════════════════════════════════════════════════════════

def _normalize_anchor(s: str) -> str:
    """Removes accents and converts to lowercase for cross-language comparison."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s.lower())
        if unicodedata.category(c) != 'Mn'
    )


def _extract_anchors(text: str) -> set[str]:
    """
    Extracts terms that appear nearly identical in EN and PT:
      - Scientific / Latin names  (Homo erectus, Neanderthalensis…)
      - Digit numbers              (2, 3.5, 200,000…)
      - Simple proper nouns        (Africa/África → normalized the same)

    Removes accents BEFORE the regex so 'África' matches 'Africa'.
    Uses text without HTML tags to avoid false positives.
    """
    no_tags = re.sub(r'<[^>]+>', '', text)
    # Remove accents → 'África' becomes 'Africa', 'gênero' becomes 'genero', etc.
    normalized = ''.join(
        c for c in unicodedata.normalize('NFD', no_tags)
        if unicodedata.category(c) != 'Mn'
    )
    anchors: set[str] = set()
    # Digit numbers
    anchors.update(re.findall(r'\b\d[\d,\.]*\b', normalized))
    # Capitalized sequences (proper names / Latin / toponyms)
    anchors.update(re.findall(r'\b[A-Z][a-z]{2,}(?:\s[a-z]{3,})?\b', normalized))
    # Already lowercase (accents already removed above)
    return {a.lower() for a in anchors if len(a) > 3}


def detect_consecutive_duplicates(
    translated: list[str],
    threshold: float = 0.65,
) -> list[int]:
    """
    Returns 0-based indices of subtitles very similar to the previous one.

    Uses Jaccard similarity over a normalized bag-of-words.
    threshold=0.65 means 65% of words in common — adjust if needed.

    Example:
        trad[2] = "a deixar a África."
        trad[3] = "para deixar a África."   ← Jaccard ≈ 0.75 → duplicate
    """
    duplicates = []
    for i in range(1, len(translated)):
        a = set(_normalize_anchor(strip_tags(translated[i - 1])).split())
        b = set(_normalize_anchor(strip_tags(translated[i])).split())
        # Ignore music blocks / very short ones
        if len(a) < 2 or len(b) < 2:
            continue
        jaccard = len(a & b) / len(a | b)
        if jaccard >= threshold:
            duplicates.append(i)
    return duplicates


def detect_shift_by_anchors(
    originals: list[str],
    translated: list[str],
    min_votes: int = 2,
) -> int | None:
    """
    Detects systematic misalignment between originals and translated
    by summing anchor votes found out of position.

    Returns the most-voted delta (e.g., -1 = translation advanced 1 position)
    or None if there is not enough confidence.

    Vote logic:
        For each orig[i] with anchors, scan all trad[j].
        If anchors from orig[i] appear in trad[j] → vote for delta = j - i.
        The most-voted delta indicates the dominant shift.
        If delta=0 dominates (majority aligned) → no global shift.
    """
    votes: dict[int, int] = {}
    for i, orig in enumerate(originals):
        orig_anchors = _extract_anchors(orig)
        if not orig_anchors:
            continue
        for j, trad in enumerate(translated):
            overlap = len(orig_anchors & _extract_anchors(trad))
            if overlap:
                delta = j - i
                votes[delta] = votes.get(delta, 0) + overlap

    if not votes:
        return None

    best = max(votes, key=votes.get)
    if best == 0:
        return None  # already aligned

    # Only report if delta=0 does NOT dominate and has enough votes
    aligned_votes = votes.get(0, 0)
    if votes[best] < min_votes or aligned_votes >= votes[best]:
        return None

    return best


def correct_shift(
    originals: list[str],
    translated: list[str],
    delta: int,
) -> tuple[list[str], list[int]]:
    """
    Applies shift correction and returns 0-based indices that were left
    without a translation and need individual reprocessing.

    delta < 0  →  translation "advanced" (content of orig[i] went to trad[i+delta])
                  Shifts translated forward; first |delta| slots revert
                  to originals for reprocessing.

    delta > 0  →  translation "lagged"
                  Shifts translated backward; last delta slots revert
                  to originals for reprocessing.

    Example with delta = -1 and batch of 7 subtitles:
        Before: trad = [T0, T1, T2, T3, T4, T5, T6]
                       (T0 = content of orig[1], T1 = orig[2], …)
        After:  corr  = [orig[0], T0, T1, T2, T3, T4, T5]
                to_reprocess = [0]
    """
    n = len(originals)
    corrected = list(translated)
    to_reprocess: list[int] = []

    if delta < 0:
        d = abs(delta)
        corrected[d:] = translated[:n - d]
        for i in range(d):
            corrected[i] = originals[i]
            to_reprocess.append(i)
    else:
        corrected[:n - delta] = translated[delta:]
        for i in range(n - delta, n):
            corrected[i] = originals[i]
            to_reprocess.append(i)

    return corrected, to_reprocess


def detect_and_correct_misalignment(
    originals: list[str],
    translated: list[str],
    global_offset: int = 0,
) -> tuple[list[str], list[int]]:
    """
    Full misalignment detection and correction pipeline without LLM.

    Step 1 — Duplicates: detects subtitles nearly identical to the previous one.
              Before voting on shift, blanks them out so a repeated anchor
              (e.g., "Africa" in the duplicate) doesn't wrongly vote for
              delta=0 and tie with the real shift.

    Step 2 — Global shift: after decontaminating the list, votes by anchors
              to detect systematic misalignment and applies correct_shift.

    Step 3 — Residual duplicates: any remaining duplicates (not fixed by
              the shift) are flagged for individual reprocessing.

    Returns (corrected_list, indices_to_reprocess_individually).
    """
    corrected = list(translated)
    to_reprocess: list[int] = []

    # ── Step 1: detect duplicates before voting on shift ───────────
    # Temporarily blank them out so repeated anchors don't
    # contaminate the delta=0 vote and tie with the real shift.
    initial_duplicates = detect_consecutive_duplicates(corrected)
    translated_for_voting = list(corrected)
    for i in initial_duplicates:
        translated_for_voting[i] = ''   # invisible to the shift voter

    delta = detect_shift_by_anchors(originals, translated_for_voting)
    if delta is not None:
        print(f'\n  🔀 Misalignment of {delta:+d} position(s) detected — correcting...')
        corrected, orphans = correct_shift(originals, corrected, delta)
        to_reprocess.extend(orphans)
        for i in orphans:
            num = global_offset + i + 1
            print(f'     ↳ Subtitle #{num} left without translation → reprocessing queue')

    duplicates = detect_consecutive_duplicates(corrected)
    new_dup = [i for i in duplicates if i not in to_reprocess]
    if new_dup:
        print(f'\n  🔁 {len(new_dup)} residual duplicate(s) → reprocessing queue')
        for i in new_dup:
            num = global_offset + i + 1
            print(f'     ↳ Subtitle #{num} (high Jaccard with #{num - 1})')
            corrected[i] = originals[i]   # revert to original for clean reprocessing
        to_reprocess.extend(new_dup)

    return corrected, sorted(set(to_reprocess))


def detect_problems(
    originals: list[str],
    translated: list[str],
    chars_threshold: int,
    expansion_factor: float,
) -> list[tuple]:
    """Detects structural and size issues in translations."""
    problems = []
    stopwords_en = {
        'the', 'a', 'an', 'is', 'it', 'in', 'on', 'at', 'to', 'of', 'and', 'or', 'but',
        'for', 'with', 'this', 'that', 'was', 'are', 'be', 'have', 'has', 'had', 'do',
        'did', 'not', 'he', 'she', 'we', 'you', 'they', 'i', 'my', 'your', 'his', 'her',
        'its', 'our', 'their',
    }

    for i, (orig, trad) in enumerate(zip(originals, translated)):
        orig_clean = orig.strip().lower()
        trad_clean = trad.strip().lower()

        if not trad_clean:
            if not is_empty_block(orig):
                problems.append((i, 'empty'))
            continue

        orig_lines_n = len(orig.strip().splitlines())
        trad_lines_n = len(trad.strip().splitlines())
        if trad_lines_n > orig_lines_n and trad_lines_n > 1:
            problems.append((i, f'extra lines ({orig_lines_n}→{trad_lines_n})'))
            continue

        symbols_only = re.fullmatch(r'[\W\d\s]+', orig_clean)
        if not symbols_only and orig_clean == trad_clean:
            problems.append((i, 'not translated'))
            continue

        en_words   = set(re.findall(r'\b[a-zA-Z]+\b', trad))
        orig_words = set(re.findall(r'\b[a-zA-Z]+\b', orig))
        en_in_trad = en_words & stopwords_en
        if len(orig_words) > 3 and len(en_in_trad) >= 3 and len(en_words) > len(orig_words) * 0.5:
            problems.append((i, 'possible English'))
            continue

        if ' | ' in trad:
            problems.append((i, 'visible | separator'))
            continue

        orig_lines_split = orig.split('\n')
        trad_lines_split = re.sub(r'<[^>]+>', '', trad).split('\n')
        for j, lt in enumerate(trad_lines_split):
            lo       = orig_lines_split[j] if j < len(orig_lines_split) else max(orig_lines_split, key=len)
            lo_clean = re.sub(r'<[^>]+>', '', lo)
            if len(lo_clean) > chars_threshold:
                limit = int(len(lo_clean) * expansion_factor)
                if len(lt) > limit:
                    problems.append((i, f'long line ({len(lt)} > {limit} chars)'))
                    break

    return problems


def reprocess_individual(
    global_idx: int,
    original_text: str,
    style: dict,
    translation_model: str,
    hybrid_mode: bool,
    chars_threshold: int,
    expansion_factor: float,
) -> str:
    """Retranslates a single subtitle with precise line and character limit control."""
    if is_empty_block(original_text):
        return original_text

    orig_lines        = original_text.replace(' | ', '\n').split('\n')
    target_line_count = len(orig_lines)

    def _line_limit(lo):
        lo_clean = re.sub(r'<[^>]+>', '', lo)
        return int(len(lo_clean) * expansion_factor) if len(lo_clean) > chars_threshold else None

    line_limit_info = ''
    for i, lim in enumerate([_line_limit(l) for l in orig_lines]):
        if lim:
            line_limit_info += f'- Line {i+1}: aim for max {lim} characters.\n'

    source_text = original_text.replace(chr(10), ' | ')

    google_ref = ''
    if hybrid_mode:
        try:
            google_trad = _gt.translate(source_text)
            if google_trad:
                google_ref = f'\nGoogle Translate reference: {google_trad}\n'
        except Exception:
            pass

    p = prompt_individual({
        'instruction'      : style['instruction'],
        'target_line_count': target_line_count,
        'line_limit_info'  : line_limit_info,
        'google_ref'       : google_ref,
        'source_text'      : source_text,
    })

    best_result = None  # best translated result even if line count is off
    for attempt in range(3):
        try:
            resp = ollama.chat(
                model=translation_model,
                messages=[{'role': 'user', 'content': p}],
                options={'temperature': 0.0, 'num_ctx': 2048,
                         'num_predict': 256, 'repeat_penalty': 1.1},
            )
            result = clean_result(resp.message.content).replace(' | ', '\n')
            if result.strip() and result.strip() != original_text.strip() and best_result is None:
                best_result = result
            if len(result.split('\n')) != target_line_count:
                continue
            return result
        except Exception:
            time.sleep(2 ** attempt)

    # Prefer any actual translation over returning the untranslated original
    return best_result if best_result is not None else original_text


def validate_and_correct(
    originals: list[str],
    translated: list[str],
    style: dict,
    translation_model: str,
    hybrid_mode: bool,
    chars_threshold: int,
    expansion_factor: float,
    global_offset: int = 0,
) -> tuple[list[str], int]:
    """Detects problems and individually reprocesses problematic subtitles."""

    # ── Pre-validation: misalignment without LLM ──────────────────
    pre_corrected, pre_reprocess = detect_and_correct_misalignment(
        originals, translated, global_offset=global_offset,
    )
    n_pre = 0
    for local_idx in pre_reprocess:
        subtitle_num = global_offset + local_idx + 1
        orig = originals[local_idx]
        print(f'     ↳ Subtitle #{subtitle_num} (misalignment) → reprocessing...')
        new_text = reprocess_individual(
            local_idx, orig, style, translation_model,
            hybrid_mode, chars_threshold, expansion_factor,
        )
        if new_text.strip() != orig.strip():
            pre_corrected[local_idx] = new_text
            n_pre += 1
            print(f'       ✅ Corrected')
        else:
            print(f'       ⚠️  No change — kept')

    # ── Normal structural validation ──────────────────────────────
    problems = detect_problems(originals, pre_corrected, chars_threshold, expansion_factor)
    if not problems and n_pre == 0:
        return pre_corrected, 0

    if not problems:
        return pre_corrected, n_pre

    print(f'\n  🔍 {len(problems)} structural problem(s) detected — reprocessing individually...')
    corrected   = pre_corrected
    n_corrected = n_pre

    for local_idx, reason in problems:
        subtitle_num = global_offset + local_idx + 1
        orig = originals[local_idx]
        trad = translated[local_idx]
        print(f'     ↳ Subtitle #{subtitle_num} ({reason}) → reprocessing...')
        new_text = reprocess_individual(
            local_idx, orig, style, translation_model,
            hybrid_mode, chars_threshold, expansion_factor,
        )

        if reason.startswith('long line'):
            chars_before = max((len(l) for l in re.sub(r'<[^>]+>', '', trad).split('\n')), default=0)
            chars_after  = max((len(l) for l in re.sub(r'<[^>]+>', '', new_text).split('\n')), default=0)
            orig_lines   = orig.split('\n')
            limit = next(
                (int(len(re.sub(r'<[^>]+>', '', lo)) * expansion_factor)
                 for lo in orig_lines if len(re.sub(r'<[^>]+>', '', lo)) > chars_threshold),
                None,
            )
            lim_str = str(limit) if limit else '?'
            if limit and chars_after <= limit:
                corrected[local_idx] = new_text; n_corrected += 1
                print(f'       ✅ Corrected: {chars_before} → {chars_after} chars (limit: {lim_str})')
            elif chars_after < chars_before:
                corrected[local_idx] = new_text; n_corrected += 1
                print(f'       ⚠️  Partial improvement: {chars_before} → {chars_after} chars (limit: {lim_str})')
            else:
                print(f'       ❌ Not corrected: kept (limit: {lim_str})')

        elif reason == 'empty':
            if new_text.strip() and re.sub(r'<[^>]+>', '', new_text).strip():
                corrected[local_idx] = new_text; n_corrected += 1
                print(f'       ✅ Corrected: "{new_text.replace(chr(10), " | ")[:50]}"')
            else:
                print(f'       ⚠️  Still empty — kept')

        elif reason.startswith('extra lines'):
            n_orig = len(orig.strip().splitlines())
            if len(new_text.strip().splitlines()) <= n_orig:
                corrected[local_idx] = new_text; n_corrected += 1
                print(f'       ✅ Corrected: {len(trad.splitlines())} → {len(new_text.splitlines())} lines')
            else:
                print(f'       ⚠️  Still has extra lines — kept')

        else:
            if new_text.strip() != trad.strip():
                corrected[local_idx] = new_text; n_corrected += 1
                print(f'       ✅ Corrected')
            else:
                print(f'       ⚠️  No change — kept')

    print(f'  ✔️  {n_corrected}/{len(problems)} corrected')
    return corrected, n_corrected


# ══════════════════════════════════════════════════════════════════
# QUALITY ESTIMATION — TRANSQUEST
# ══════════════════════════════════════════════════════════════════

_tq_model = None


def _load_tq_model():
    """Lazily loads the TransQuest multilingual DA quality estimation model."""
    global _tq_model
    if _tq_model is not None:
        return _tq_model
    try:
        from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel
    except ImportError:
        raise ImportError(
            '❌ transquest not installed. Run: pip install transquest'
        )
    print('📥 Loading TransQuest model (first run downloads ~440 MB)...')
    _tq_model = MonoTransQuestModel(
        'xlmroberta',
        'TransQuest/monotransquest-da-multilingual',
        num_labels=1,
        use_cuda=False,
        args={
            'silent': True,
            'use_multiprocessing': False,
            'use_multiprocessing_for_evaluation': False,
        },
    )
    print('✅ TransQuest model ready.')
    return _tq_model


def score_translations_qe(
    pairs: list[tuple],
    threshold: float = 50.0,
) -> list[tuple]:
    """
    Scores EN→PT subtitle pairs using TransQuest (Direct Assessment, 0–100).
    Returns a list of (idx, en, pt, score) for pairs whose score < threshold.

    Typical thresholds:
      60 → strict  (catches mediocre translations)
      50 → balanced (recommended)
      40 → lenient  (only catches clearly wrong translations)
    """
    model = _load_tq_model()
    input_pairs = [
        [
            strip_tags(en).replace('\n', ' '),
            strip_tags(pt).replace('\n', ' '),
        ]
        for _, en, pt in pairs
    ]
    predictions, _ = model.predict(input_pairs)
    return [
        (idx, en, pt, float(score))
        for (idx, en, pt), score in zip(pairs, predictions)
        if score < threshold
    ]


# ══════════════════════════════════════════════════════════════════
# FINAL REVIEW
# ══════════════════════════════════════════════════════════════════

def line_too_long(
    orig: str,
    trad: str,
    chars_threshold: int,
    expansion_factor: float,
) -> tuple[bool, str]:
    """Checks if any translation line exceeds the dynamic character limit."""
    orig_lines = orig.split('\n')
    trad_lines = re.sub(r'<[^>]+>', '', trad).split('\n')
    for j, lt in enumerate(trad_lines):
        lo       = orig_lines[j] if j < len(orig_lines) else max(orig_lines, key=len)
        lo_clean = re.sub(r'<[^>]+>', '', lo)
        if len(lo_clean) > chars_threshold:
            limit = int(len(lo_clean) * expansion_factor)
            if len(lt) > limit:
                return True, f'long line ({len(lt)} > {limit} chars)'
    return False, ''


def review_semantic_batch(pairs: list[tuple], revision_model: str) -> dict:
    """
    Sends a batch of EN/PT pairs to the model for semantic error checking.
    Returns dict {position: {status, problem, suggestion}}.
    """
    lines = []
    for i, (idx, orig, trad) in enumerate(pairs):
        lines.append(f'[{i+1}] EN: {orig.replace(chr(10), " | ")}')
        lines.append(f'[{i+1}] PT: {trad.replace(chr(10), " | ")}')

    resp = ollama.chat(
        model=revision_model,
        messages=[{'role': 'user', 'content': prompt_semantic_revision({
            'pairs_text': '\n'.join(lines),
        })}],
        options={'temperature': 0.0, 'num_ctx': 8192, 'num_predict': 2048},
    )
    raw = resp.message.content.strip()

    results = {}
    pattern = re.compile(r'^\[(\d+)\]\s*(OK|ERRO:\s*(.*?)\s*\|\s*SUGESTÃO:\s*(.*))', re.IGNORECASE)
    for line in raw.splitlines():
        m = pattern.match(line)
        if m:
            n = int(m.group(1))
            if m.group(2).upper().startswith('OK'):
                results[n] = {'status': 'OK', 'problem': '', 'suggestion': ''}
            else:
                results[n] = {
                    'status'    : 'ERRO',
                    'problem'   : m.group(3).strip() if m.group(3) else 'Unspecified error',
                    'suggestion': m.group(4).strip() if m.group(4) else '',
                }
    return results


# ══════════════════════════════════════════════════════════════════
# MAIN TRANSLATION LOOP
# ══════════════════════════════════════════════════════════════════

def translate_file(
    input_file: str,
    chosen_style: str,
    translation_model: str,
    revision_model: str,
    batch_size: int,
    chars_threshold: int,
    expansion_factor: float,
    use_google_as_base: bool,
    google_available: bool,
) -> str:
    """
    Runs the full translation pipeline for an SRT file.
    Returns the output file path.
    """
    if chosen_style not in STYLES:
        raise ValueError(f'❌ Invalid style: "{chosen_style}". Choose from: {list(STYLES.keys())}')

    style        = STYLES[chosen_style]
    hybrid_mode  = use_google_as_base and google_available

    if use_google_as_base and not google_available:
        print('⚠️  Google Translate unavailable — using pure model mode.')

    content = None
    for enc in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252'):
        try:
            with open(input_file, 'r', encoding=enc) as f:
                content = f.read()
            print(f'📄 File read with encoding: {enc}')
            break
        except FileNotFoundError:
            raise FileNotFoundError(f'❌ File not found: {input_file}')
        except UnicodeDecodeError:
            continue

    if content is None:
        raise ValueError('❌ Could not read file with any supported encoding.')

    blocks = parse_srt(content)
    total  = len(blocks)
    if total == 0:
        raise ValueError('❌ No subtitles found. Check if it is a valid .srt file.')

    total_batches   = (total + batch_size - 1) // batch_size
    base            = os.path.splitext(input_file)[0]
    output_filename = f'{base}_{style["name"]}_pt-BR.srt'
    checkpoint_file = f'{base}_checkpoint.json'

    all_texts = [b['text'] for b in blocks]
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        all_translated = checkpoint.get('translated', [])
        start_from = len(all_translated)
        if start_from >= total:
            print('✅ Complete checkpoint found. Skipping translation.')
        else:
            print(f'♻️  Checkpoint found. Resuming from batch {start_from // batch_size + 1}/{total_batches}...')
    except (FileNotFoundError, json.JSONDecodeError):
        all_translated = []
        start_from = 0

    mode_str = '🔄 Hybrid' if hybrid_mode else '🤖 Pure model'
    print(f'\n📊 {total} subtitles | 📦 {total_batches} batches of {batch_size}')
    print(f'🌐 Model: {translation_model} | Style: {style["name"]} | Mode: {mode_str}\n')

    errors = total_long = total_corrections = 0
    start_time   = time.time()
    recent_times = deque(maxlen=3)

    for i in range(start_from, total, batch_size):
        batch       = all_texts[i:i + batch_size]
        batch_num   = (i // batch_size) + 1
        batch_start = time.time()

        real_indices = [j for j, t in enumerate(batch) if not is_empty_block(t)]
        real_texts   = [batch[j] for j in real_indices]

        real_results = (
            translate_batch_with_retry(
                real_texts, style, translation_model, revision_model,
                batch_size, hybrid_mode, chars_threshold, expansion_factor,
            )
            if real_texts else list(batch)
        )

        result = list(batch)
        for pos, j in enumerate(real_indices):
            if pos < len(real_results):
                result[j] = real_results[pos]

        if result == batch:
            errors += len(batch)
        else:
            problems_before  = detect_problems(batch, result, chars_threshold, expansion_factor)
            total_long      += sum(1 for _, m in problems_before if 'long' in m)
            result, n_corr   = validate_and_correct(
                batch, result, style, translation_model,
                hybrid_mode, chars_threshold, expansion_factor, global_offset=i,
            )
            total_corrections += n_corr

        all_translated.extend(result)

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({'translated': all_translated}, f, ensure_ascii=False)

        recent_times.append(time.time() - batch_start)
        eta = int(sum(recent_times) / len(recent_times) * (total_batches - batch_num))
        pct = int(batch_num / total_batches * 100)
        bar = '█' * (pct // 2) + '░' * (50 - pct // 2)
        print(f'\r[{bar}] {pct}% | batch {batch_num}/{total_batches} | ⏱ ~{eta}s remaining', end='', flush=True)

    for i, block in enumerate(blocks):
        block['text'] = all_translated[i]

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(build_srt(blocks))

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    total_time = int(time.time() - start_time)
    mins, secs = divmod(total_time, 60)

    print()
    print(f'\n{"═"*52}')
    print(f'  📊 FINAL REPORT')
    print(f'  ✅ Translated  : {total - errors}/{total} subtitles')
    print(f'  🔄 Mode        : {"Hybrid (Google + model)" if hybrid_mode else "Pure model"}')
    if total_corrections: print(f'  🔍 Corrected   : {total_corrections} (individual reprocessing)')
    if total_long:        print(f'  📏 Long lines  : {total_long} detected')
    if errors:            print(f'  ❌ Errors      : {errors} (original kept)')
    print(f'  ⏱  Total time  : {mins}m {secs}s')
    if total_time > 0:    print(f'  ⚡ Average     : {total_time/total:.1f}s per subtitle')
    print(f'  💾 Saved to    : {output_filename}')
    print(f'{"═"*52}')

    return output_filename


# ══════════════════════════════════════════════════════════════════
# FINAL REVIEW PIPELINE
# ══════════════════════════════════════════════════════════════════

def review_and_correct_file(
    input_file: str,
    output_filename: str,
    revision_model: str,
    translation_model: str,
    style: dict,
    hybrid_mode: bool,
    chars_threshold: int,
    expansion_factor: float,
    review_batch_size: int = 20,
    save_report: bool = True,
    use_transquest: bool = False,
    qe_threshold: float = 50.0,
) -> str:
    """
    Runs the final review: programmatic + semantic verification,
    optional TransQuest quality estimation, automatic correction
    and report generation.
    Returns the corrected file path.

    use_transquest  — enable TransQuest QE pass (requires: pip install transquest)
    qe_threshold    — DA score threshold (0–100); pairs below this are flagged.
                      50 = balanced (recommended), 40 = lenient, 60 = strict.
    """
    print('📂 Loading files...')
    orig_blocks      = parse_srt_dict(input_file)
    trad_blocks      = parse_srt_dict(output_filename)
    full_trad_blocks = parse_srt_full(output_filename)

    orig_indices   = set(orig_blocks.keys())
    trad_indices   = set(trad_blocks.keys())
    common_indices = sorted(orig_indices & trad_indices)
    only_orig      = sorted(orig_indices - trad_indices)
    only_trad      = sorted(trad_indices - orig_indices)

    print(f'   Original  : {len(orig_blocks)} blocks')
    print(f'   Translated: {len(trad_blocks)} blocks')
    if only_orig: print(f'   ⚠️  Only in original   : {only_orig}')
    if only_trad: print(f'   ⚠️  Only in translation: {only_trad}')

    print('\n🔎 Programmatic checks...')
    prog_errors = []
    for idx in common_indices:
        orig = orig_blocks[idx]
        trad = trad_blocks[idx]
        if is_empty_block(orig):
            continue
        orig_clean = strip_tags(orig).lower()
        trad_clean = strip_tags(trad).lower()
        if not trad_clean:
            prog_errors.append((idx, 'empty', orig, trad, '')); continue
        if not re.fullmatch(r'[\W\d\s]+', orig_clean) and orig_clean == trad_clean:
            prog_errors.append((idx, 'not translated', orig, trad, '')); continue
        lo_n = len(orig.strip().splitlines())
        lt_n = len(trad.strip().splitlines())
        if lt_n > lo_n and lt_n > 1:
            prog_errors.append((idx, f'extra lines ({lo_n}→{lt_n})', orig, trad, '')); continue
        is_long, desc = line_too_long(orig, trad, chars_threshold, expansion_factor)
        if is_long:
            prog_errors.append((idx, desc, orig, trad, ''))
        if ' | ' in trad:
            prog_errors.append((idx, 'visible | separator', orig, trad, ''))
    print(f'   {len(prog_errors)} structural problem(s) found')

    print(f'\n🤖 Semantic review with {revision_model}...')
    prog_error_indices = set(e[0] for e in prog_errors)
    llm_indices        = [i for i in common_indices if i not in prog_error_indices and not is_empty_block(orig_blocks[i])]
    total_llm_batches  = (len(llm_indices) + review_batch_size - 1) // review_batch_size
    print(f'   {len(llm_indices)} blocks | {total_llm_batches} batches of {review_batch_size}\n')

    llm_errors   = []
    review_start = time.time()

    for b, start in enumerate(range(0, len(llm_indices), review_batch_size)):
        batch_indices = llm_indices[start:start + review_batch_size]
        pairs = [(idx, orig_blocks[idx], trad_blocks[idx]) for idx in batch_indices]
        try:
            results = review_semantic_batch(pairs, revision_model)
        except Exception as e:
            print(f'\n  ⚠️  Batch {b+1} failed: {e}'); continue
        for i, (idx, orig, trad) in enumerate(pairs):
            r = results.get(i + 1)
            if r and r['status'] == 'ERRO':
                llm_errors.append((idx, r['problem'], orig, trad, r['suggestion']))
        pct = int((b + 1) / total_llm_batches * 100)
        bar = '█' * (pct // 2) + '░' * (50 - pct // 2)
        eta = int((time.time() - review_start) / (b + 1) * (total_llm_batches - b - 1))
        print(f'\r[{bar}] {pct}% | batch {b+1}/{total_llm_batches} | ⏱ ~{eta}s remaining', end='', flush=True)

    print()

    # ── Optional: TransQuest Quality Estimation pass ───────────────
    qe_errors = []
    if use_transquest:
        print(f'\n🎯 Quality Estimation — TransQuest (threshold: {qe_threshold})...')
        already_flagged = prog_error_indices | set(e[0] for e in llm_errors)
        qe_candidates = [
            (idx, orig_blocks[idx], trad_blocks[idx])
            for idx in common_indices
            if idx not in already_flagged and not is_empty_block(orig_blocks[idx])
        ]
        try:
            low_quality = score_translations_qe(qe_candidates, threshold=qe_threshold)
            for idx, en, pt, score in low_quality:
                qe_errors.append((idx, f'low QE score ({score:.1f} < {qe_threshold})', en, pt, ''))
            print(f'   {len(low_quality)} block(s) flagged by QE (score < {qe_threshold})')
        except Exception as e:
            print(f'   ⚠️  TransQuest check failed: {e}')

    all_errors = sorted(prog_errors + llm_errors + qe_errors, key=lambda x: x[0])
    print(f'\n🔧 Correcting {len(all_errors)} problem(s)...')
    corrected      = {}
    correction_log = []

    for idx, reason, orig, trad, suggestion in all_errors:
        if reason == 'visible | separator':
            new_text = re.sub(r'\s*\|\s*', '\n', trad)
            corrected[idx] = new_text
            correction_log.append((idx, reason, trad, new_text, 'programmatic'))
            print(f'   #{idx} pipe → corrected programmatically'); continue

        if suggestion:
            new_text = re.sub(r'\s*\|\s*', '\n', re.sub(r'\*(.+?)\*', r'<i>\1</i>', suggestion))
            corrected[idx] = new_text
            correction_log.append((idx, reason, trad, new_text, 'LLM suggestion'))
            print(f'   #{idx} {reason} → LLM suggestion applied'); continue

        print(f'   #{idx} {reason} → reprocessing individually...')
        new_text = reprocess_individual(
            idx, orig, style, translation_model,
            hybrid_mode, chars_threshold, expansion_factor,
        )

        if reason.startswith('long line'):
            resolved = not line_too_long(orig, new_text, chars_threshold, expansion_factor)[0]
        elif reason == 'empty':
            resolved = bool(strip_tags(new_text).strip())
        elif reason.startswith('extra lines'):
            resolved = len(new_text.strip().splitlines()) <= len(orig.strip().splitlines())
        else:
            resolved = new_text.strip() != trad.strip()

        if resolved:
            corrected[idx] = new_text
            correction_log.append((idx, reason, trad, new_text, 'individual reprocessing'))
            print(f'     ✅ corrected')
        else:
            correction_log.append((idx, reason, trad, trad, 'not corrected — kept'))
            print(f'     ⚠️  could not correct — kept')

    base_corr   = os.path.splitext(output_filename)[0]
    output_corr = f'{base_corr}_corrected.srt'
    with open(output_corr, 'w', encoding='utf-8') as f:
        f.write(build_srt_from_dict(full_trad_blocks, corrected))

    total_blocks = len(common_indices)
    total_errors = len(all_errors)
    total_corr   = sum(1 for *_, m in correction_log if 'not corrected' not in m)
    total_failed = total_errors - total_corr
    ok_rate      = (total_blocks - total_failed) / total_blocks * 100 if total_blocks else 0
    review_time  = int(time.time() - review_start)
    mins_r, secs_r = divmod(review_time, 60)
    n_pipe   = sum(1 for _, m, *_ in all_errors if m == 'visible | separator')
    n_llm    = sum(1 for *_, m in correction_log if m == 'LLM suggestion')
    n_reproc = sum(1 for *_, m in correction_log if m == 'individual reprocessing')

    lr = [
        '═' * 60, '  📋 REVIEW AND CORRECTION REPORT', '═' * 60,
        f'  Original file     : {input_file}',
        f'  Translated file   : {output_filename}',
        f'  Corrected file    : {output_corr}',
        f'  Revision model    : {revision_model}',
        f'  Blocks reviewed   : {total_blocks}',
        f'  ❌ Problems       : {total_errors}',
        f'     ↳ Structural   : {len(prog_errors)}',
        f'     ↳ Semantic     : {len(llm_errors)}',
    ]
    if use_transquest:
        lr.append(f'     ↳ QE (TransQ.)  : {len(qe_errors)} (threshold: {qe_threshold})')
    lr += [
        f'  🔧 Corrected      : {total_corr}',
        f'     ↳ Pipe (prog.) : {n_pipe}',
        f'     ↳ LLM suggest. : {n_llm}',
        f'     ↳ Reprocessed  : {n_reproc}',
    ]
    if total_failed:
        lr.append(f'  ⚠️  Not corrected  : {total_failed} (original kept)')
    lr += [
        f'  ✅ Final quality   : {ok_rate:.1f}%',
        f'  ⏱  Review time    : {mins_r}m {secs_r}s',
        '═' * 60,
    ]

    if correction_log:
        completed     = [(a, b, c, d, e) for a, b, c, d, e in correction_log if 'not corrected' not in e]
        not_completed = [(a, b, c, d, e) for a, b, c, d, e in correction_log if 'not corrected' in e]
        if completed:
            lr += ['', f'  ✅ COMPLETED CORRECTIONS ({len(completed)})', '─' * 60]
            for idx, reason, before, after, method in completed:
                lr += [
                    '', f'  Block #{idx} — {reason}',
                    f'  EN     : {orig_blocks.get(idx, "?").replace(chr(10), " | ")}',
                    f'  Before : {before.replace(chr(10), " | ")}',
                    f'  After  : {after.replace(chr(10), " | ")}',
                    f'  Method : {method}', '  ' + '─' * 56,
                ]
        if not_completed:
            lr += ['', f'  ⚠️  INCOMPLETE CORRECTIONS ({len(not_completed)})', '─' * 60]
            for idx, reason, before, _, method in not_completed:
                lr += [
                    '', f'  Block #{idx} — {reason}',
                    f'  EN     : {orig_blocks.get(idx, "?").replace(chr(10), " | ")}',
                    f'  Before : {before.replace(chr(10), " | ")}',
                    f'  Method : {method}', '  ' + '─' * 56,
                ]
    else:
        lr += ['', '  🎉 No problems found!']

    report = '\n'.join(lr)
    print('\n' + report)

    if save_report:
        report_path = f'{base_corr}_review.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f'\n💾 Report saved to  : {report_path}')

    print(f'💾 Corrected file   : {output_corr}')
    return output_corr
