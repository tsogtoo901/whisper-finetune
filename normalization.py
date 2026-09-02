"""
Text normalization for Mongolian ASR evaluation.

WHY THIS FILE EXISTS
--------------------
WER (word error rate) compares the model's output text against the human
transcript. If one side says "Сайн байна уу?" and the other says
"сайн байна уу", a naive comparison counts errors that aren't real.
So BOTH sides (reference transcript and model prediction) are passed
through this exact function before comparison — at every point in the
pipeline. This module is imported by train.py and evaluate_models.py;
there is deliberately no second copy of these rules anywhere.

THE RULES (documented per the spec, so the benchmark is defensible)
-------------------------------------------------------------------
1. Unicode NFC normalization (so visually-identical Cyrillic characters
   composed differently compare as equal).
2. Lowercase everything (Cyrillic lowercases correctly in Python).
3. Remove all punctuation (anything that is not a letter, digit, or
   whitespace becomes a space).
4. Digits are KEPT as digits. Documented decision: if a speaker says a
   number and the transcript has "25" while the model writes
   "хорин тав" (or vice versa), that counts as an error — for BOTH the
   stock model and the fine-tuned model equally, so the comparison
   stays fair. Numeric utterances are rare enough that this does not
   move the headline number meaningfully.
5. Collapse all whitespace runs to a single space; strip ends.

Changing any rule here changes the WER number. If a rule is ever
changed, ALL evaluations (baseline and fine-tuned) must be re-run.
"""

import re
import unicodedata

# \w in Unicode mode matches letters (incl. all Cyrillic: Өө, Үү, Ёё)
# and digits, plus underscore (handled separately below).
_NON_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_mn(text: str) -> str:
    """Normalize a Mongolian transcript or model prediction for scoring."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)
    text = text.replace("_", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


if __name__ == "__main__":
    # Quick self-check you can run with: python normalization.py
    samples = [
        "Сайн байна уу? Би 25 настай.",
        "  сайн   байна уу би 25 настай  ",
    ]
    outs = [normalize_mn(s) for s in samples]
    print(outs)
    assert outs[0] == outs[1], "Normalization self-check FAILED"
    print("Normalization self-check OK")
