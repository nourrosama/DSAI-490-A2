"""
Hybrid custom tokenizer for conditional date generation.

Vocabulary (size = 35)
───────────────────────────────────────────────────
PAD = 0   BOS = 1   EOS = 2
Day tokens   : [MON]=3  [TUE]=4  [WED]=5  [THU]=6  [FRI]=7  [SAT]=8  [SUN]=9
Month tokens : [JAN]=10 [FEB]=11 [MAR]=12 [APR]=13 [MAY]=14 [JUN]=15
               [JUL]=16 [AUG]=17 [SEP]=18 [OCT]=19 [NOV]=20 [DEC]=21
Leap tokens  : [False]=22  [True]=23
Digits 0–9   : 24–33
Hyphen  '-'  : 34
───────────────────────────────────────────────────

Encoding rules
  • Day / Month / Leap  → single atomic token (brackets included)
  • Decade [XYZ]        → strip brackets, tokenise each digit individually
  • Date  dd-mm-yyyy    → fully character-level (each digit and '-' separately)
  • Every sequence is wrapped with BOS … EOS
"""

from __future__ import annotations

from typing import List


class DateTokenizer:
    """Hybrid tokenizer: atomic tokens for conditions, char-level for date output."""

    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2

    _DAY_TOKENS: List[str] = [
        "[MON]", "[TUE]", "[WED]", "[THU]", "[FRI]", "[SAT]", "[SUN]"
    ]
    _MONTH_TOKENS: List[str] = [
        "[JAN]", "[FEB]", "[MAR]", "[APR]", "[MAY]", "[JUN]",
        "[JUL]", "[AUG]", "[SEP]", "[OCT]", "[NOV]", "[DEC]",
    ]
    _LEAP_TOKENS: List[str] = ["[False]", "[True]"]

    def __init__(self) -> None:
        self._token2id: dict[str, int] = {}
        self._id2token: dict[int, str] = {}

        def _add(token: str, idx: int) -> None:
            self._token2id[token] = idx
            self._id2token[idx] = token

        _add("[PAD]", 0)
        _add("[BOS]", 1)
        _add("[EOS]", 2)

        for i, t in enumerate(self._DAY_TOKENS):      # 3 – 9
            _add(t, 3 + i)

        for i, t in enumerate(self._MONTH_TOKENS):    # 10 – 21
            _add(t, 10 + i)

        _add("[False]", 22)
        _add("[True]",  23)

        for d in range(10):                            # 24 – 33
            _add(str(d), 24 + d)

        _add("-", 34)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (always 35)."""
        return 35

    def encode(self, line: str) -> List[int]:
        """
        Encode a full line or a condition-only line.

        Full line:       "[MON] [DEC] [False] [196] 3-12-1962"
        Condition only:  "[MON] [DEC] [False] [196]"

        Returns a list of token IDs with BOS prepended and EOS appended.
        """
        parts = line.strip().split()
        ids: List[int] = [self.BOS_ID]

        # Day, Month, Leap – atomic tokens
        ids.append(self._token2id[parts[0]])   # day   e.g. [MON]
        ids.append(self._token2id[parts[1]])   # month e.g. [DEC]
        ids.append(self._token2id[parts[2]])   # leap  e.g. [False]

        # Decade – strip brackets, encode digit-by-digit
        decade: str = parts[3][1:-1]           # "[196]" → "196"
        for ch in decade:
            ids.append(self._token2id[ch])

        # Date (optional) – fully character-level
        if len(parts) > 4:
            for ch in parts[4]:
                ids.append(self._token2id[ch])

        ids.append(self.EOS_ID)
        return ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Reconstruct the original line from a list of token IDs.

        Drops PAD / BOS / EOS.
        Positions 0-2 in the clean sequence are condition atomics.
        Positions 3-5 are decade digits → re-wrapped in brackets.
        Remaining positions are date characters → joined directly.

        Returns either "day month leap [decade] date"
        or       "day month leap [decade]" when no date tokens are present.
        """
        clean = [t for t in token_ids
                 if t not in (self.PAD_ID, self.BOS_ID, self.EOS_ID)]
        if len(clean) < 6:
            return ""

        day_str    = self._id2token[clean[0]]
        month_str  = self._id2token[clean[1]]
        leap_str   = self._id2token[clean[2]]
        decade_str = "[" + "".join(self._id2token[clean[i]] for i in range(3, 6)) + "]"
        date_str   = "".join(self._id2token[clean[i]] for i in range(6, len(clean)))

        parts = [day_str, month_str, leap_str, decade_str]
        if date_str:
            parts.append(date_str)
        return " ".join(parts)

    def decode_date_only(self, token_ids: List[int]) -> str:
        """
        Reconstruct just the date string from a list that contains only
        date-portion token IDs (digits and hyphens).  Stops at the first
        PAD or EOS token encountered.
        """
        chars: List[str] = []
        for t in token_ids:
            if t in (self.PAD_ID, self.EOS_ID):
                break
            if t == self.BOS_ID:
                continue
            chars.append(self._id2token[t])
        return "".join(chars)

    # ------------------------------------------------------------------ #
    #  Convenience helpers                                                 #
    # ------------------------------------------------------------------ #

    def encode_conditions_only(self, line: str) -> List[int]:
        """Encode the first four whitespace-separated tokens (no date)."""
        parts = line.strip().split()[:4]
        return self.encode(" ".join(parts))

    def condition_ids(self, line: str) -> List[int]:
        """
        Return the raw condition token IDs (no BOS/EOS) as a flat list of
        length 6: [day, month, leap, d1, d2, d3].
        Suitable for direct use as the model's condition tensor.
        """
        parts = line.strip().split()
        ids: List[int] = []
        ids.append(self._token2id[parts[0]])
        ids.append(self._token2id[parts[1]])
        ids.append(self._token2id[parts[2]])
        decade = parts[3][1:-1]
        for ch in decade:
            ids.append(self._token2id[ch])
        return ids
