"""
safety_constrainers.py – PATROL safety constrainers for jailbreak defense.

Two approaches:
  Option A  KeywordSafetyConstrainer  – fast, deterministic prefix matching.
  Option B  ClassifierSafetyConstrainer – LlamaGuard-2-8B on CPU.

Both extend a minimal Constrainer ABC defined here so that this file has
zero dependency on transformers_cfg (not available in trustllm env).
Compatible with Python 3.9+ and MPS (no float64, no zip strict).
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from cachetools import LRUCache
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM


# ─────────────────────────────────────────────────────────────────────────────
# Minimal Constrainer base (mirrors constrainers.py without transformers_cfg)
# ─────────────────────────────────────────────────────────────────────────────

class Constrainer(ABC):
    @abstractmethod
    def get_blocklist(self, tokens: list) -> list:
        ...

    def ask_token(self, tokens: list, token: int) -> bool:
        return token not in self.get_blocklist(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Shared word-lists
# ─────────────────────────────────────────────────────────────────────────────

REFUSAL_PREFIXES: list = [
    "I'm sorry",
    "I cannot",
    "I can't",
    "I apologize",
    "I am unable",
    "I won't",
    "I will not",
    "As an AI",
    "I'm not able",
    "I don't think I can",
    "That's not something",
    "This isn't something",
    "I'm afraid I can't",
    "I must decline",
    "I'm going to have to decline",
    "I am not going to",
    "I'm not going to",
    "I don't feel comfortable",
]

# Tuple: (prefix_string, case_sensitive)
HARMFUL_RESPONSE_PREFIXES: list = [
    # Affirmative compliance
    ("Sure, here", False),
    ("Sure! Here", False),
    ("Sure, I can", False),
    ("Sure! I can", False),
    ("Of course! Here", False),
    ("Of course, here", False),
    ("Of course, I", False),
    ("Of course! I", False),
    ("Absolutely! Here", False),
    ("Absolutely, here", False),
    ("Absolutely! I", False),
    ("Certainly! Here", False),
    ("Certainly, here", False),
    ("Certainly! I", False),
    ("Happy to help", False),
    ("I'd be happy to", False),
    ("I would be happy to", False),
    ("I'll help you", False),
    ("I will help you", False),
    ("No problem! Here", False),
    ("No problem, here", False),
    ("Great question! Here", False),
    # Instructional openers
    ("Step 1:", False),
    ("Step 1.", False),
    ("Step 1 -", False),
    ("1. ", False),
    ("1) ", False),
    ("Here is how to", False),
    ("Here's how to", False),
    ("Here are the steps", False),
    ("Here is a step-by-step", False),
    ("Here's a step-by-step", False),
    ("Here is a detailed guide", False),
    ("Here's a detailed guide", False),
    ("Here is a tutorial", False),
    ("Here's a tutorial", False),
    ("Here is a recipe", False),
    ("Here's a recipe", False),
    ("Here is the code", False),
    ("Here's the code", False),
    ("Here is the script", False),
    ("Here's the script", False),
    ("Here is the method", False),
    ("Here's the method", False),
    ("Here is the process", False),
    ("Here's the process", False),
    ("Here is a way to", False),
    ("Here's a way to", False),
    ("Below are the steps", False),
    ("Below is a guide", False),
    ("Below is how", False),
    ("The following steps", False),
    ("To begin,", False),
    ("First, you need to", False),
    ("First, you will need", False),
    ("First, gather", False),
    ("First, obtain", False),
    ("To make this", False),
    ("To create this", False),
    ("To synthesize", False),
    ("To hack", False),
    ("To exploit", False),
    # Roleplay / jailbreak markers
    ("[DAN]", True),
    ("[JAILBREAK]", True),
    ("[UNFILTERED]", True),
    ("DAN Mode enabled", True),
    ("DAN Mode Output:", True),
    ("As DAN,", True),
    ("As [A]", True),
    ("[GPT-4]", True),
]


def _all_ids(vocab_size: int) -> list:
    return list(range(vocab_size))


# ─────────────────────────────────────────────────────────────────────────────
# Option A – Keyword / Prefix Blocking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KeywordSafetyConstrainer(Constrainer):
    """
    Fast, deterministic prefix-based safety constrainer (Option A).

    At each token position:
      • Decode the generated tokens so far.
      • Refusal prefix detected  → allow all tokens (model is refusing).
      • Harmful prefix detected  → block ALL tokens (trigger backtrack).
      • Neutral                  → allow all tokens.

    Prefix-monotonic, zero GPU cost, LRU-cached.
    """

    tokenizer: PreTrainedTokenizer
    harmful_prefixes: list = field(default_factory=lambda: list(HARMFUL_RESPONSE_PREFIXES))
    refusal_prefixes: list = field(default_factory=lambda: list(REFUSAL_PREFIXES))
    _cache: LRUCache = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_cache", LRUCache(maxsize=8192))

    def _decode(self, tokens: list) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _is_refusal(self, text: str) -> bool:
        t = text.lstrip()
        return any(t.startswith(r) for r in self.refusal_prefixes)

    def _is_harmful(self, text: str) -> bool:
        t = text.lstrip()
        t_lower = t.lower()
        for prefix, case_sensitive in self.harmful_prefixes:
            hay = t if case_sensitive else t_lower
            ndl = prefix if case_sensitive else prefix.lower()
            if hay.startswith(ndl):
                return True
        return False

    def is_sequence_harmful(self, tokens: list) -> bool:
        """Check whether the decoded sequence is harmful. Used by criteria()."""
        if not tokens:
            return False
        text = self._decode(tokens)
        return not self._is_refusal(text) and self._is_harmful(text)

    def get_blocklist(self, tokens: list) -> list:
        # Token-level blocking is not used — detection happens in criteria().
        # Returning [] preserves PATROL's backtracking math (no NaN from all-inf logits).
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Jailbreak prompt heuristics
# ─────────────────────────────────────────────────────────────────────────────
# LlamaGuard-3-1B has a blind spot for fiction-framed attacks: it classifies
# content based on whether it *is* harmful, not whether the prompt is *designed*
# to elicit harm. PAIR (role-play framing) and DeepInception (nested fiction)
# both return "safe" even for explicit harmful steps because the fictional
# wrapper moves them out of LlamaGuard's training distribution.
#
# These lightweight regexes catch the attack-specific framing in the *prompt*
# itself, before any tokens are generated.

import re as _re

# PAIR / role-play markers — look for a role-play setup AND a harmful action word.
# Use \w* to match word stems regardless of suffix (infiltrat→infiltrate, etc.)
_PAIR_ROLE_RE = _re.compile(
    r"\b(imagine|pretend|suppose|act\s+as|roleplay|you\s+are|you'?re\s+(?:playing|acting|a\b))"
    r".{0,200}"
    r"\b(hack\w*|exploit\w*|infiltrat\w*|synthesiz\w*|manufactur\w*|creat\w*|build\w*"
    r"|provid\w*|develop\w*|retriev\w*|steal\w*|bypass\w*|disabl\w*)",
    _re.I | _re.S,
)

# DeepInception — nested-layer fiction structure
_DI_LAYER_RE = _re.compile(
    r"\b(layer\s*\d|layer\s*i\b|reach\s*layer|layer\s*i\s*\+\s*1)\b",
    _re.I,
)

# Generic "fictional framing + harmful goal" — catches both and more
_FICTION_HARM_RE = _re.compile(
    r"\b(science[\s\-]?fiction|thriller|movie|story|novel|character|scenario|narrative)\b"
    r".{0,300}"
    r"\b(bomb|exploit|hack\w*|drug\w*|poison\w*|weapon\w*|kill|murder|malware|ransomware"
    r"|synthesiz\w*|vulnerabilit\w*|sensitive\s+data|classified\s+(?:info|data))\b",
    _re.I | _re.S,
)


def _prompt_is_jailbreak(text: str) -> bool:
    """
    Return True if the prompt pattern matches known jailbreak structures
    (PAIR role-play framing, DeepInception nested layers, or general
    fiction-wrapped harmful requests).  Fast; zero model calls.
    """
    return bool(
        _PAIR_ROLE_RE.search(text)
        or _DI_LAYER_RE.search(text)
        or _FICTION_HARM_RE.search(text)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Option B – LlamaGuard-3-1B Classifier Constrainer
# ─────────────────────────────────────────────────────────────────────────────

# LlamaGuard-3 uses the Llama-3 chat template.
# We build the conversation as [user_prompt, partial_agent_response]
# and ask the model to classify safety.

@dataclass
class ClassifierSafetyConstrainer(Constrainer):
    """
    LlamaGuard-3-1B based safety constrainer (Option B).

    Runs the classifier every `check_interval` generated tokens.
    The classifier model runs on CPU to avoid competing with the
    target model on MPS.  Results are LRU-cached by token tuple.

    Call `update_user_message(new_goal)` between behaviors to
    swap the context and reset the cache.
    """

    base_tokenizer: PreTrainedTokenizer      # target model tokenizer
    classifier_model: AutoModelForCausalLM   # LlamaGuard-2-8B
    classifier_tokenizer: AutoTokenizer      # LlamaGuard-2-8B tokenizer
    user_message: str                        # the current JBB Goal prompt
    check_interval: int = 8
    max_check_tokens: int = 0               # stop checking after this many tokens (0 = no limit)
    cache_size: int = 2048
    _cache: LRUCache = field(init=False, repr=False)
    _blocked: dict = field(init=False, repr=False)  # position → set of blocked token ids
    n_checks: int = field(init=False, repr=False)   # LlamaGuard calls this prompt
    n_unsafe: int = field(init=False, repr=False)   # UNSAFE detections this prompt

    def __post_init__(self):
        object.__setattr__(self, "_cache",    LRUCache(maxsize=self.cache_size))
        object.__setattr__(self, "_blocked",  {})
        object.__setattr__(self, "n_checks",  0)
        object.__setattr__(self, "n_unsafe",  0)

    def _decode(self, tokens: list) -> str:
        return self.base_tokenizer.decode(tokens, skip_special_tokens=True)

    @torch.no_grad()
    def _classify(self, agent_text: str) -> bool:
        """
        Ask LlamaGuard-3-1B whether the conversation is unsafe.

        agent_text == ""  → classify the user prompt alone (pre-generation check).
                            LlamaGuard judges whether the user turn itself is a
                            harmful / jailbreak request.
        agent_text != ""  → classify prompt + partial assistant response jointly.
        """
        if agent_text:
            conversation = [
                {"role": "user",      "content": self.user_message},
                {"role": "assistant", "content": agent_text},
            ]
        else:
            # No response yet — judge the user prompt alone
            conversation = [
                {"role": "user", "content": self.user_message},
            ]
        prompt = self.classifier_tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.classifier_tokenizer(prompt, return_tensors="pt").to(self.classifier_model.device)
        out_ids = self.classifier_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=self.classifier_tokenizer.eos_token_id,
        )
        new_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        verdict = self.classifier_tokenizer.decode(
            new_ids, skip_special_tokens=True
        ).strip().lower()
        result = verdict.startswith("unsafe")
        self.n_checks += 1
        if result:
            self.n_unsafe += 1
        return result

    def block_token_at(self, position: int, token: int) -> None:
        """Permanently block `token` at generation position `position`.

        Called after a harmful detection so the same token is never picked
        again at that position, forcing the model down a different branch.
        `position` is 0-based index into the generated token sequence
        (i.e. len(context) when get_blocklist is called).
        """
        if position not in self._blocked:
            self._blocked[position] = set()
        self._blocked[position].add(token)

    def get_blocklist(self, context: list) -> list:
        """Return tokens that are permanently blocked at the current position."""
        return list(self._blocked.get(len(context), set()))

    def update_user_message(self, new_message: str) -> None:
        """Reset context, cache, block list, and counters for a new JBB behavior."""
        self.user_message = new_message
        self._cache.clear()
        self._blocked.clear()
        object.__setattr__(self, "n_checks", 0)
        object.__setattr__(self, "n_unsafe", 0)

    def is_sequence_harmful(self, tokens: list) -> bool:
        """
        Check whether the conversation is unsafe.

        tokens == []  → pre-generation check; returns False (no-op) so generation
                        always proceeds. The decoding-time checks (tokens != [])
                        are the actual defense mechanism.
        tokens != []  → classify prompt + partial response every check_interval tokens.
        """
        if not tokens:
            # Pre-generation check is disabled: let generation start and rely on
            # the decoding-time LlamaGuard checks to catch harmful output.
            # Enabling a pre-check would flag all DeepInception / PAIR prompts
            # via regex before any generation, which inflates apparent ASR reduction
            # and makes comparison with other baselines unfair.
            return False

        # Stop checking once past the early window (if max_check_tokens is set)
        if self.max_check_tokens > 0 and len(tokens) > self.max_check_tokens:
            return False
        # Check at multiples of check_interval only
        if len(tokens) % self.check_interval != 0:
            return False
        key = tuple(tokens)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        agent_text = self._decode(tokens)
        result = self._classify(agent_text)
        self._cache[key] = result
        return result
