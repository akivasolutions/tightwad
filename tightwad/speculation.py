"""Core speculative decoding verification logic (pure, no I/O)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class DraftToken:
    token_id: int
    logprob: float  # log probability from draft model
    text: str = ""


@dataclass
class TargetLogprob:
    token_id: int  # target's top token at this position
    logprob: float  # log probability of that top token
    # logprob of the draft token according to the target model
    draft_token_logprob: float | None = None


@dataclass
class VerificationResult:
    accepted_tokens: list[DraftToken]
    bonus_token: DraftToken | None  # extra token from target if all accepted
    accepted_count: int = 0
    rejected_at: int | None = None  # index of first rejection, None if all accepted
    resample_token: DraftToken | None = None  # target's token at rejection point

    @property
    def total_tokens(self) -> int:
        n = self.accepted_count
        if self.bonus_token is not None:
            n += 1
        if self.resample_token is not None:
            n += 1
        return n


def verify_greedy(
    draft_tokens: list[DraftToken],
    target_logprobs: list[TargetLogprob],
) -> VerificationResult:
    """Verify draft tokens in greedy (temperature=0) mode.

    Accept if target's argmax matches draft token, reject otherwise.
    """
    accepted: list[DraftToken] = []

    for i, (draft, target) in enumerate(zip(draft_tokens, target_logprobs)):
        if draft.token_id == target.token_id:
            accepted.append(draft)
        else:
            # Reject: use target's token at this position
            resample = DraftToken(
                token_id=target.token_id,
                logprob=target.logprob,
            )
            return VerificationResult(
                accepted_tokens=accepted,
                bonus_token=None,
                accepted_count=len(accepted),
                rejected_at=i,
                resample_token=resample,
            )

    # All accepted — bonus token comes from target at position after last draft
    bonus = None
    if len(target_logprobs) > len(draft_tokens):
        tp = target_logprobs[len(draft_tokens)]
        bonus = DraftToken(token_id=tp.token_id, logprob=tp.logprob)

    return VerificationResult(
        accepted_tokens=accepted,
        bonus_token=bonus,
        accepted_count=len(accepted),
        rejected_at=None,
    )


def verify_stochastic(
    draft_tokens: list[DraftToken],
    target_logprobs: list[TargetLogprob],
) -> VerificationResult:
    """Verify draft tokens with standard rejection sampling.

    Accept token i with probability min(1, P_target(token_i) / P_draft(token_i)).
    """
    accepted: list[DraftToken] = []

    for i, (draft, target) in enumerate(zip(draft_tokens, target_logprobs)):
        # Get P_target for the draft token
        if target.draft_token_logprob is not None:
            p_target = math.exp(target.draft_token_logprob)
        elif draft.token_id == target.token_id:
            p_target = math.exp(target.logprob)
        else:
            # Draft token not in target's top — treat as very low probability
            p_target = 0.0

        p_draft = math.exp(draft.logprob)
        if p_draft <= 0:
            # Avoid division by zero; accept if target also agrees
            accept = draft.token_id == target.token_id
        else:
            accept_prob = min(1.0, p_target / p_draft)
            accept = random.random() < accept_prob

        if accept:
            accepted.append(draft)
        else:
            resample = DraftToken(
                token_id=target.token_id,
                logprob=target.logprob,
            )
            return VerificationResult(
                accepted_tokens=accepted,
                bonus_token=None,
                accepted_count=len(accepted),
                rejected_at=i,
                resample_token=resample,
            )

    # All accepted
    bonus = None
    if len(target_logprobs) > len(draft_tokens):
        tp = target_logprobs[len(draft_tokens)]
        bonus = DraftToken(token_id=tp.token_id, logprob=tp.logprob)

    return VerificationResult(
        accepted_tokens=accepted,
        bonus_token=bonus,
        accepted_count=len(accepted),
        rejected_at=None,
    )


def verify_draft_tokens(
    draft_tokens: list[DraftToken],
    target_logprobs: list[TargetLogprob],
    temperature: float = 0.0,
) -> VerificationResult:
    """Verify draft tokens against target model logprobs.

    At temperature=0, uses greedy comparison (accept iff argmax matches).
    At temperature>0, uses standard rejection sampling.
    """
    if not draft_tokens:
        # Empty draft — nothing to verify
        bonus = None
        if target_logprobs:
            tp = target_logprobs[0]
            bonus = DraftToken(token_id=tp.token_id, logprob=tp.logprob)
        return VerificationResult(
            accepted_tokens=[],
            bonus_token=bonus,
            accepted_count=0,
            rejected_at=None,
        )

    if temperature == 0.0:
        return verify_greedy(draft_tokens, target_logprobs)
    else:
        return verify_stochastic(draft_tokens, target_logprobs)
