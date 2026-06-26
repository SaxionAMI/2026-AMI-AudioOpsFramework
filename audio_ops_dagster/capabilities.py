"""
capabilities.py — what a task can DO, instead of what it opts out of.

This replaces the old `dagster.training_enabled` guard. A task declares a
*positive* set of capabilities; the Dagster builder (assets.py / jobs.py) then
assembles only the stages that match, and the inference service loads only the
tasks that declare `serve`. Nothing is described by subtraction anymore.

Declaration lives in a task's config.yaml, in priority order:

  1. explicit list ............ capabilities: [train, evaluate, serve]
  2. archetype (sugar) ........ archetype: trainable_model
  3. legacy fallback .......... dagster: { training_enabled: false }

The legacy fallback exists only so unmodified configs keep working during the
migration. New tasks should use (1) or (2).

The four capabilities
---------------------
  train        produces a checkpoint              → features/splits/train assets
  evaluate     has labels + compute_metrics       → + evaluation asset
  batch_infer  run model over a dataset offline    → batch inference asset
  serve        register with the online inference  → loaded by the serve daemon
               service (api.py); NOT a Dagster asset

`serve` deliberately contributes nothing to the Dagster graph — that is the
whole point. A serve-only task (Whisper, YAMNet) yields zero assets and zero
jobs here and lives entirely in the inference service's registry.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable

# ── Capability tokens ────────────────────────────────────────────────────────

TRAIN = "train"
EVALUATE = "evaluate"
BATCH_INFER = "batch_infer"
SERVE = "serve"

ALL_CAPABILITIES: FrozenSet[str] = frozenset({TRAIN, EVALUATE, BATCH_INFER, SERVE})


# ── Archetypes (named capability bundles) ────────────────────────────────────
# Sugar over the primitives. The builders only ever consume the expanded set.

ARCHETYPES: Dict[str, FrozenSet[str]] = {
    # full arc: trains, evaluates, can batch-infer, and serves online
    "trainable_model": frozenset({TRAIN, EVALUATE, BATCH_INFER, SERVE}),
    # pretrained, online only (Whisper, YAMNet) — no Dagster footprint at all
    "pretrained_servable": frozenset({SERVE}),
    # pretrained, offline batch scoring of a dataset, no online serving
    "pretrained_batch": frozenset({BATCH_INFER}),
    # benchmarking against an existing checkpoint, no serving
    "eval_only": frozenset({EVALUATE}),
}


class CapabilityError(ValueError):
    """Raised when a config declares an unknown capability or archetype."""


def _expand_tokens(tokens: Iterable[str]) -> FrozenSet[str]:
    """
    Expand a list of tokens into a capability set. Each token may be a raw
    capability (``train``) or an archetype name (``trainable_model``); archetype
    names expand to their bundle so the two styles can be mixed freely.
    """
    resolved = set()
    for tok in tokens:
        key = str(tok).strip()
        if key in ALL_CAPABILITIES:
            resolved.add(key)
        elif key in ARCHETYPES:
            resolved.update(ARCHETYPES[key])
        else:
            raise CapabilityError(
                f"Unknown capability/archetype '{key}'. "
                f"Valid capabilities: {sorted(ALL_CAPABILITIES)}; "
                f"valid archetypes: {sorted(ARCHETYPES)}."
            )
    return frozenset(resolved)


def resolve_capabilities(config: dict) -> FrozenSet[str]:
    """
    Resolve a task's capability set from its parsed config.yaml dict.

    Priority: explicit ``capabilities`` list → ``archetype`` → legacy
    ``dagster.training_enabled`` fallback. An empty/missing declaration with no
    legacy flag defaults to ``trainable_model`` (the historical default, which
    treated every task as fully trainable unless it opted out).
    """
    explicit = config.get("capabilities")
    if explicit:
        return _expand_tokens(explicit)

    archetype = config.get("archetype")
    if archetype:
        if archetype not in ARCHETYPES:
            raise CapabilityError(
                f"Unknown archetype '{archetype}'. Valid archetypes: {sorted(ARCHETYPES)}."
            )
        return ARCHETYPES[archetype]

    # Legacy fallback — keeps pre-capability configs working unchanged.
    training_enabled = config.get("dagster", {}).get("training_enabled", True)
    return ARCHETYPES["trainable_model"] if training_enabled else ARCHETYPES["pretrained_servable"]


# ── Convenience predicates (used by definitions.py and the inference service) ──

def has(caps: FrozenSet[str], capability: str) -> bool:
    return capability in caps


def serves(config: dict) -> bool:
    """True if this task should be loaded by the online inference service."""
    return SERVE in resolve_capabilities(config)


def produces_dagster_assets(caps: FrozenSet[str]) -> bool:
    """True if the task contributes anything to the Dagster graph."""
    return bool(caps & {TRAIN, EVALUATE, BATCH_INFER})


def describe(caps: FrozenSet[str]) -> str:
    """Stable, human-readable rendering for logs."""
    return "+".join(c for c in (TRAIN, EVALUATE, BATCH_INFER, SERVE) if c in caps) or "(none)"


__all__ = [
    "TRAIN",
    "EVALUATE",
    "BATCH_INFER",
    "SERVE",
    "ALL_CAPABILITIES",
    "ARCHETYPES",
    "CapabilityError",
    "resolve_capabilities",
    "has",
    "serves",
    "produces_dagster_assets",
    "describe",
]