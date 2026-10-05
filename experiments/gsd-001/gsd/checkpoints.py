from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, order=True)
class CheckpointRef:
    step: int
    revision: str
    tokens_b: float | None = None


def parse_checkpoint_revision(name: str) -> CheckpointRef | None:
    prefix = "stage1-step"
    if not name.startswith(prefix):
        return None

    tail = name[len(prefix):]
    step_text, separator, token_text = tail.partition("-tokens")
    if not step_text.isdigit():
        return None

    tokens_b = None
    if separator:
        if not token_text.endswith("B"):
            return None
        numeric = token_text[:-1]
        try:
            tokens_b = float(numeric)
        except ValueError:
            return None

    return CheckpointRef(
        step=int(step_text),
        revision=name,
        tokens_b=tokens_b,
    )


def normalize_refs(names: Iterable[str]) -> list[CheckpointRef]:
    parsed = [p for name in names if (p := parse_checkpoint_revision(name))]
    return sorted({p.revision: p for p in parsed}.values())


def list_hf_checkpoint_refs(repo_id: str) -> list[CheckpointRef]:
    from huggingface_hub import HfApi

    refs = HfApi().list_repo_refs(repo_id=repo_id, repo_type="model")
    names = [r.name for r in refs.branches] + [r.name for r in refs.tags]
    return normalize_refs(names)


def resolve_hf_model_sha(repo_id: str, revision: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id=repo_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Hugging Face returned no SHA for {repo_id}@{revision}")
    return str(info.sha)


def resolve_hf_dataset_sha(repo_id: str, revision: str = "main") -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id=repo_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Hugging Face returned no SHA for dataset {repo_id}@{revision}")
    return str(info.sha)


def stride_refs(
    refs: list[CheckpointRef],
    stride: int,
    *,
    include_final: bool = True,
) -> list[CheckpointRef]:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if not refs:
        return []
    selected = refs[::stride]
    if include_final and selected[-1] != refs[-1]:
        selected = [*selected, refs[-1]]
    return selected


def first_tranche_20(refs: list[CheckpointRef]) -> list[CheckpointRef]:
    """Frozen 20-point OLMo-2 1B discovery design.

    The current public early-training series contains 37 checkpoints at
    1000-step cadence from step 0 through step 36000. Select every
    2000-step checkpoint (19 points) and add step 35000 as a terminal flank.
    Fail closed if the expected public series shape changes.
    """
    by_step = {r.step: r for r in refs}
    expected = set(range(0, 36001, 1000))
    if set(by_step) != expected:
        missing = sorted(expected - set(by_step))
        extra = sorted(set(by_step) - expected)
        raise ValueError(
            f"unexpected OLMo-2 early-training revision set; missing={missing} extra={extra}"
        )
    steps = list(range(0, 36001, 2000)) + [35000]
    selected = [by_step[step] for step in steps]
    if len(selected) != 20:
        raise AssertionError(f"expected 20 revisions, got {len(selected)}")
    return sorted(selected)
