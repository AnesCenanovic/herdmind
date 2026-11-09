"""World laws: named, swappable rules for what happens, chosen per world in YAML.

Adding a variant (say, "brutal wolves") means adding one entry to a family's
``KINDS``; the spec validates names against ``FAMILIES`` at load time.
"""

from __future__ import annotations

from dataclasses import dataclass

from herdmind.world.laws import breeding, disease, metabolism, predation
from herdmind.world.laws.base import Law

FAMILIES: dict[str, dict[str, Law]] = {
    "predation": predation.KINDS,
    "breeding": breeding.KINDS,
    "disease": disease.KINDS,
    "metabolism": metabolism.KINDS,
}


@dataclass(frozen=True)
class ActiveLaws:
    """The one law of each family that holds in a given world."""

    predation: Law
    breeding: Law
    disease: Law
    metabolism: Law


def make_laws(laws_cfg) -> ActiveLaws:
    return ActiveLaws(**{f: FAMILIES[f][getattr(laws_cfg, f)] for f in FAMILIES})


def describe_laws(spec) -> list[str]:
    active = make_laws(spec.laws)
    return [getattr(active, f).describe(spec) for f in FAMILIES]
