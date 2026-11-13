"""World laws: named, swappable rules for what happens, chosen per world in YAML.

Adding a variant (say, "brutal wolves") means adding one entry to a family's
``KINDS``; the spec validates names against ``FAMILIES`` at load time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from herdmind.world.laws import breeding, disease, metabolism, predation
from herdmind.world.laws.base import Law

if TYPE_CHECKING:
    from herdmind.world.spec import LawsConfig

FAMILIES: dict[str, dict[str, Law]] = {
    "predation": predation.KINDS,
    "breeding": breeding.KINDS,
    "disease": disease.KINDS,
    "metabolism": metabolism.KINDS,
}


def make_laws(cfg: LawsConfig) -> dict[str, Law]:
    """The one law of each family that holds in a world, keyed by family."""
    return {family: kinds[getattr(cfg, family)] for family, kinds in FAMILIES.items()}
