"""The Hugging Face stack is pinned EXACTLY, once, and the pins agree with each other.

**Operator directive 2026-10-18: "update ALL of the huggingface software to the latest
version."** The stack moved together (transformers 4.57.6 -> 5.20.1, huggingface_hub
0.36.2 -> 2.2.0, tokenizers 0.22.2 -> 0.23.3, bitsandbytes -> 0.50.2) and is pinned with
``==`` in ``pyproject.toml``, the one table the image (``backend/Dockerfile`` installs
``.[bnb]``), CI (``backend-tests.yml`` installs ``.[dev]``) and a developer venv all resolve
from. Under a range the dev venv and the image take different releases on different days -
the shape of miLLM's 2026-07-27 production failure (suite memory
``dev-venv-vs-image-version-drift``) and of window 4's (miLLM on 5.x wrote a tokenizer
class miForge's 4.x could not read).

What this pins, all read from the files rather than restated:

* every Hugging Face distribution miForge declares is pinned with ``==`` (a range creeping
  back is red);
* no other build file (Dockerfiles, workflows, compose) pins one a SECOND time, so the
  ``pyproject.toml`` value is the only one;
* the pins satisfy each other's declared requirements, read from the INSTALLED
  distributions' metadata (``transformers`` 5.20.1 needs ``huggingface-hub>=1.31,<3``,
  ``tokenizers>=0.23.1,<0.24``, ``safetensors>=0.8``) - so bumping one alone is red here
  rather than a resolver error in the image build;
* the environment running the suite IS the pinned set (CI installs from the table, so a red
  here in CI means the table and the resolver disagree).
"""

from __future__ import annotations

import re
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PYPROJECT = REPO / "pyproject.toml"

#: The Hugging Face distributions miForge declares. A NEW one added to pyproject without
#: being added here is caught by ``test_every_hf_distribution_declared_is_in_the_set``.
HF_DISTRIBUTIONS = {
    "transformers",
    "tokenizers",
    "safetensors",
    "peft",
    "accelerate",
    "huggingface-hub",
    "bitsandbytes",
}
#: Distributions published by Hugging Face that would belong in the set if ever declared.
HF_PUBLISHED = HF_DISTRIBUTIONS | {
    "datasets",
    "trl",
    "optimum",
    "evaluate",
    "sentence-transformers",
    "hf-xet",
    "diffusers",
    "timm",
}


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _split(requirement: str) -> tuple[str, str]:
    """``"transformers==5.20.1"`` -> ``("transformers", "==5.20.1")``."""
    head = re.split(r";", requirement, maxsplit=1)[0]
    head = re.sub(r"\[[^\]]*\]", "", head).strip()
    match = re.match(r"([A-Za-z0-9_.\-]+)\s*(.*)", head)
    assert match, requirement
    return _canon(match.group(1)), match.group(2).strip()


def _declared() -> dict[str, str]:
    """Every requirement in ``dependencies`` and every extra, by canonical name."""
    project = tomllib.loads(PYPROJECT.read_text("utf-8"))["project"]
    requirements = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    return dict(_split(r) for r in requirements)


def _pins() -> dict[str, str]:
    declared = _declared()
    return {name: declared[name] for name in HF_DISTRIBUTIONS if name in declared}


class TestThePins:
    def test_the_parser_reads_a_pin_and_a_range(self) -> None:
        assert _split("huggingface_hub==2.2.0") == ("huggingface-hub", "==2.2.0")
        assert _split("uvicorn[standard]>=0.32,<0.33") == ("uvicorn", ">=0.32,<0.33")
        assert _split('torch>=2.4,<3; platform_system == "Linux"') == ("torch", ">=2.4,<3")

    def test_every_hf_distribution_declared_is_in_the_set(self) -> None:
        declared = set(_declared())
        unlisted = (declared & HF_PUBLISHED) - HF_DISTRIBUTIONS
        assert not unlisted, f"pin and add to HF_DISTRIBUTIONS: {sorted(unlisted)}"
        assert HF_DISTRIBUTIONS <= declared, sorted(HF_DISTRIBUTIONS - declared)

    def test_each_is_pinned_exactly(self) -> None:
        loose = {n: s for n, s in _pins().items() if not re.fullmatch(r"==[0-9][0-9A-Za-z.]*", s)}
        assert not loose, f"pin with == so image, CI and venv agree: {loose}"

    def test_transformers_is_5x(self) -> None:
        """miLLM saves tokenizers as 5.x's ``TokenizersBackend``; 4.x cannot read them."""
        assert _pins()["transformers"].startswith("==5.")

    def test_no_other_build_file_pins_one_a_second_time(self) -> None:
        candidates = [
            *REPO.glob("**/Dockerfile*"),
            *(REPO / ".github" / "workflows").glob("*.y*ml"),
            *REPO.glob("docker-compose*.y*ml"),
        ]
        names = "|".join(re.escape(n).replace(r"\-", "[-_]") for n in HF_DISTRIBUTIONS)
        pattern = re.compile(rf"\b({names})\s*(==|>=|<=|~=|<|>)", re.IGNORECASE)
        found = [
            f"{path.relative_to(REPO)}: {m.group(0)}"
            for path in candidates
            if "node_modules" not in path.parts
            for m in pattern.finditer(path.read_text("utf-8", errors="replace"))
        ]
        assert not found, f"a second pin drifts from pyproject.toml: {found}"


class TestThePinsAgreeWithEachOther:
    @pytest.mark.parametrize("dependent", sorted(HF_DISTRIBUTIONS))
    def test_each_installed_pin_satisfies_the_others_requirements(self, dependent: str) -> None:
        from packaging.requirements import Requirement
        from packaging.version import Version

        try:
            requires = metadata.requires(dependent) or []
        except metadata.PackageNotFoundError:
            pytest.skip(f"{dependent} is not installed here (bitsandbytes: the [bnb] extra)")
        pins = {n: Version(s.removeprefix("==")) for n, s in _pins().items()}
        broken = []
        for line in requires:
            requirement = Requirement(line)
            if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                continue
            name = _canon(requirement.name)
            if name in pins and pins[name] not in requirement.specifier:
                broken.append(f"{dependent} needs {line}; pinned {pins[name]}")
        assert not broken, broken

    def test_the_running_environment_is_the_pinned_set(self) -> None:
        drift = {}
        for name, spec in _pins().items():
            try:
                installed = metadata.version(name)
            except metadata.PackageNotFoundError:
                continue  # bitsandbytes outside the image: an extra, by design
            if f"=={installed}" != spec:
                drift[name] = {"pinned": spec, "installed": installed}
        assert not drift, f"this environment is not the pinned set: {drift}"


class TestTheTransformersPinMatchesMiLLM:
    """Window 4's defect was exactly this drift: miLLM on 5.x wrote a tokenizer class miForge's
    4.x could not read. Both ends read and write the same checkpoint directories and rebuild
    the same module graph from one config (§5.10's parameter names must line up), so the
    transformers MAJOR.MINOR must agree. Compared against miLLM's exact pin; the sibling is
    found as ``test_torch_pin_matches_millm`` finds it (``MIFORGE_MILLM_ROOT``, also
    ``MILLM_REPO``), skipping loudly when absent and failing under
    ``MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1``."""

    def test_transformers_major_minor_agrees_with_millm(self) -> None:
        import os

        from tests.unit.test_torch_pin_matches_millm import _absent, _millm_root

        root = Path(os.environ["MILLM_REPO"]) if os.environ.get("MILLM_REPO") else _millm_root()
        pyproject = root / "pyproject.toml"
        if not pyproject.is_file():
            _absent(f"miLLM is not checked out at {root}; the transformers pin is unchecked")
        match = re.search(r"""["']transformers==([0-9][^"'\s,;]*)""", pyproject.read_text())
        assert match, f"miLLM's {pyproject} carries no exact transformers== pin"
        theirs = match.group(1).split(".")[:2]
        ours = _pins()["transformers"].removeprefix("==").split(".")[:2]
        assert ours == theirs, f"miForge transformers {ours} vs miLLM {theirs}"
