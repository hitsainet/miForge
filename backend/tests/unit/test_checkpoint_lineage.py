"""The parent chain, and the distinction that makes it honest  (FR-005.39, E-11).

**``truncated`` is not ``root``.** One means *this is the beginning*, the other *the record
stops here*. Conflating them makes a partial tree look complete, which is the quietly wrong
answer ADR-013 forbids - and it is the easy mistake, because both end the loop.

The walk reads **manifests**, never the index: ADR-004 gives the index no authority over a
correctness answer, and a lineage chain is one. Every chain below is built by calling the
real writer in sequence, so what is walked is what a save actually produces.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from core.errors import Refusal
from services.checkpoints.lineage import walk
from services.checkpoints.paths import checkpoint_dir, parse_qualified_name
from tests.support.checkpoints import saved_checkpoint, score, write_chain


class TestTheWalkReachesTheRoot:
    def test_lineage_walk_reaches_the_root(self, tmp_path: Path):
        """**MUT-9's second target.** A chain whose ``base_model`` was set to the parent on
        each hop would still *validate* and still look plausible - and would terminate one
        hop short of the model the work actually started from."""
        write_chain(tmp_path, steps=(100, 200, 300))
        result = walk("run-a", 300, root=tmp_path)

        assert result.terminated == "root"
        assert result.terminated_at is None
        assert [n.step for n in result.nodes] == [300, 200, 100]
        assert result.nodes[-1].parent is None
        assert {n.base_model for n in result.nodes} == {"LiquidAI/LFM2-1.2B"}, (
            "every node must name the same lineage root; a differing one means the root was "
            "re-derived from the thing most recently loaded"
        )

    def test_a_branch_joins_its_parents_chain(self, tmp_path: Path):
        """AC-005.8: a branch's first manifest's ``parent`` is the branch-point name and the
        walk crosses the run boundary, which is what R-02.23's *"lineage remains traceable
        across runs"* means."""
        chain = write_chain(tmp_path, steps=(100, 200))
        saved_checkpoint(tmp_path, run_id="run-b", step=0, parent=chain[-1].name)

        result = walk("run-b", 0, root=tmp_path)
        assert result.terminated == "root"
        assert [n.run_id for n in result.nodes] == ["run-b", "run-a", "run-a"]


class TestTruncatedIsNotRoot:
    def test_a_missing_ancestor_truncates_at_that_name(self, tmp_path: Path):
        """E-11. The branch still proceeds - lineage is a chain of names, not of live files
        - and the response says **where** the record ran out."""
        chain = write_chain(tmp_path, steps=(100, 200))
        shutil.rmtree(checkpoint_dir("run-a", 100, root=tmp_path))

        result = walk("run-a", 200, root=tmp_path)
        assert result.terminated == "truncated"
        assert result.terminated_at == chain[0].name
        assert len(result.nodes) == 1

    def test_an_unparseable_parent_name_truncates_rather_than_rooting(self, tmp_path: Path):
        saved_checkpoint(tmp_path, run_id="run-x", step=1, parent="not-a-qualified-name")
        result = walk("run-x", 1, root=tmp_path)
        assert result.terminated == "truncated"
        assert result.terminated_at == "not-a-qualified-name"

    def test_the_limit_is_its_own_termination(self, tmp_path: Path):
        write_chain(tmp_path, steps=(1, 2, 3, 4))
        result = walk("run-a", 4, root=tmp_path, limit=2)
        assert result.terminated == "limit"
        assert len(result.nodes) == 2

    def test_a_cycle_terminates_as_truncated_not_as_a_root(self, tmp_path: Path):
        """A copied-in manifest can name a parent that names it back. Reported as the record
        stopping rather than as the beginning, for the same reason a missing one is."""
        a = saved_checkpoint(tmp_path, run_id="cyc", step=1, parent=None)
        b = saved_checkpoint(tmp_path, run_id="cyc", step=2, parent=a.name)
        # Rewrite step 1's parent to point back at step 2.
        import json

        path = checkpoint_dir("cyc", 1, root=tmp_path) / "manifest.json"
        doc = json.loads(path.read_text())
        doc["parent"] = b.name
        path.write_text(json.dumps(doc))

        result = walk("cyc", 2, root=tmp_path)
        assert result.terminated == "truncated"
        assert len(result.nodes) == 2


class TestWhatANodeCarries:
    def test_a_node_carries_the_score_with_its_sample_and_set(self, tmp_path: Path):
        """ADR-012: **every number names its scale and sample.** A score without its ``n``
        is not a measurement, which is why ``n`` is in the ``ScoreRecord`` at all."""
        saved_checkpoint(tmp_path, run_id="r", step=1, test=score(n=240, value=0.77))
        node = walk("r", 1, root=tmp_path).nodes[0]
        assert node.test == {
            "set": "contract-clauses-test",
            "score": 0.77,
            "n": 240,
            "provider": "in-process:lfm2-1.2b",
        }

    def test_a_null_domain_is_rendered_as_an_assertion_not_as_a_blank(self, tmp_path: Path):
        """``domain`` may be null and the node carries the **state** beside the value, so a
        renderer says "no domain" rather than leaving a blank that reads like missing data."""
        saved_checkpoint(tmp_path, run_id="r", step=1, domain=None)
        node = walk("r", 1, root=tmp_path).nodes[0]
        assert node.domain is None
        assert node.domain_state == "asserted_null"
        assert node.domain_display and node.domain_display != ""

    def test_a_merged_node_carries_its_second_edge(self, tmp_path: Path):
        """R-02.45: two ancestors, two fields. Without ``merged_adapter`` two checkpoints at
        the same step with the same parent - one merged, one not - are indistinguishable."""
        parent = saved_checkpoint(tmp_path, run_id="base", step=1)
        saved_checkpoint(
            tmp_path,
            run_id="merged",
            step=1,
            parent=parent.name,
            merged_adapter={"name": "x:lora:step-800", "step": 800, "manifest_sha256": "e" * 64},
        )
        node = walk("merged", 1, root=tmp_path).nodes[0]
        assert node.parent == parent.name
        assert node.merged_adapter["name"] == "x:lora:step-800"

    def test_no_depth_value_is_shipped(self, tmp_path: Path):
        """**F9 computes depth from the chain.** Depth is a property of a walk rather than
        of a node, so nothing could rebuild a stored one - and the mockup's own handoff notes
        that its indentation came from a ``depth`` field and that the real tree needs a walk."""
        saved_checkpoint(tmp_path, run_id="r", step=1)
        assert "depth" not in walk("r", 1, root=tmp_path).nodes[0].as_dict()


class TestTheStartingCheckpointStillRefuses:
    def test_an_absent_starting_checkpoint_refuses(self, tmp_path: Path):
        """A broken link *further up* is a truncated chain; the thing asked about not
        existing at all is a refusal."""
        with pytest.raises(Refusal) as caught:
            walk("nobody", 1, root=tmp_path)
        assert caught.value.code == "checkpoint_not_found"


class TestQualifiedNameParsing:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("LiquidAI/LFM2-1.2B:run-a:step-500", ("LiquidAI/LFM2-1.2B", "run-a", 500)),
            # A base model identity may legitimately carry a colon - a repo at a revision -
            # so the split is from the RIGHT. Splitting from the left would attribute the
            # revision's tail to the run.
            ("hf.co/org/model:rev1:run-a:step-0", ("hf.co/org/model:rev1", "run-a", 0)),
        ],
    )
    def test_the_name_splits_from_the_right(self, name, expected):
        assert parse_qualified_name(name) == expected

    @pytest.mark.parametrize("name", ["", "no-colons", "a:b", "a:b:step-x", "a:b:500"])
    def test_an_unparseable_name_is_none_rather_than_a_refusal(self, name):
        """``None`` is what lets the walk terminate as ``truncated``; a refusal there would
        make one bad ancestor hide a perfectly good chain."""
        assert parse_qualified_name(name) is None
