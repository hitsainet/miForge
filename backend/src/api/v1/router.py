"""The ``/api/v1`` router  (task 5.1).

One place where every v1 endpoint router is included. A feature adds its router here and
**nowhere else**, which is what makes the reachability test's manifest a complete statement
of the API: a route registered by a second mechanism would be reachable without appearing
in this file, and then the manifest would be a partial list that looks exhaustive.

Health is deliberately **not** here - ADR-013 puts it at ``/api/health``, outside the
version prefix, so a liveness probe does not move when the API version does.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.v1.endpoints import (
    approvals,
    archive,
    attachment,
    chains,
    checkpoints,
    devices,
    host,
    midataworks,
    models_local,
    packs,
    packs_drafts,
    packs_parts,
    parts,
    purge,
    runs,
    runs_acquisition,
    runs_index,
    runs_loop,
    runs_review,
    runtypes,
    scoring,
    selectors,
    settings,
    signal_batches,
    signal_sources,
    tokenizers,
    tree,
)

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(runs.router)
# F5 (R-02.20 - R-02.23, R-02.51). Eight routes, each addressed by `(run_id, step)` rather
# than by a surrogate id. Deleting this line must make the reachability test red - the save
# request, the verify, the lineage walk, the continue and the branch would all still be
# implemented, unit-tested and documented, and would exist for no caller. That is the
# sixteen-unregistered-MCP-tools failure, and `test_checkpoint_reachability.py` runs the
# control explicitly rather than trusting that it would.
api_v1_router.include_router(checkpoints.router)
# F3. Deleting this line must make the reachability test red - the two selector routes
# would still be defined, still unit-tested and still documented, and would not exist for
# any caller. That is the sixteen-unregistered-MCP-tools failure, one layer up.
api_v1_router.include_router(selectors.router)
# F1. Submit, cancel, the step log and the acquisition-mode list. A separate module from
# `runs` on purpose: that one is the foundation's floor for `GET /runs/{run_id}` and carries
# the liveness block, and two handlers for one path is how a response loses a block only one
# of them knew to add. Deleting this line must make the reachability test red.
api_v1_router.include_router(runs_acquisition.router)
# F10. Three routers, because they are three surfaces: the part index and its validator,
# the miDataworks integration's own status, and one run's review counts. Deleting any of
# these lines must make the reachability test red - and `test_f010_reachability.py` runs
# that control explicitly rather than trusting that it would.
api_v1_router.include_router(parts.router)
api_v1_router.include_router(midataworks.router)
# A second router on the `/runs` prefix, deliberately: `GET /runs/{run_id}` is the
# foundation's and F6 extends it, while the review block is F10's. Separate modules means
# neither feature edits the other's file (PPRD section 6).
api_v1_router.include_router(runs_review.router)
# F6 (R-02.24 - R-02.27, R-02.53). A THIRD router on the `/runs` prefix, for the same
# reason: `POST /runs/validate` and `POST /runs/define` are R-02.53's surface and the
# boundary/stop-condition reads are the loop's, while `GET /runs/{run_id}` stays the
# foundation's and F1's four acquisition routes stay F1's. Deleting this line must make
# `test_run_definition_reachability.py` red - the ONE validator R-02.53 requires would
# still be implemented, unit-tested and documented, and would exist for no caller, which
# is the sixteen-unregistered-MCP-tools failure applied to the product's primary action.
api_v1_router.include_router(runs_loop.router)
# F6's two registry listings, on their own prefix rather than under `/runs/`: a collection
# at `/runs/stop-conditions` would be matched by `GET /runs/{run_id}` as a run id, so it
# would depend on the order of lines in this file. **F9's New run form reads these instead
# of transcribing F6's prose** - it had built its stop-condition field from the documented
# precedence order, marked "documentation, not a registry", which is the drift that
# produced `DB_RUN_TYPE` and the stale `RUN_TYPES`. Deleting this line must make
# `test_run_loop_reachability.py` red.
api_v1_router.include_router(runs_loop.registries_router)
# F2 (R-02.4 - R-02.8). Two routers, because they have two readers: the source listing and
# its dry pre-flight are what F7's pack validation and F9's New-run form call before a run
# exists, while the batch records are what F9's step log and F6's stop conditions read after
# one does. Deleting either line must make the reachability test red - every route would
# still be implemented, unit-tested and documented, and would exist for no caller.
api_v1_router.include_router(signal_sources.router)
api_v1_router.include_router(signal_batches.router)
# F4 (R-02.16 - R-02.19, R-02.49). Two routers, because they are two surfaces with two
# readers: the provider registry is what F7's dry-run and F9's New-run form read before a run
# exists, and the score series is what F9's curves read after one does. Deleting either line
# must make the reachability test red - every route would still be implemented, unit-tested
# and documented, and would exist for no caller.
api_v1_router.include_router(scoring.providers_router)
api_v1_router.include_router(scoring.scores_router)
# F8 (R-02.41 - R-02.45). Three routers, because they are three surfaces with three
# readers: the run-type list is what F9's New-run form reads before a run exists, the budget
# preview is what tells it fit-or-not before submit, and the adapter routes are what F9's
# tree and an operator's merge request read after a run has produced one. Deleting any of
# these lines must make `test_runtype_reachability.py` red - every route would still be
# implemented, unit-tested and documented, and would exist for no caller, which is the
# sixteen-unregistered-MCP-tools failure.
api_v1_router.include_router(runtypes.run_types_router)
api_v1_router.include_router(runtypes.budget_router)
api_v1_router.include_router(runtypes.adapters_router)
# `mf_merges` was written, constrained and tested while NOTHING read it - a table
# recording an irreversible operation that an operator could reach only by querying the
# database by hand. For a `refused` merge this table is the ONLY record, because a refused
# merge writes no checkpoint whose manifest could be authoritative instead (ADR-004).
api_v1_router.include_router(runtypes.merges_router)
# F7 (R-02.35, R-02.38 - R-02.40). Three routers, because they are three surfaces with three
# readers: the registry and a pack version's validity, lineage, export and import; the
# authoring flow's drafts, panels and dry-run; and the one read that turns an indexed part
# into a pinnable slot reference. Deleting any of these lines must make the reachability test
# red - every route would still be implemented, unit-tested and documented, and would exist
# for no caller, which is the sixteen-unregistered-MCP-tools failure.
#
# **No queue router, and that absence is a requirement**: the in-run review queue is
# miDataworks' (R-02.36), the pack-authoring dry-run is synchronous and in-request, and
# `test_pack_no_review_queue.py` asserts no queue route is mounted rather than reading the
# endpoint modules.
api_v1_router.include_router(packs.router)
api_v1_router.include_router(packs_drafts.router)
api_v1_router.include_router(packs_parts.router)
# F6 (R-02.27). The four chain routes. The SWEEP is the only path that advances a stage, so
# these are the create/list/detail/stop surface around it and not the advance itself.
# Deleting this line turns `test_chain_wiring.py` red; the four rows in
# `test_reachability.py`'s EXPECTED_ROUTES turn it red in the OTHER direction if the mount
# lands without them, because that manifest is compared both ways.
api_v1_router.include_router(chains.router)

# F9 (R-02.29). ONE route, and the only backend code F9 writes: the artifact forest across
# runs, two node kinds joined on two different edges, a merged checkpoint carrying both.
# Deleting this line must make `test_tree_api.py` and `test_reachability.py` red - the
# composer, the three edge resolutions and the whole of `schemas/tree.py` would still be
# implemented, unit-tested and documented, and the checkpoint tree screen would have no
# data source. That is the sixteen-unregistered-MCP-tools failure, which is why the mount
# was kept out of the implementing agent's scope and the red was observed before it: 51
# failed with this line absent, and a self-certified mount proves nothing.
api_v1_router.include_router(tree.router)

# F9 task 14.0 (operator request 2026-10-18: "selection boxes should be dropdowns that list
# what is available"). THREE read-only listings, each on its own line so its removal control
# removes only it:
#
# * `GET /runs` - every run, newest first, paged. Until this line the path carried only
#   F1's POST, so a GET answered 405 and the Runs screen could not list the runs on the
#   cluster. Its own module rather than `runs.py`, so deleting this line takes the listing
#   and not the run view with it.
# * `GET /models/local` - the model directories under the roots a load is confined to.
# * `GET /gpus` - the visible cards by UUID, never by index.
#
# Deleting any of these lines must make `test_picker_listings.py` and `test_reachability.py`
# red: the New run form's and the Runs screen's pickers would still be implemented and
# unit-tested, and would list nothing for any caller.
api_v1_router.include_router(runs_index.router)
api_v1_router.include_router(models_local.router)
api_v1_router.include_router(devices.router)

# The shell's resource strip (2026-10-19). Until this line nothing served host readings, so
# every metric on every screen read "not reported" while the API pod could see the 3090.
# Deleting it must turn `test_host_resources.py` and `test_reachability.py` red.
api_v1_router.include_router(host.router)
# The attachment chip's live read. The chip showed the open run's last observation, which a
# run that failed while attached left at "attached" for ever. Deleting this line must turn
# `test_attachment_status.py` and `test_reachability.py` red.
api_v1_router.include_router(attachment.router)

# F12 (BRD-MIFORGE-MCP-001). The approval queue, and the settings surface that makes D2's
# "configuration control" reachable at all. Deleting either line must turn
# `test_reachability.py` and the feature's own tests red: the gate would still store
# approvals that nothing could decide, and the settings routes would exist for no caller.
api_v1_router.include_router(approvals.router)
api_v1_router.include_router(settings.router)

# Archive (operator decision 2026-10-20): hide finished runs and unwanted parts from the
# default lists without deleting anything. Deleting this line must turn `test_archive.py`
# (which drives the mounted app) and `test_reachability.py` red.
api_v1_router.include_router(archive.router)

# The tokenizer a create run ships (operator request 2026-10-20): the tokenizer directories
# under the model roots and a Hub revision check, which the New run form's tokenizer picker
# reads. Until this line a create run started from the form was refused `tokenizer_required`,
# because the form had nowhere to name one. Deleting it must turn `test_tokenizer_picker.py`
# and `test_reachability.py` red.
api_v1_router.include_router(tokenizers.router)

# Purge (operator decision 2026-10-20): permanently delete ARCHIVED runs and parts, operator
# only (an agent is refused 403 purge_operator_only; no MCP tool). Deleting this line must turn
# `test_purge.py` (which drives the mounted app) and `test_reachability.py` red.
api_v1_router.include_router(purge.router)
