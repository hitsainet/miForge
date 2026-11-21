"""The two rendering layers, and the one direction between them  (task 2.2; FR-007.27).

R-02.35 is explicit, and this is the claim an earlier draft of this project got wrong:

* **``<type>/<name>@<version>`` is miDataworks' CONTRACT form.** Their reference is
  ``{type, name, version}`` and they are normative for it (BRD-03 R-03.47). The vendored
  schema says so in its own description: *"miForge renders ``<type>/<name>@<version>``
  (M-4). No pack in v1 (P-25)."*
* **``<pack>/<type>/<name>@<version>`` is miForge prefixing ITS OWN pack identity for
  DISPLAY.** The pack segment is this project's layer and belongs to nobody else.

**"The two are different layers and neither is a rival spelling of the other."** Conflating
them is how a cross-repo handoff was once recorded for something neither side had agreed to
build. So:

1. **One renderer, two functions**, and nothing else in miForge renders a part.
2. The display form is **display only**: never written into a contract field, never sent to
   miDataworks, never parsed back, never stored as the pinned reference. :class:`DisplayRef`
   is a distinct type so MyPy strict flags it where a storage or outbound signature wants a
   ``str`` or a :class:`~packs.references.ContractPartRef`.
3. **There is no parser.** No ``parse_display_form``, no regex splitting on ``/`` and ``@``.
   A round trip through the display string *is* the layer confusion R-02.35 forbids.
4. The interim ``<dataset>@<version>`` rendering from miDataworks' P-25 amendment is
   **superseded** and appears nowhere - asserted by a grep test over this feature's source,
   because "we did not write it" is a claim and a grep is a check.
5. An **external** part renders with its own marker and never contract-shaped (FR-007.24).
6. A part addressed through **no pack** gets **no invented pack segment**. ``display_form``
   refuses it; :func:`unowned_form` renders the inner form alone. That is FPRD OQ-9, **settled
   2026-10-14**: the mockup's ``general/retention/mixed@2`` is inconsistent, not evidence of an
   implicit ``general`` namespace, because *"miForge has no built-in domain"* is a banner this
   very feature renders. Both halves are kept, because ``pack_id`` arrives from nullable
   storage: a ``None`` that merely *lost* its pack must still refuse, and only a caller that
   names the unowned case gets a rendering.
"""

from __future__ import annotations

from packs.errors import PACK_PART_UNRESOLVED, PackRefusal
from packs.references import ContractPartRef, ExternalPartRef, PartRef


class DisplayRef(str):
    """A rendered part reference, **for display only** (R-02.35).

    A subclass of ``str`` so it renders and serialises naturally in a template or a JSON
    response, while remaining a distinct type that MyPy strict rejects where a plain ``str``
    is required. What must survive any refactor is that **storage and outbound signatures
    take ``str`` or a typed reference and never this.**
    """

    __slots__ = ()


def contract_form(ref: ContractPartRef) -> str:
    """``<type>/<name>@<version>`` - **miDataworks' form**, the inner layer only.

    Takes a :class:`~packs.references.ContractPartRef` and nothing else, deliberately. An
    external part has no contract form, and a function that accepted either would be the
    place an external part acquired one.
    """
    return f"{ref.type}/{ref.name}@{ref.version}"


#: How many hex characters of a digest a rendered external reference carries. Twelve, as
#: everywhere else in this project's operator-facing output: enough to be unambiguous in one
#: installation, short enough to read, and never truncated anywhere the full value is stored.
DIGEST_PREFIX = 12


def external_form(ref: ExternalPartRef) -> str:
    """``external/<origin>@<digest prefix>`` - an external part's own inner form.

    Carries the ``external/`` marker in the position a contract reference carries its
    ``type``, so the two can never be read as the same shape (FR-007.24). The origin is
    rendered **verbatim**; it is operator-supplied text, and escaping it is the display
    layer's job, not this function's - a renderer that HTML-escaped here would put escapes
    into a log line and into a test's expected value.

    **The digest is what is rendered after the ``@``, because the digest is the pin**
    (see :class:`packs.references.ExternalPartRef` on FPRD OQ-2). A rendered prefix is a
    display convenience; the stored reference always carries all 64 characters.
    """
    return f"external/{ref.origin}@{ref.content_sha256[:DIGEST_PREFIX]}"


def display_form(ref: PartRef, pack_id: str | None) -> DisplayRef:
    """``<pack>/<inner form>`` - **miForge's display layer** (FR-007.27).

    Args:
        ref: The pinned reference, of either source kind.
        pack_id: The pack the part is **addressed through** - which may be a different pack
            from the one holding the slot (FR-007.25's cross-pack mixing). Passed in rather
            than read off the reference because the reference does not know: addressing is a
            property of the composition, not of the part.

    Returns:
        The display form, as a :class:`DisplayRef` that no storage or outbound signature
        accepts.

    Raises:
        PackRefusal: ``pack_part_unresolved`` when ``pack_id`` is ``None``. **Refused, not
            invented**: a display form with a guessed pack segment would read as a real
            address, and FPRD OQ-9 is the open question about what pack owns a part that
            belongs to none. A refusal puts the question in front of the operator; a guess
            hides it.
    """
    if pack_id is None:
        raise PackRefusal(
            PACK_PART_UNRESOLVED,
            "This part is addressed through no pack, so there is no pack segment to "
            "render. Address it through a pack, or call unowned_form() to render the inner "
            "form alone - miForge will not invent a namespace, because it has no built-in "
            "domain.",
            reference=_inner(ref),
            source=ref.source,
        )
    return DisplayRef(f"{pack_id}/{_inner(ref)}")


def unowned_form(ref: PartRef) -> DisplayRef:
    """The inner form alone, for a part addressed through **no pack** - FPRD OQ-9, settled.

    This is the "**or render the inner form alone**" that :func:`display_form`'s refusal points
    at, made into a function a caller can actually call. Before it existed the refusal named a
    remedy with no implementation, and the only route to the inner form from outside this
    module was the private ``_inner``.

    **The decision (2026-10-14, operator).** A part that belongs to no pack renders as its own
    inner form - ``<type>/<name>@<version>`` for a contract part, ``external/<origin>@<digest>``
    for an external one - and **no pack segment is invented for it**. So the mockup's
    ``general/retention/mixed@2`` is inconsistent rather than evidence of an implicit ``general``
    namespace: there is no unowned-parts pack, because *"miForge has no built-in domain"* is a
    banner this very feature renders (FR-007.16).

    **Why this is a second function rather than making ``display_form(ref, None)`` lenient.**
    ``pack_id`` arrives from a nullable column and a nullable JSON field, so a ``None`` reaching
    ``display_form`` is far more often a *lost* pack than a genuinely unowned part - and a
    lenient ``display_form`` would render both identically, silently dropping a segment that
    should have been there. Keeping the refusal and requiring the caller to **name** the unowned
    case is what keeps the two distinguishable: one is a question put to the operator, the other
    is an answer the caller asserted.

    Returns:
        A :class:`DisplayRef`, for the same reason :func:`display_form` returns one: this is a
        rendering for a screen, and no storage or outbound signature accepts it. A caller that
        needs the contract string *as data* calls :func:`contract_form` and gets a ``str``.
    """
    return DisplayRef(_inner(ref))


def _inner(ref: PartRef) -> str:
    """The inner layer for either source kind. One ``match``, so neither branch is forgotten."""
    if isinstance(ref, ContractPartRef):
        return contract_form(ref)
    return external_form(ref)
