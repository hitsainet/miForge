"""The browser tab shows miForge's mark, and the file it needs reaches every build.

**Why a test and not just a file.** Until 2026-10-14 `frontend/index.html` had a
``<title>`` and **no icon link of any kind**, and there was no ``frontend/public/``
directory. The failure was invisible rather than broken, which is why it lasted: nginx's
``location /`` does ``try_files $uri $uri/ /index.html``, so a browser's automatic
``GET /favicon.ico`` was answered with **index.html, at 200, as ``text/html``**. No 404 in
the access log, no broken-image glyph, no console error — just the browser's default mark
in the tab. A green measurement of something that never happened, which is the family of
failure this project has paid for four times over.

**This file is modelled on `test_license_in_build_contexts.py`** and asserts the same three
things that test learned to assert, because the favicon is in exactly the position the
LICENSE was in:

1. the document **references** it;
2. the file **exists** where the build expects it;
3. **``.dockerignore`` does not exclude it** — the subtler half, where a committed file is
   absent from the build context anyway. BuildKit resolves ``.dockerignore`` per context,
   so it is ``frontend/.dockerignore`` that governs, not the repository root's. That exact
   trap cost real time on the LICENSE the same day.

plus one the LICENSE did not need: that **nginx serves it rather than falling through**.

**The mark is not new art.** ``Sidebar.tsx`` renders ``<IconTile icon={Flame}
surface="logo-tile" />``; ``IconTile.tsx`` renders that surface as ``bg-accent text-white``
with ``rounded-control``. So the SVG is that component in static form, and the assertions
below read its numbers out of ``config/tokens.ts`` rather than restating them — a fixture
whose fields agree by construction is how a suite stays green over a real bug.

**On ADR-012, because this is the one place the accent rule reads as forbidding what is
correct.** "Orange is accent only" is a rule that NAMES ITS SURFACES, and the PADR's own
sentence lists *the logo tile* among the seven: *"orange appears on the active nav item,
primary buttons, the logo tile, list-row icon tiles, the loss curve, running progress bars,
and the callout banner tint."* ``tokens.ts`` carries that list as
``ADR_012_ACCENT_SURFACES``, ``accent.test.tsx`` enforces it, and ``accentAllowlist.ts``
records ``IconTile.tsx`` / ``logo-tile``. A solid accent tile here is therefore the
allowlisted surface, not an exception — and the test asserts it so the next reader does not
"correct" it to a neutral tile.

**Two icons are deliberately NOT shipped**, and the reason is recorded rather than left to
be rediscovered: ``apple-touch-icon.png`` and ``favicon.ico`` must be rasters (iOS ignores
an SVG touch icon), no rasteriser exists on the workstation this was written on, and a
hand-approximated mark shipped as the real one is worse than none. All three siblings ship
SVG only. Their absence is now an honest 404 instead of an HTML document, which is the
better of the two failures available without a rasteriser — and the nginx assertion below
is what makes that true.
"""

from __future__ import annotations

import re
import xml.dom.minidom
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FRONTEND = REPO_ROOT / "frontend"
FAVICON = FRONTEND / "public" / "favicon.svg"

#: Vite copies `public/` to the build root, so this is the served path in `npm run dev`,
#: in `vite preview` (what Playwright serves) and in the nginx image.
SERVED_PATH = "/favicon.svg"


def _read(relative: str) -> str:
    return (FRONTEND / relative).read_text("utf-8")


def _token(name: str) -> str:
    """One value out of `config/tokens.ts`, so this test cannot drift from the UI.

    Read from the TypeScript source rather than duplicated here. If the token is renamed
    this raises instead of silently asserting against a stale constant.
    """
    source = _read("src/config/tokens.ts")
    match = re.search(rf"^\s*{re.escape(name)}:\s*'?([^',\n]+)'?,", source, re.MULTILINE)
    assert match, f"config/tokens.ts no longer declares {name!r}"
    return match.group(1).strip()


class TestTheDocumentReferencesTheIcon:
    def test_index_html_links_the_svg_favicon(self) -> None:
        """The assertion that was false until 2026-10-14."""
        assert re.search(
            r'<link\s+rel="icon"\s+type="image/svg\+xml"\s+href="/favicon\.svg"\s*/?>',
            _read("index.html"),
        ), (
            "frontend/index.html does not link the SVG favicon. Without the link the tab "
            "shows the browser's default mark, and a browser's automatic GET /favicon.ico "
            "is the only request made — which the SPA fallback answers with index.html at "
            "200 as text/html, i.e. silently."
        )

    def test_it_links_no_raster_icon_it_does_not_ship(self) -> None:
        """A link to a file that is not there is the silent failure wearing a fix.

        It would look like support for iOS and Windows tiles while serving a 404 (or, worse,
        before the nginx rule below, index.html). If a rasteriser is ever added, ship the
        files and the links together — this test then fails and says so.
        """
        document = _read("index.html")
        for rel in ("apple-touch-icon", "mask-icon", "manifest"):
            if f'rel="{rel}"' not in document:
                continue
            href = re.search(rf'rel="{rel}"[^>]*href="([^"]+)"', document)
            assert href, f'index.html has rel="{rel}" with no href'
            target = FRONTEND / "public" / href.group(1).lstrip("/")
            assert target.is_file(), (
                f'index.html links rel="{rel}" at {href.group(1)} but '
                f"frontend/public has no such file, so the browser gets a 404 for an "
                f"icon the document promises."
            )


class TestTheFileExistsAndReachesTheImage:
    def test_the_icon_ships_in_public(self) -> None:
        assert FAVICON.is_file(), (
            f"{FAVICON} is missing. Vite copies `public/` to the build root; with no such "
            f"directory the link in index.html resolves to nothing in every environment."
        )

    def test_it_is_well_formed_svg(self) -> None:
        """A malformed SVG renders as nothing, and nothing is what was there before."""
        document = xml.dom.minidom.parseString(FAVICON.read_text("utf-8"))
        assert document.documentElement.tagName == "svg"

    def test_public_is_not_excluded_by_the_dockerignore(self) -> None:
        """The subtler half of the LICENSE failure, checked rather than assumed.

        A committed file BuildKit ignores is not in the build context either: the file
        would be present in the working tree, every assertion above would pass, and the
        published image would have no icon. BuildKit resolves ``.dockerignore`` **per
        context**, and the frontend image's context is ``./frontend`` — so it is that
        file's exclusions that matter here, not the repository root's.
        """
        ignore = FRONTEND / ".dockerignore"
        if not ignore.is_file():
            return
        patterns = [
            line.strip()
            for line in ignore.read_text("utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        offending = [p for p in patterns if p.lstrip("!/").rstrip("/") in {"public", "*"}]
        assert not offending, (
            f"frontend/.dockerignore excludes the public directory via {offending}, so "
            f"favicon.svg never reaches the build context and the published image serves "
            f"no icon — while the file sits committed in the repository."
        )

    def test_vite_has_not_been_pointed_at_a_different_public_directory(self) -> None:
        """`publicDir` defaults to `public`; an override would move the file out of reach."""
        config = _read("vite.config.ts")
        assert "publicDir" not in config, (
            "vite.config.ts now sets `publicDir`, so `public/favicon.svg` may no longer "
            "be copied to the build root. Check the served path before trusting this file."
        )


class TestTheMarkIsTheSidebarsLogoTile:
    """Every number is read from the source it mirrors, not restated."""

    def test_the_tile_is_drawn_at_the_icon_tile_size(self) -> None:
        size = _token("iconTilePx")
        svg = FAVICON.read_text("utf-8")
        assert f'viewBox="0 0 {size} {size}"' in svg, (
            f"the favicon's viewBox is not the tile's own size ({size} px, "
            f"SHELL.iconTilePx). Drawing at the token's value is what lets the radius and "
            f"glyph below be the tokens themselves rather than values scaled off them."
        )
        assert f'<rect width="{size}" height="{size}"' in svg

    def test_the_corner_radius_is_rounded_control(self) -> None:
        """`IconTile.tsx` uses `rounded-control`, which Tailwind maps to this token."""
        assert f'rx="{_token("controlRadiusPx")}"' in FAVICON.read_text("utf-8")

    def test_the_tile_is_filled_with_the_accent(self) -> None:
        """ADR-012's allowlisted `logo-tile` surface: `bg-accent text-white`.

        The accent is identical in both themes on purpose ("the accent is miForge's
        identity"), which is why this mark needs no ``prefers-color-scheme`` variant — a
        mark built from ``panel`` or ``text`` would.
        """
        accent = _token("accent")
        assert accent.startswith("#"), f"accent token is not a hex colour: {accent!r}"
        svg = FAVICON.read_text("utf-8")
        assert f'fill="{accent}"' in svg, (
            f"the favicon is not filled with the accent ({accent}). ADR-012 names the "
            f"logo tile among the seven surfaces orange is permitted on, and "
            f"accentAllowlist.ts records IconTile.tsx/logo-tile — so a solid accent tile "
            f"here is the allowlisted surface, not an exception."
        )
        assert 'stroke="#ffffff"' in svg, "the glyph is not white, as `text-white` renders"

    def test_the_glyph_is_the_icon_the_sidebar_imports(self) -> None:
        """If the sidebar's icon changes, this says so rather than quietly disagreeing."""
        sidebar = _read("src/components/layout/Sidebar.tsx")
        assert "import { Flame } from 'lucide-react';" in sidebar, (
            "Sidebar.tsx no longer imports Flame, so the favicon and the sidebar now show "
            "different marks."
        )
        assert re.search(
            r'<IconTile\s+icon=\{Flame\}\s+surface="logo-tile"', sidebar
        ), "Sidebar.tsx no longer renders Flame as the logo tile"

    def test_the_glyph_is_scaled_the_way_IconTile_scales_it(self) -> None:
        """`IconTile` renders the icon at ``round(sizePx / 2)`` in lucide's 24-unit box.

        36 px tile, 18 px glyph, so ``scale(0.75)`` and ``translate(9 9)``. Derived here
        rather than hard-coded so a change to ``iconTilePx`` fails loudly.
        """
        size = int(_token("iconTilePx"))
        glyph = round(size / 2)
        scale = glyph / 24
        offset = (size - glyph) / 2
        svg = FAVICON.read_text("utf-8")
        assert f"translate({offset:g} {offset:g}) scale({scale:.4g})" in svg, (
            f"the glyph transform does not centre a {glyph} px icon on a {size} px tile; "
            f"expected translate({offset:g} {offset:g}) scale({scale:.4g})"
        )


class TestNginxServesItRatherThanFallingThrough:
    """The layer that turned a missing icon into a silent one."""

    def test_root_icons_do_not_fall_through_to_the_spa(self) -> None:
        conf = _read("nginx.conf")
        block = re.search(r"location\s+~\s+\^/\(favicon[^{]*\{(.*?)\n\s{4}\}", conf, re.DOTALL)
        assert block, (
            "nginx.conf has no regex location for the root-level icons, so "
            "`location /`'s `try_files $uri $uri/ /index.html` answers a missing "
            "/favicon.ico with index.html at 200 and text/html — the original silent "
            "failure."
        )
        body = block.group(1)
        assert "try_files $uri =404;" in body, (
            "the icon location does not end in `=404`, so an absent or renamed icon is "
            "masked by a 200 instead of saying so. That is the fix for the icons that "
            "EXIST as much as for the two that are deliberately not shipped."
        )
        assert "/index.html" not in body, (
            "the icon location falls back to index.html, which is the behaviour it exists "
            "to prevent."
        )

    def test_the_shipped_icon_is_matched_by_that_location(self) -> None:
        """A rule that does not match the file it was written for protects nothing."""
        pattern = re.search(r"location\s+~\s+(\^/\([^)]*\)[^\s{]*)", _read("nginx.conf"))
        assert pattern, "could not read the icon location's regex out of nginx.conf"
        assert re.match(pattern.group(1), SERVED_PATH), (
            f"nginx.conf's icon location {pattern.group(1)!r} does not match "
            f"{SERVED_PATH!r}, the path index.html actually requests."
        )

    def test_the_icon_is_not_cached_as_immutable(self) -> None:
        """It is not content-hashed, unlike `/assets/`.

        A one-year `immutable` entry would strand a changed logo in every cache that saw
        the old one, with no hashed filename to break the tie.
        """
        block = re.search(
            r"location\s+~\s+\^/\(favicon[^{]*\{(.*?)\n\s{4}\}",
            _read("nginx.conf"),
            re.DOTALL,
        )
        assert block and "immutable" not in block.group(1), (
            "the root icons are served `immutable`, but they carry no content hash, so a "
            "new mark could never reach a browser that had cached the old one."
        )

    def test_svg_is_compressed(self) -> None:
        """nginx.conf had no `gzip` directive at all until 2026-10-14."""
        conf = _read("nginx.conf")
        assert re.search(r"^\s*gzip\s+on;", conf, re.MULTILINE), (
            "nginx.conf does not enable gzip, so the whole bundle goes out uncompressed. "
            "miStudio and miDataworks both enable it."
        )
        assert "image/svg+xml" in conf, "gzip_types does not include image/svg+xml"
