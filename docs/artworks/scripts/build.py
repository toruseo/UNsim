"""Builds unsim/svg/unsim-{hero,logo,icon}-{white,dark,transparent-light,transparent-dark,transparent-outline}.svg and exports unsim/png/.

Usage (from any directory): python unsim/scripts/build.py
Needs fontTools, uharfbuzz, numpy, scipy and contourpy importable (tmp/pylib or the user environment, see README.md) and Chrome for the PNGs.
UNsim uses the UXsim mark with a nabla N in its centre; the hero adds a loss landscape with a steepest-ascent trajectory (gen_surface.py).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "common"), os.path.dirname(os.path.abspath(__file__)), os.path.join(ROOT, "tmp", "pylib")]
import glyphs  # noqa: E402
import export_png  # noqa: E402
import gen_surface  # noqa: E402
import svgparts as sp  # noqa: E402
from gen_surface import f1  # noqa: E402

NAME = "unsim"
TITLE = "UNsim"
HERO_TITLE = "UNsim: Python/JAX traffic simulator with AutoDiff"
# Wordmark "UNsim": Poppins SemiBold with the N from Poppins SemiBold Italic, tracking -40; (font, character, x in font units), one path per group
WORDMARK = [[("SemiBold", "U", 0), ("SemiBoldItalic", "N", 657)], [("SemiBold", "s", 1352), ("SemiBold", "i", 1857), ("SemiBold", "m", 2095)]]
WM_COMMENT = "Wordmark: Poppins SemiBold with the N from Poppins SemiBold Italic (SIL OFL 1.1), tracking -40, converted to outlines; font units, baseline at y=0"
# Tagline of the hero: Poppins Regular; character ranges of the two paths ("Python/JAX" and the rest) and the placement (x, y, scale)
TAGLINE = "Python/JAX traffic simulator with AutoDiff"
TAGLINE_GROUPS = [(0, 10), (10, 42)]
TAGLINE_TF = (584.454, 398.053, 0.0306339)

# Nabla N in the centre of the mark, in font units with the baseline at y=0.
# Nabla: a triangle with sharp corners on the box of the increment sign of Poppins SemiBold (674 x 700, left side bearing 20), with its top bar (108) and side strokes (130).
# N: Poppins SemiBold Italic at x=674 (tracking -40). The group is scaled by NABLA_SCALE and centred on the mark.
NABLA = "M20 -700H694L357 0Z M497.7 -592H216.3L357 -299.7Z"
NABLA_BOX = (20, -700, 694, 0)
NABLA_N = [("SemiBoldItalic", "N", 674)]
NABLA_SCALE = 0.125
NABLA_FILL = {"light": "#306998", "dark": "#FFD43B"}

# Hero: loss landscape (gen_surface.py) with the foothills reaching down to the height of the summit mirrored about the canvas centre
SURFACE = dict(hp=240, sq=150, start="761.5,196.8", t1=0.3)
LAND = {"light": ("#306998", "#9FB6CE"), "dark": ("#6FA8DC", "#5F80A8")}   # contour lines, mesh
ROUTE = {"light": ("#306998", "#5A9FD4", "#F5B800", "#FFD43B"), "dark": ("#5A9FD4", "#7DB6E2", "#F5B800", "#FFD43B")}  # trajectory gradient, foot to summit
ROUTE_W, HALO_W = 4, 7.2         # trajectory line and its white halo (in gen_surface coordinates)
START_R, END_R, END_STROKE = 4.2, 6.2, 2.4   # start and summit markers; the halo of the markers is END_STROKE wide
TEXT_GAP = 4                      # px: the lines of the landscape stop this far from the wordmark and the tagline


def nabla_n(style, kind, n_path, n_bbox):
    x0, y0 = min(NABLA_BOX[0], n_bbox[0]), min(NABLA_BOX[1], n_bbox[1])
    x1, y1 = max(NABLA_BOX[2], n_bbox[2]), max(NABLA_BOX[3], n_bbox[3])
    x, y = 256 - (x0 + x1) / 2 * NABLA_SCALE, 256 - (y0 + y1) / 2 * NABLA_SCALE
    attrs = f'transform="{sp.tf(x, y, NABLA_SCALE)}" fill="{NABLA_FILL[sp.palette(style)]}"'
    if style == "transparent-outline":
        attrs += f' stroke="#fff" stroke-width="{sp.sw(sp.OUTLINE_PX[kind], sp.MARK_TF[kind][2] * NABLA_SCALE)}" stroke-linejoin="round" paint-order="stroke"'
    return ("  <!-- Nabla N in the centre: a triangle with sharp corners drawn on the increment sign of Poppins SemiBold, and the N of Poppins SemiBold Italic (SIL OFL 1.1) -->\n"
            f"  <g {attrs}>\n    <path d=\"{NABLA}\"/>\n    <path d=\"{n_path}\"/>\n  </g>\n")


def text_outlines(wm, tag, colour):
    """the wordmark and the tagline, filled and stroked in colour: the gap of the landscape around the text"""
    out = ""
    for (x, y, s), paths in ((sp.WORDMARK_TF["hero"], wm), (TAGLINE_TF, tag)):
        out += (f'  <g transform="{sp.tf(x, y, s)}" fill="{colour}" stroke="{colour}" stroke-width="{sp.sw(TEXT_GAP, s)}" stroke-linejoin="round">\n'
                + "".join(f'    <path d="{d}"/>\n' for d in paths) + "  </g>\n")
    return out


def hero(style, wm, tag, inner, geo):
    p = f"{NAME}-hero-{style}"
    pal = sp.palette(style)
    halo = style in ("white", "transparent-outline")   # white edges around the trajectory and its markers
    masked = style != "white"                           # white: the text is underlaid in white; other styles: the landscape is masked around the text
    TF = gen_surface.TF
    (ax, ay), (bx, by), d = geo["start"], geo["end"], geo["traj"]
    stops = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in zip(("0", ".3", ".65", "1"), ROUTE[pal]))
    defs = f'    <linearGradient id="{p}-route" gradientUnits="userSpaceOnUse" x1="{f1(ax)}" y1="{f1(ay)}" x2="{f1(bx)}" y2="{f1(by)}">{stops}</linearGradient>\n'
    if masked:
        defs += (f"    <!-- Mask of the landscape: the lines stop {TEXT_GAP} px short of the wordmark and the tagline (the same outlines, stroked), and leave a gap around the trajectory and its markers -->\n"
                 f'    <mask id="{p}-land" maskUnits="userSpaceOnUse" x="0" y="0" width="{sp.HERO[0]}" height="{sp.HERO[1]}">\n'
                 f'      <rect width="{sp.HERO[0]}" height="{sp.HERO[1]}" fill="#fff"/>\n' + sp.ind(text_outlines(wm, tag, "#000"), 2)
                 + f"      <g {TF}>\n"
                 f'        <path fill="none" stroke="#000" stroke-width="{HALO_W}" stroke-linecap="round" stroke-linejoin="round" d="{d}"/>\n'
                 f'        <circle cx="{f1(ax)}" cy="{f1(ay)}" r="{START_R + END_STROKE / 2}"/>\n'
                 f'        <circle cx="{f1(bx)}" cy="{f1(by)}" r="{END_R + END_STROKE / 2}"/>\n      </g>\n    </mask>\n')
    col, mesh = LAND[pal]
    land = (f"  <!-- Loss landscape with a single peak: wireframe and contour lines seen in perspective, hidden lines removed; the lines fade out towards the foothills; unsim/scripts/gen_surface.py -->\n"
            f'  <g {TF} fill="none" stroke="{col}" stroke-linejoin="round">\n'
            + "".join(f'    <path{f" stroke=\"{mesh}\"" if is_mesh else ""} stroke-width="{w}" stroke-opacity="{o:.3f}" d="{ld}"/>\n' for is_mesh, w, o, ld in geo["lines"])
            + "  </g>\n")
    if masked:
        land = f'  <g mask="url(#{p}-land)">\n' + sp.ind(land) + "  </g>\n"
    traj = ("  <!-- Steepest ascent to the summit; the white dots are iterates with geometrically shrinking steps -->\n"
            f"  <g {TF}>\n" '    <g fill="none" stroke-linecap="round" stroke-linejoin="round">\n')
    if halo:
        traj += f'      <path stroke="#fff" stroke-width="{HALO_W}" d="{d}"/>\n'
    traj += (f'      <path stroke="url(#{p}-route)" stroke-width="{ROUTE_W}" d="{d}"/>\n    </g>\n    <g fill="#fff">\n'
             + "".join(f'      <circle cx="{f1(x)}" cy="{f1(y)}" r="1.4"/>\n' for x, y in geo["iterates"]) + "    </g>\n"
             f'    <circle cx="{f1(ax)}" cy="{f1(ay)}" r="{START_R}" fill="{"#fff" if halo else "none"}" stroke="{ROUTE[pal][0]}" stroke-width="{END_STROKE}"/>\n'
             f'    <circle cx="{f1(bx)}" cy="{f1(by)}" r="{END_R}" fill="{ROUTE[pal][2]}"' + (f' stroke="#fff" stroke-width="{END_STROKE}"' if halo else "") + "/>\n  </g>\n")
    after_mark = "" if masked else (f"  <!-- White outline ({TEXT_GAP} px) under the wordmark and the tagline, so that the lines of the landscape stop short of the letters -->\n"
                                    + text_outlines(wm, tag, "#fff"))
    return sp.hero_doc(p, style, HERO_TITLE, wm, WM_COMMENT, tag, TAGLINE_TF, inner, defs, land + traj, after_mark)


def main():
    wm_paths = [glyphs.glyphs(group) for group in WORDMARK]
    wm = [d for d, _ in wm_paths]
    wm_xmax = wm_paths[1][1][2]
    tag = [d for d, _ in glyphs.text("Regular", TAGLINE, TAGLINE_GROUPS)]
    n_path, n_bbox = glyphs.glyphs(NABLA_N)
    geo = gen_surface.geometry(SURFACE)
    print(geo["info"])
    svg_dir = os.path.join(ROOT, NAME, "svg")
    os.makedirs(svg_dir, exist_ok=True)
    for style in sp.STYLES:
        p = f"{NAME}-icon-{style}"
        sp.write(os.path.join(svg_dir, p + ".svg"), sp.icon_doc(p, style, TITLE, nabla_n(style, "icon", n_path, n_bbox)))
        p = f"{NAME}-logo-{style}"
        sp.write(os.path.join(svg_dir, p + ".svg"), sp.logo_doc(p, style, TITLE, wm, wm_xmax, WM_COMMENT, nabla_n(style, "logo", n_path, n_bbox)))
        p = f"{NAME}-hero-{style}"
        sp.write(os.path.join(svg_dir, p + ".svg"), hero(style, wm, tag, nabla_n(style, "hero", n_path, n_bbox), geo))
    export_png.export(svg_dir, os.path.join(ROOT, NAME, "png"), os.path.join(ROOT, "tmp", "chrome-profile"))


if __name__ == "__main__":
    main()
