# Geometry of the UNsim hero (1280x592) landscape: a single-peaked loss landscape drawn as a 3D wireframe with contour lines that
# fade out towards the foothills, and a steepest-ascent trajectory that climbs to the summit. geometry() returns the screen polylines;
# unsim/scripts/build.py turns them into SVG (and holds the arguments used for the hero).
# Usage for a summary: python unsim/scripts/gen_surface.py [key=value ...]   (CFG holds the view, the line widths and opacities and the start of the climb; the arguments override it)
# Ground coordinates: p to the right, q away from the viewer, height h up (all in px). Hidden lines are removed by marching from each sample point towards the camera.
# Screen coordinates are those of an earlier hero whose mark sat at translate(130.92 90.72) scale(0.88); the SVG places them with TF,
# which scales them by 1.17 about (650, 316) and moves that point to (640, 296.218), the centre of the mark.
import math, sys
import numpy as np
import contourpy
from scipy.optimize import minimize

TF = 'transform="translate(-120.5 -73.502) scale(1.17)"'
W, H_CANVAS = 1280, 592
IX0, IX1, IY0, IY1 = 650 - 640 / 1.17, 650 + 640 / 1.17, 316 - 296.218 / 1.17, 316 + (H_CANVAS - 296.218) / 1.17   # the canvas in these coordinates
def f1(v): return f"{v:.1f}".replace("-0.0", "0.0")

def simplify(pts, tol):
    # Douglas-Peucker
    pts = np.asarray(pts, float)
    if len(pts) < 3: return pts
    keep = np.zeros(len(pts), bool); keep[0] = keep[-1] = True; stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        a, b = pts[i], pts[j]; ab = b - a; n = np.hypot(*ab)
        seg = pts[i + 1:j]
        if len(seg) == 0: continue
        d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / n if n > 0 else np.hypot(*(seg - a).T)
        k = int(np.argmax(d))
        if d[k] > tol: keep[i + 1 + k] = True; stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]
def path_d(pts):
    # absolute moveto, relative lineto
    s = f"M{f1(pts[0][0])} {f1(pts[0][1])}"
    px, py = round(pts[0][0], 1), round(pts[0][1], 1)
    out = []
    for x, y in pts[1:]:
        x, y = round(x, 1), round(y, 1)
        out.append(f"{f1(x - px)} {f1(y - py)}"); px, py = x, y
    return s + ("l" + " ".join(out) if out else "")
CFG = dict(elev=30.0, azim=-45.0, dist=1600.0, X0=640.0, Y0=316.0, k=1.0, mesh=24.0, dz=16.0, start="636,185",
           px=900.0, py=112.0, sp=140.0, sq=120.0, hp=190.0, rot=0.0,
           mw=0.6, mo=0.8, cw=0.9, co=0.5, t0=0.04, t1=0.5, bins=10)
# outlines of the letters (screen boxes), for the clearance of the trajectory: U, N, s, dot of i, stem of i, m, tagline
TEXT = [(604.5, 221.3, 704.3, 346.0), (711.9, 221.1, 839.5, 344.8), (838.7, 245.2, 920.1, 346.3), (930.7, 206.1, 961.2, 235.1),
        (933.4, 246.7, 958.1, 344.8), (975.5, 245.3, 1137.3, 344.8), (604.5, 378.5, 1137.3, 409.2)]

# ---------- landscape: a single peak, placed so that its summit projects to (px, py); ground sizes sp, sq; height hp; rotation rot in deg ----------
FEAT, H = None, None   # set by fragments()
def height(p, q):
    cp, cq, sp, sq, A, rot = FEAT
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    p = np.asarray(p, float) - cp; q = np.asarray(q, float) - cq
    a, b = p * c + q * s, -p * s + q * c
    return A * np.exp(-0.5 * ((a / sp) ** 2 + (b / sq) ** 2))
def grad(p, q, e=0.5): return ((height(p + e, q) - height(p - e, q)) / (2 * e), (height(p, q + e) - height(p, q - e)) / (2 * e))

# ---------- camera: orthographic (dist = 0) or perspective; the ground origin projects to (X0, Y0) ----------
class Cam:
    def __init__(s, c):
        th = math.radians(c["elev"]); s.st, s.ct = math.sin(th), math.cos(th)
        s.X0, s.Y0, s.k, s.L = c["X0"], c["Y0"], c["k"], c["dist"]
    def project(s, p, q, h):
        yc = q * s.st + h * s.ct; zc = q * s.ct - h * s.st
        f = s.L / np.maximum(s.L + zc, 1e-6) if s.L > 0 else 1.0
        return s.X0 + s.k * p * f, s.Y0 - s.k * yc * f
    def place(s, X, Y, h):
        # point (p, q) whose surface point at height h projects to (X, Y)
        m = (s.Y0 - Y) / s.k
        q = (h * s.ct * s.L + m * h * s.st - m * s.L) / (m * s.ct - s.st * s.L) if s.L > 0 else (m - h * s.ct) / s.st
        f = s.L / (s.L + q * s.ct - h * s.st) if s.L > 0 else 1.0
        return (X - s.X0) / (s.k * f), q
    def front(s, p, q, h):
        # in front of the camera (always true for the orthographic view)
        return (s.L + q * s.ct - h * s.st > 10) if s.L > 0 else np.ones(np.shape(p), bool)
    def hidden(s, p, q, h, hmax, dt=2.0, eps=0.4, t0=1.5):
        if s.L > 0:
            dx, dy, dz = -p, -s.L * s.ct - q, s.L * s.st - h; n = np.sqrt(dx * dx + dy * dy + dz * dz); dx, dy, dz = dx / n, dy / n, dz / n
        else:
            dx, dy, dz = np.zeros(p.size), np.full(p.size, -s.ct), np.full(p.size, s.st)
        hid = np.zeros(p.size, bool); idx = np.arange(p.size); t = t0
        while idx.size:
            pp, qq, hh = p[idx] + t * dx[idx], q[idx] + t * dy[idx], h[idx] + t * dz[idx]
            occ = height(pp, qq) > hh + eps
            hid[idx[occ]] = True
            idx = idx[~occ & (hh < hmax)]
            t += dt
        return hid

def surface_point(cam, X, Y):
    # visible point (p, q) on the surface that projects to (X, Y): the highest crossing of the line of sight with the surface, found by scanning
    # the height downwards from the summit and bisecting
    g = lambda h: float(height(*cam.place(X, Y, h))) - h
    hs = np.linspace(H, 0.0, 401)
    k = next(i for i in range(1, len(hs)) if g(hs[i]) >= 0)
    lo, hi = hs[k], hs[k - 1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if g(mid) >= 0: lo = mid
        else: hi = mid
    return np.array(cam.place(X, Y, lo))
def alpha(h, c):
    # opacity factor of the lines: 0 at the foothills (h <= t0 H), 1 above t1 H, smoothstep in between
    x = np.clip((np.asarray(h, float) / H - c["t0"]) / (c["t1"] - c["t0"]), 0, 1)
    return x * x * (3 - 2 * x)

def polylines(curves, cam, hmax, pad=6):
    # visible parts of curves on the surface ((N, 3) arrays): screen polylines inside the canvas, with the heights of their points
    if not curves: return []
    P = np.vstack(curves); lid = np.repeat(np.arange(len(curves)), [len(cv) for cv in curves])
    X, Y = cam.project(P[:, 0], P[:, 1], P[:, 2])
    ok = cam.front(P[:, 0], P[:, 1], P[:, 2]) & (X > IX0 - pad) & (X < IX1 + pad) & (Y > IY0 - pad) & (Y < IY1 + pad)
    i = np.where(ok)[0]
    ok[i] = ~cam.hidden(P[i, 0], P[i, 1], P[i, 2], hmax)
    idx = np.where(ok)[0]
    if idx.size == 0: return []
    brk = np.where((np.diff(idx) > 1) | (np.diff(lid[idx]) != 0))[0] + 1
    return [(np.column_stack([X[g], Y[g]]), P[g, 2]) for g in np.split(idx, brk) if len(g) >= 2]

def by_opacity(runs, c):
    # cut each polyline where its opacity, quantised to 1/bins, changes; {level: [polylines]} (level 0 is dropped)
    out = {}
    for xy, h in runs:
        k = np.rint(alpha(h, c) * c["bins"]).astype(int); i = 0
        for j in range(1, len(k) + 1):
            if j == len(k) or k[j] != k[i]:
                piece = xy[i:min(j + 1, len(k))]
                if k[i] > 0 and len(piece) >= 2: out.setdefault(int(k[i]), []).append(piece)
                i = j
    return out

def ground_box(cam, hmax, pad=40):
    # part of the ground that can appear inside the canvas
    PP, QQ = np.meshgrid(np.linspace(-3000, 3000, 601), np.linspace(-3000, 6000, 901))
    ok = np.zeros(PP.shape, bool)
    for h in (0.0, hmax):
        Hh = np.full(PP.shape, h); X, Y = cam.project(PP, QQ, Hh)
        ok |= cam.front(PP, QQ, Hh) & (X > IX0 - pad) & (X < IX1 + pad) & (Y > IY0 - pad) & (Y < IY1 + pad)
    return PP[ok].min() - 20, PP[ok].max() + 20, QQ[ok].min() - 20, QQ[ok].max() + 20

def mesh(c, cam, box, hmax):
    # lines of constant u and constant v of the mesh, which is turned by azim against the screen
    P0, P1, Q0, Q1 = box; ph = math.radians(c["azim"]); cph, sph = math.cos(ph), math.sin(ph)
    cs = [(P0, Q0), (P0, Q1), (P1, Q0), (P1, Q1)]
    U = [p * cph + q * sph for p, q in cs]; V = [-p * sph + q * cph for p, q in cs]
    d = c["mesh"]; curves = []
    for fam in (0, 1):
        A, B = (U, V) if fam == 0 else (V, U)
        b = np.arange(min(B), max(B), 2.0)
        for a in np.arange(math.floor(min(A) / d) * d, max(A) + d, d):
            u, v = (np.full(b.size, a), b) if fam == 0 else (b, np.full(b.size, a))
            p, q = u * cph - v * sph, u * sph + v * cph
            k = (p >= P0) & (p <= P1) & (q >= Q0) & (q <= Q1)
            if k.sum() >= 2: curves.append(np.column_stack([p[k], q[k], height(p[k], q[k])]))
    return polylines(curves, cam, hmax)

def contours(c, cam, box, hmax):
    P0, P1, Q0, Q1 = box
    ps, qs = np.arange(P0, P1 + 1, 2.5), np.arange(Q0, Q1 + 1, 2.5)
    gen = contourpy.contour_generator(ps, qs, height(*np.meshgrid(ps, qs)))
    curves = []
    for L in np.arange(c["dz"], hmax, c["dz"]):
        curves += [np.column_stack([ln, np.full(len(ln), L)]) for ln in gen.lines(L) if len(ln) >= 4]
    return polylines(curves, cam, hmax)

def trajectory(c, cam, hmax):
    # steepest ascent from the surface point under the screen point "start", with 0.5 px steps along the normalised gradient, until the steps
    # turn back; then the summit itself
    p, q = surface_point(cam, *map(float, c["start"].split(","))); pts = [(p, q)]; prev = None
    for _ in range(40000):
        gp, gq = grad(p, q); n = math.hypot(gp, gq)
        if n < 1e-9: break
        st = (gp / n, gq / n)
        if prev is not None and st[0] * prev[0] + st[1] * prev[1] < 0: break
        p, q = p + 0.5 * st[0], q + 0.5 * st[1]; pts.append((p, q)); prev = st
    top = minimize(lambda x: -float(height(x[0], x[1])), [p, q], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-9)).x
    pts.append(tuple(top))
    G = np.array(pts); Hp = height(G[:, 0], G[:, 1]) + 0.3
    X, Y = cam.project(G[:, 0], G[:, 1], Hp)
    nhid = int(cam.hidden(G[:, 0], G[:, 1], Hp, hmax).sum())
    # iterates: positions along the 3D path with geometrically shrinking steps (as in proposal 1)
    S = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(G[:, 0]) ** 2 + np.diff(G[:, 1]) ** 2 + np.diff(Hp) ** 2))]); tot = S[-1]
    s, d, its = 0.0, tot * (1 - 0.68), []
    while d > 4:
        s += d; d *= 0.68; j = min(int(np.searchsorted(S, s)), len(S) - 1); its.append((X[j], Y[j]))
    # clearance from the letters
    gap = min(float(np.min(np.hypot(np.maximum(np.maximum(x0 - X, 0), X - x1), np.maximum(np.maximum(y0 - Y, 0), Y - y1)))) for x0, y0, x1, y1 in TEXT)
    return np.column_stack([X, Y]), its, f"hidden {nhid}, clearance {gap:.1f}, start height {Hp[0] / H:.2f} H", tot

def geometry(over=None):
    # lines: [(is_mesh, stroke width, opacity, path data)] from the faintest to the strongest; traj: path data of the trajectory;
    # iterates: [(x, y)]; start, end: (x, y) of the foot and of the summit; all in the TF coordinates
    global FEAT, H
    c = dict(CFG); c.update({k: type(CFG[k])(v) for k, v in (over or {}).items() if k in CFG})
    cam = Cam(c); H = hmax = c["hp"]
    FEAT = (*cam.place(c["px"], c["py"], H), c["sp"], c["sq"], H, c["rot"])
    box = ground_box(cam, hmax)
    lines = []
    for is_mesh, runs, w, o in ((True, mesh(c, cam, box, hmax), c["mw"], c["mo"]), (False, contours(c, cam, box, hmax), c["cw"], c["co"])):
        for lev, pls in sorted(by_opacity(runs, c).items()):
            lines.append((is_mesh, w, o * lev / c["bins"], " ".join(path_d(simplify(pl, 0.35)) for pl in pls)))
    T, its, tinfo, tot = trajectory(c, cam, hmax)
    Ts = simplify(T, 0.25); a, b = T[0], T[-1]
    info = f"box {[round(v) for v in box]} paths {len(lines)} chars {sum(len(l[3]) for l in lines)} traj {f1(a[0])},{f1(a[1])} -> {f1(b[0])},{f1(b[1])} len {tot:.0f} {tinfo}"
    return dict(lines=lines, traj=path_d(Ts), iterates=its, start=(a[0], a[1]), end=(b[0], b[1]), info=info)

if __name__ == "__main__":
    print(geometry(dict(a.split("=", 1) for a in sys.argv[1:]))["info"])
