#!/usr/bin/env python3
"""
静止スキャン(会場)をLiDAR座標系に合わせ、3DGSの色をローポリメッシュに焼く。

  python3 bake_venue.py [--scan DIR] [--out venue.npz]

Z軸は両者とも既に一致している(床がZ≒0, メートル, Z-up)ため、位置合わせは
Z軸まわりの回転+XY平行移動の3自由度に落ちる。床面へ投影した占有マップの
FFT相互相関でヨー角を全探索し、そのままXYシフトを読む。

色は3DGS(sh_degree=0なのでf_dcがそのままRGB)からボクセル平均で拾う。
"""
import argparse
import os
import sys

import numpy as np

SH0 = 0.28209479177387814
GS_STRIDE = 17            # x,y,z,nx,ny,nz,f_dc*3,opacity,scale*3,rot*4


def read_ply_header(path):
    """(ヘッダ長, 要素ごとの件数) を返す。"""
    counts, n_hdr = [], 0
    with open(path, "rb") as fh:
        buf = b""
        while b"end_header" not in buf:
            chunk = fh.read(4096)
            if not chunk:
                raise SystemExit(f"not a PLY: {path}")
            buf += chunk
        n_hdr = buf.index(b"end_header") + len(b"end_header\n")
        for line in buf[:n_hdr].split(b"\n"):
            if line.startswith(b"element"):
                counts.append(int(line.split()[2]))
    return n_hdr, counts


def load_mesh(path):
    n_hdr, (nv, nf) = read_ply_header(path)
    with open(path, "rb") as fh:
        fh.seek(n_hdr)
        verts = np.frombuffer(fh.read(nv * 12), np.float32).reshape(-1, 3)
        # face: uchar(3) + uint32*3 = 13 bytes
        raw = np.frombuffer(fh.read(nf * 13), np.uint8).reshape(nf, 13)
        if not np.all(raw[:, 0] == 3):
            raise SystemExit("mesh has non-triangular faces")
        faces = raw[:, 1:].copy().view("<u4").reshape(nf, 3)
    return verts.astype(np.float64), faces


def load_gaussians(path):
    n_hdr, (n,) = read_ply_header(path)
    with open(path, "rb") as fh:
        fh.seek(n_hdr)
        a = np.frombuffer(fh.read(n * GS_STRIDE * 4), np.float32)
    a = a.reshape(n, GS_STRIDE)
    xyz = a[:, 0:3].astype(np.float64)
    rgb = np.clip(a[:, 6:9] * SH0 + 0.5, 0.0, 1.0)
    opacity = 1.0 / (1.0 + np.exp(-a[:, 9]))
    return xyz, rgb, opacity


def sample_mesh(verts, faces, n, seed=0):
    """三角形の面積に比例して表面上に点を撒く。頂点をそのまま使うと、
    細かい造作に頂点が集中し大きな平面がスカスカになって相関が偏る。"""
    rng = np.random.default_rng(seed)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    cdf = np.cumsum(area)
    if cdf[-1] <= 0:
        raise SystemExit("degenerate mesh")
    t = rng.random(n) * cdf[-1]
    f = np.searchsorted(cdf, t)
    u = rng.random(n)
    v = rng.random(n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    return a[f] + (b[f] - a[f]) * u[:, None] + (c[f] - a[f]) * v[:, None]


def occupancy(xy, half, cell):
    """XY点群を原点中心の正方格子へ二値ラスタライズ。"""
    n = int(2 * half / cell)
    ij = np.floor((xy + half) / cell).astype(np.int64)
    ok = np.all((ij >= 0) & (ij < n), axis=1)
    g = np.zeros((n, n), np.float32)
    g[ij[ok, 0], ij[ok, 1]] = 1.0
    return g


class VoxelSet:
    """点群を占有ボクセルの集合にして、任意の点が載るか高速に判定する。"""

    def __init__(self, pts, vox):
        self.vox = vox
        self.o = pts.min(0) - vox
        self.d = np.ceil((pts.max(0) + vox - self.o) / vox).astype(np.int64) + 1
        k = ((pts - self.o) / vox).astype(np.int64)
        self.keys = np.unique((k[:, 0] * self.d[1] + k[:, 1]) * self.d[2] + k[:, 2])

    def frac_inside(self, pts):
        k = ((pts - self.o) / self.vox).astype(np.int64)
        inb = np.all((k >= 0) & (k < self.d), axis=1)
        if not inb.any():
            return 0.0
        q = (k[inb, 0] * self.d[1] + k[inb, 1]) * self.d[2] + k[inb, 2]
        p = np.clip(np.searchsorted(self.keys, q), 0, len(self.keys) - 1)
        return float((self.keys[p] == q).sum()) / len(pts)


def to_mesh_frame(p_lidar, deg, t, off, dz):
    """LiDAR座標 -> メッシュ座標 (順変換の逆)。"""
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    Rt = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])   # R^-1
    q = p_lidar.copy()
    q[:, 2] -= dz
    return (q - np.array([t[0], t[1], 0.0])) @ Rt.T + off


def yaw_candidates(lidar_xyz, surf, off, zlo, zhi, half=25.6, cell=0.10, keep=12):
    """床面へ投影した占有マップのFFT相関で (yaw, tx, ty) 候補を列挙する。
    矩形平面図は180°対称なのでここでは決め切れない。最終判定は3Dで行う。"""
    a = lidar_xyz[(lidar_xyz[:, 2] > zlo) & (lidar_xyz[:, 2] < zhi)][:, :2]
    b = surf[(surf[:, 2] > zlo) & (surf[:, 2] < zhi)][:, :2] - off[:2]
    A = np.fft.rfft2(occupancy(a, half, cell))
    n = int(2 * half / cell)

    out = []
    for deg in np.arange(0.0, 360.0, 1.0):
        th = np.deg2rad(deg)
        c, s = np.cos(th), np.sin(th)
        rb = np.column_stack([b[:, 0] * c - b[:, 1] * s,
                              b[:, 0] * s + b[:, 1] * c])
        corr = np.fft.irfft2(A * np.conj(np.fft.rfft2(occupancy(rb, half, cell))),
                             s=(n, n))
        k = int(np.argmax(corr))
        di, dj = k // n, k % n
        sx = di if di < n // 2 else di - n
        sy = dj if dj < n // 2 else dj - n
        out.append((float(corr.flat[k]), float(deg),
                    np.array([sx * cell, sy * cell])))
    out.sort(key=lambda r: -r[0])
    return out[:keep]


def distance_grid(surf, cell=0.10, max_d=12):
    """会場表面からの距離をボクセル単位で持つグリッド (uint8, 未到達は255)。
    これを配れば、動体判定のしきい値を再計算なしで実行時に変えられる。"""
    pad = cell * (max_d + 1)
    o = surf.min(0) - pad
    dims = np.ceil((surf.max(0) + pad - o) / cell).astype(np.int64) + 1
    k = ((surf - o) / cell).astype(np.int64)
    k = k[np.all((k >= 0) & (k < dims), axis=1)]
    cur = np.zeros(tuple(dims), bool)
    cur[k[:, 0], k[:, 1], k[:, 2]] = True
    dist = np.full(tuple(dims), 255, np.uint8)
    dist[cur] = 0
    for d in range(1, max_d + 1):
        nxt = cur.copy()
        nxt[1:] |= cur[:-1]
        nxt[:-1] |= cur[1:]
        nxt[:, 1:] |= cur[:, :-1]
        nxt[:, :-1] |= cur[:, 1:]
        nxt[:, :, 1:] |= cur[:, :, :-1]
        nxt[:, :, :-1] |= cur[:, :, 1:]
        dist[nxt & ~cur] = d
        cur = nxt
    return dist, o


def dist_lookup(dist, origin, cell, pts):
    """各点の会場表面からの距離をcmで返す (グリッド外・未到達は255)。"""
    dims = np.array(dist.shape)
    k = ((pts - origin) / cell).astype(np.int64)
    ok = np.all((k >= 0) & (k < dims), axis=1)
    out = np.full(len(pts), 255, np.int32)
    kk = k[ok]
    v = dist[kk[:, 0], kk[:, 1], kk[:, 2]].astype(np.int32)
    out[ok] = np.where(v == 255, 255,
                       np.minimum(v * int(round(cell * 100)), 254))
    return out.astype(np.uint8)


def bake_colors(verts, gxyz, grgb, vox=0.08, rings=2):
    """各頂点の近傍ボクセルにあるガウシアンの平均色を割り当てる。"""
    origin = gxyz.min(0) - vox
    dims = np.ceil((gxyz.max(0) + vox - origin) / vox).astype(np.int64) + 1
    key = np.floor((gxyz - origin) / vox).astype(np.int64)
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]

    order = np.argsort(lin, kind="stable")
    lin_s, rgb_s = lin[order], grgb[order]
    uniq, start = np.unique(lin_s, return_index=True)
    sums = np.add.reduceat(rgb_s.astype(np.float64), start, axis=0)
    cnts = np.diff(np.append(start, len(lin_s))).astype(np.float64)
    mean = sums / cnts[:, None]

    vk = np.floor((verts - origin) / vox).astype(np.int64)
    acc = np.zeros((len(verts), 3))
    hits = np.zeros(len(verts))
    offs = range(-rings, rings + 1)
    for dx in offs:
        for dy in offs:
            for dz in offs:
                q = ((vk[:, 0] + dx) * dims[1] + (vk[:, 1] + dy)) * dims[2] \
                    + (vk[:, 2] + dz)
                pos = np.searchsorted(uniq, q)
                pos = np.clip(pos, 0, len(uniq) - 1)
                hit = uniq[pos] == q
                w = 1.0 / (1.0 + dx * dx + dy * dy + dz * dz)
                acc[hit] += mean[pos[hit]] * w
                hits[hit] += w
    found = hits > 0
    col = np.full((len(verts), 3), 0.45)
    col[found] = acc[found] / hits[found, None]
    return col, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "DATA", "20260701_可視化用技研講堂3次元データ（静止画）", "data"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "venue.npz"))
    ap.add_argument("--frames", type=int, default=24,
                    help="LiDAR静止構造を作るのに中央値を取る枚数")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import server as S

    mesh_p = os.path.join(args.scan, "mesh", "mesh.ply")
    gs_p = os.path.join(args.scan, "ply-result", "point_cloud",
                        "iteration_100", "point_cloud.ply")
    for p in (mesh_p, gs_p):
        if not os.path.exists(p):
            raise SystemExit(f"not found: {p}")

    print("[1/4] loading scan ...")
    verts, faces = load_mesh(mesh_p)
    gxyz, grgb, gopa = load_gaussians(gs_p)
    print(f"  mesh {len(verts)} verts / {len(faces)} tris")
    print(f"  gaussians {len(gxyz)}  (opacity p50 {np.median(gopa):.2f})")

    print("[2/4] building LiDAR static reference ...")
    src = S.OusterPcap(S.DEFAULT_PCAP, S.DEFAULT_META)
    idx = np.linspace(0, src.n_frames - 1, args.frames).astype(int)
    st = np.stack([src._decode_raw(int(i))[0] for i in idx]).astype(np.float32)
    st[st == 0] = np.nan
    med = np.nanmedian(st, axis=0)
    med[np.isnan(med)] = 0.0
    xyz = src.lut_dir * med[None] + src.lut_off
    lid = np.moveaxis(xyz, 0, -1).reshape(-1, 3)[med.reshape(-1) > 0]
    lid = lid.astype(np.float64)
    print(f"  static points {len(lid)}  (人や移動体は中央値で除去)")

    print("[3/4] registering (yaw + XY) ...")
    zlo, zhi = 0.5, 3.5
    surf = sample_mesh(verts, faces, 1500000)
    ctr = surf[(surf[:, 2] > zlo) & (surf[:, 2] < zhi)][:, :2].mean(0)
    off = np.array([ctr[0], ctr[1], 0.0])
    dz = np.percentile(lid[:, 2], 1) - np.percentile(surf[:, 2], 1)

    vs = VoxelSet(surf, 0.20)
    rng0 = np.random.default_rng(0)
    probe = lid[rng0.choice(len(lid), 40000, replace=False)]

    def score3d(deg, t):
        return vs.frac_inside(to_mesh_frame(probe, deg, t, off, dz))

    cands = yaw_candidates(lid, surf, off, zlo, zhi)
    print("  2D相関の上位候補を3Dで再評価:")
    scored = [(score3d(d, t), d, t) for _, d, t in cands]
    scored.sort(key=lambda r: -r[0])
    for sc, d, t in scored[:5]:
        print(f"    yaw {d:6.1f} deg  XY ({t[0]:+6.2f},{t[1]:+6.2f})  一致 {100*sc:.1f}%")
    best_s, deg, t = scored[0]

    # 3D一致率を直接最大化して詰める (yaw 0.25°, XY 5cm)
    for _ in range(3):
        improved = False
        for dd in (-0.5, -0.25, 0.0, 0.25, 0.5):
            for dx in (-0.10, -0.05, 0.0, 0.05, 0.10):
                for dy in (-0.10, -0.05, 0.0, 0.05, 0.10):
                    cand_t = t + np.array([dx, dy])
                    sc = score3d(deg + dd, cand_t)
                    if sc > best_s:
                        best_s, deg, t, improved = sc, deg + dd, cand_t, True
        if not improved:
            break

    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.array([t[0], t[1], 0.0])

    def to_lidar(p):
        q = (p - off) @ R.T + T
        q[:, 2] += dz
        return q

    v_l = to_lidar(verts)
    g_l = to_lidar(gxyz)
    print(f"  確定: yaw {deg:.2f} deg   XY ({t[0]:+.2f}, {t[1]:+.2f}) m   "
          f"dz {dz:+.3f} m   一致 {100*best_s:.1f}%")
    print(f"  mesh bbox in LiDAR frame: "
          f"X {v_l[:,0].min():.1f}..{v_l[:,0].max():.1f}  "
          f"Y {v_l[:,1].min():.1f}..{v_l[:,1].max():.1f}  "
          f"Z {v_l[:,2].min():.1f}..{v_l[:,2].max():.1f}")

    # 位置合わせの品質: LiDAR点がメッシュ表面のボクセルに載る割合
    surf_l = to_lidar(surf)
    for vox in (0.15, 0.25, 0.40):
        o = surf_l.min(0) - vox
        d = np.ceil((surf_l.max(0) + vox - o) / vox).astype(np.int64) + 1
        sk = ((surf_l - o) / vox).astype(np.int64)
        mk = np.unique((sk[:, 0] * d[1] + sk[:, 1]) * d[2] + sk[:, 2])
        lk = ((lid - o) / vox).astype(np.int64)
        inb = np.all((lk >= 0) & (lk < d), axis=1)
        q = (lk[inb, 0] * d[1] + lk[inb, 1]) * d[2] + lk[inb, 2]
        p = np.clip(np.searchsorted(mk, q), 0, len(mk) - 1)
        print(f"  LiDAR点がメッシュ表面と一致 (voxel {vox*100:.0f}cm): "
              f"{100.0*(mk[p]==q).mean():.1f}%  "
              f"[bbox内 {100.0*inb.mean():.0f}%]")

    print("[4/5] baking colors from 3DGS ...")
    col, found = bake_colors(v_l, g_l, grgb)
    print(f"  colored {100.0*found.mean():.1f}% of vertices")

    print("[5/5] building distance grid (for dynamic-only extraction) ...")
    dist, dorg = distance_grid(surf_l)
    hit = (dist < 255)
    print(f"  grid {tuple(dist.shape)} @10cm  "
          f"({dist.nbytes/1e6:.1f} MB, {100.0*hit.mean():.1f}% within 1.2 m)")
    probe_d = dist_lookup(dist, dorg, 0.10, lid)
    for thr in (10, 20, 30, 40):
        print(f"  静止LiDAR点のうち会場から{thr:3d}cm超: "
              f"{100.0*(probe_d > thr).mean():.1f}%")

    np.savez_compressed(args.out,
                        verts=v_l.astype(np.float32),
                        colors=col.astype(np.float32),
                        faces=faces.astype(np.uint32),
                        dist=dist, dist_origin=dorg.astype(np.float32),
                        dist_cell=np.float32(0.10),
                        yaw_deg=np.float32(deg), txy=t.astype(np.float32),
                        dz=np.float32(dz))
    print(f"saved: {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
