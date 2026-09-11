#!/usr/bin/env python3
"""
静止スキャン(会場)をLiDAR座標系に合わせ、3DGSの色をローポリメッシュに焼く。

  python3 bake_venue.py [--scan DIR] [--out venue.npz]

位置合わせは2段: まずZ軸まわりの回転+XY平行移動の3自由度を、床面へ投影した
占有マップのFFT相互相関で全探索し(矩形平面図の180°曖昧性は3Dで再評価して解く)、
次に点対平面ICPで6自由度に詰める(インライアRMS 3cm)。スケール誤差は0.33%と
実測されたので剛体で足りる。

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


def sample_mesh(verts, faces, n, seed=0, with_normals=False):
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
    pts = a[f] + (b[f] - a[f]) * u[:, None] + (c[f] - a[f]) * v[:, None]
    if not with_normals:
        return pts
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return pts, n[f]


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




class NearestSurface:
    """表面サンプルをボクセルに集約し、最寄りボクセルの代表点と法線をO(1)で引く。
    KD-treeが無い環境向け。BFSで最寄りIDを周囲へ伝播させておく。"""

    def __init__(self, pts, nrm, cell=0.06, max_d=6):
        self.cell = cell
        pad = cell * (max_d + 1)
        self.o = pts.min(0) - pad
        self.dims = np.ceil((pts.max(0) + pad - self.o) / cell).astype(np.int64) + 1
        k = ((pts - self.o) / cell).astype(np.int64)
        lin = (k[:, 0] * self.dims[1] + k[:, 1]) * self.dims[2] + k[:, 2]
        uniq, inv = np.unique(lin, return_inverse=True)
        cnt = np.bincount(inv).astype(np.float64)
        P = np.zeros((len(uniq), 3)); np.add.at(P, inv, pts); P /= cnt[:, None]
        N = np.zeros((len(uniq), 3)); np.add.at(N, inv, nrm)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        self.P, self.N = P, N
        near = np.full(tuple(self.dims), -1, np.int32)
        near[k[:, 0], k[:, 1], k[:, 2]] = inv.astype(np.int32)
        for _ in range(max_d):
            prev = near
            near = prev.copy()
            for ax in range(3):
                for sh in (1, -1):
                    cand = np.roll(prev, sh, axis=ax)
                    fill = (near < 0) & (cand >= 0)
                    near[fill] = cand[fill]
            if np.array_equal(near, prev):
                break
        self.near = near

    def query(self, q):
        k = ((q - self.o) / self.cell).astype(np.int64)
        ok = np.all((k >= 0) & (k < self.dims), axis=1)
        idx = np.full(len(q), -1, np.int32)
        idx[ok] = self.near[k[ok, 0], k[ok, 1], k[ok, 2]]
        good = idx >= 0
        return self.P[np.maximum(idx, 0)], self.N[np.maximum(idx, 0)], good


def icp_point_to_plane(src, ns, iters=40, d_start=0.40, d_end=0.10, verbose=True):
    """src(LiDAR点, 対象座標系) を表面へ寄せる剛体変換 (R,t) を返す。
    対応距離のしきい値を徐々に絞り、法線方向の残差を最小化する。"""
    R = np.eye(3); t = np.zeros(3)
    cur = src.copy()
    for it in range(iters):
        d_max = d_start + (d_end - d_start) * min(1.0, it / max(iters * 0.6, 1))
        q, n, good = ns.query(cur)
        diff = cur - q
        dist = np.abs((diff * n).sum(1))
        m = good & (dist < d_max)
        if m.sum() < 100:
            break
        p, qq, nn = cur[m], q[m], n[m]
        # 小回転 w と並進 v について線形化: ((w x p) + v - (p - q)).n = 0
        A = np.hstack([np.cross(p, nn), nn])            # (M,6)
        b = -((p - qq) * nn).sum(1)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        w, v = x[:3], x[3:]
        th = np.linalg.norm(w)
        if th > 1e-12:
            k = w / th
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            dR = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        else:
            dR = np.eye(3)
        cur = cur @ dR.T + v
        R = dR @ R; t = dR @ t + v
        rms = float(np.sqrt((dist[m] ** 2).mean()))
        if verbose and (it % 8 == 0 or it == iters - 1):
            print(f"    icp {it:2d}: d_max {d_max*100:4.0f}cm  inliers {m.sum():6d}  "
                  f"rms {rms*100:.1f}cm  |step| {th*57.3:.3f}deg {np.linalg.norm(v)*100:.2f}cm")
        if th < 1e-6 and np.linalg.norm(v) < 1e-5:
            break
    return R, t

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
    R3 = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T3 = np.array([t[0], t[1], dz])

    # ここまでは3自由度。残るロール/ピッチと細かなずれを点対平面ICPで詰める。
    # LiDAR点をメッシュ座標へ戻し、メッシュ表面へ寄せる剛体変換を求める。
    print("  refining with 6-DoF point-to-plane ICP ...")
    surf_n, nrm_n = sample_mesh(verts, faces, 1500000, seed=1, with_normals=True)
    ns = NearestSurface(surf_n, nrm_n)
    lid_m = (lid - T3) @ R3 + off                       # LiDAR -> メッシュ座標 (R3^-1)
    sub = lid_m[rng0.choice(len(lid_m), 60000, replace=False)]
    Ri, ti = icp_point_to_plane(sub, ns)
    # 合成: mesh -> lidar は  p_l = R3 (p_m - off) + T3、ICPは p_m' = Ri p_m + ti を
    # LiDAR側に施したものなので、逆に mesh 側へ取り込む: p_m = Ri^-1 (p_m' - ti)
    Rm = R3 @ Ri.T
    Tm = T3 - R3 @ (Ri.T @ ti)

    def to_lidar(p):
        return (p - off) @ Rm.T + Tm

    def to_mesh_icp(p_l):
        return (p_l - Tm) @ Rm + off

    v_l = to_lidar(verts)
    g_l = to_lidar(gxyz)
    ang = np.degrees(np.arccos(np.clip((np.trace(Ri) - 1) / 2, -1, 1)))
    print(f"  3-DoF: yaw {deg:.2f} deg  XY ({t[0]:+.2f}, {t[1]:+.2f})  dz {dz:+.3f}  一致 {100*best_s:.1f}%")
    print(f"  ICP補正: 回転 {ang:.3f} deg  並進 {np.linalg.norm(ti)*100:.1f} cm")
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
    # 距離グリッドの元は「メッシュ表面 ∪ ガウシアン中心」にする。ローポリメッシュは
    # 頂点間隔0.5m級で、音響壁のルーバー(奥行き10-15cm)のような細部を平面に均して
    # しまい、そこに当たったLiDAR点が"会場に無い"と誤判定される。ガウシアンは
    # 細部を持つが疎な場所もあるので、メッシュを下地として合成する。
    lo_b, hi_b = v_l.min(0) - 0.5, v_l.max(0) + 0.5      # 浮遊ガウシアンで格子が肥大化しないようメッシュ範囲に切る
    g_solid = g_l[(gopa > 0.3) & np.all((g_l > lo_b) & (g_l < hi_b), axis=1)]
    occ_src = np.vstack([surf_l, g_solid])
    print(f"  距離グリッド用: メッシュ表面 {len(surf_l)} + ガウシアン {len(g_solid)}")
    dist, dorg = distance_grid(occ_src, cell=0.05, max_d=16)
    hit = (dist < 255)
    print(f"  grid {tuple(dist.shape)} @5cm  "
          f"({dist.nbytes/1e6:.1f} MB, {100.0*hit.mean():.1f}% within 1.2 m)")
    probe_d = dist_lookup(dist, dorg, 0.05, lid)
    for thr in (5, 10, 15, 20, 30):
        print(f"  静止LiDAR点のうち会場から{thr:3d}cm超: "
              f"{100.0*(probe_d > thr).mean():.1f}%")
    band = (lid[:, 2] > 1.6) & (lid[:, 2] < 2.6) & (np.hypot(lid[:, 0], lid[:, 1]) < 10)
    print(f"  舞台域の壁帯(z 1.6-2.6m)で15cm超: {100.0*(probe_d[band] >= 15).mean():.1f}%  "
          f"(座奏の頭より上=主に壁。低いほど壁の誤検出が少ない)")

    # 会場を点群として描くモード用: ガウシアン中心と色 (不透明なものだけ、会場範囲内)
    gsel = (gopa > 0.3) & np.all((g_l > lo_b) & (g_l < hi_b), axis=1)
    gp = g_l[gsel].astype(np.float32); gc = np.clip(grgb[gsel] * 255, 0, 255).astype(np.uint8)
    if len(gp) > 900000:
        keep = np.random.default_rng(0).choice(len(gp), 900000, replace=False); gp, gc = gp[keep], gc[keep]
    print(f"  会場点群 (ガウシアン中心): {len(gp)} 点")
    # メッシュ座標 -> LiDAR座標 の4x4 (元データを同じ座標で扱う外部ツール用)
    M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = Tm - Rm @ off
    np.savez_compressed(args.out,
                        M_mesh_to_lidar=M.astype(np.float64),
                        gs_pos=gp, gs_rgb=gc,
                        verts=v_l.astype(np.float32),
                        colors=col.astype(np.float32),
                        faces=faces.astype(np.uint32),
                        dist=dist, dist_origin=dorg.astype(np.float32),
                        dist_cell=np.float32(0.05),
                        yaw_deg=np.float32(deg), txy=t.astype(np.float32),
                        dz=np.float32(dz))
    print(f"saved: {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
