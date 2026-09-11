#!/usr/bin/env python3
"""撮影機材などを消した会場データを、元のスキャンと同じ形式で新規保存する。
   消去領域(venue_erase.json)はLiDAR座標なので、元データをLiDAR座標へ変換して判定し、
   出力は元の座標系のまま (他ツールで元データと同様に扱える)。"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bake_venue as B, server as S

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.join(os.path.dirname(HERE), "DATA", "20260701_可視化用技研講堂3次元データ（静止画）", "data")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "DATA", "cleaned")
os.makedirs(OUT, exist_ok=True)

z = np.load(os.path.join(HERE, "venue.npz")); M = z["M_mesh_to_lidar"]
cfg = json.load(open(os.path.join(HERE, "venue_erase.json")))
to_lidar = lambda p: p @ M[:3, :3].T + M[:3, 3]

# --- 3DGS: 全プロパティを保ったまま行を落とす ---
gs_path = os.path.join(SCAN, "ply-result", "point_cloud", "iteration_100", "point_cloud.ply")
n_hdr, (n,) = B.read_ply_header(gs_path)
with open(gs_path, "rb") as fh:
    hdr = fh.read(n_hdr); a = np.frombuffer(fh.read(n * B.GS_STRIDE * 4), np.float32).reshape(n, B.GS_STRIDE)
kill = S.erase_mask(to_lidar(a[:, :3].astype(np.float64)), cfg)
keep = a[~kill]
hdr2 = hdr.replace(f"element vertex {n}".encode(), f"element vertex {len(keep)}".encode(), 1)
p = os.path.join(OUT, "point_cloud_cleaned.ply")
with open(p, "wb") as fh: fh.write(hdr2); fh.write(np.ascontiguousarray(keep).tobytes())
print(f"3DGS: {n} -> {len(keep)} gaussians (erased {int(kill.sum())})  -> {p}  {os.path.getsize(p)/1e6:.1f} MB")

# --- メッシュ (ローポリ / 高解像度): 消す頂点に触れる面を落とし、頂点は詰め直す ---
def clean_mesh(src, dst):
    v, f = B.load_mesh(src)
    kill = S.erase_mask(to_lidar(v), cfg)
    f2 = f[~np.any(kill[f], axis=1)]
    used = np.zeros(len(v), bool); used[f2.reshape(-1)] = True
    remap = np.full(len(v), -1, np.int64); remap[used] = np.arange(used.sum())
    v2 = v[used].astype(np.float32); f3 = remap[f2].astype(np.uint32)
    hdr = (f"ply\nformat binary_little_endian 1.0\ncomment cleaned: filming equipment removed\n"
           f"element vertex {len(v2)}\nproperty float x\nproperty float y\nproperty float z\n"
           f"element face {len(f3)}\nproperty list uchar uint vertex_indices\nend_header\n").encode()
    faces = np.hstack([np.full((len(f3), 1), 3, np.uint8).view(np.uint8),
                       np.ascontiguousarray(f3).view(np.uint8).reshape(len(f3), 12)]).astype(np.uint8)
    with open(dst, "wb") as fh: fh.write(hdr); fh.write(v2.tobytes()); fh.write(faces.tobytes())
    print(f"mesh: {len(v)} verts/{len(f)} tris -> {len(v2)}/{len(f3)} (erased verts {int(kill.sum())})  -> {dst}  {os.path.getsize(dst)/1e6:.1f} MB")
clean_mesh(os.path.join(SCAN, "mesh", "mesh.ply"), os.path.join(OUT, "mesh_cleaned.ply"))
clean_mesh(os.path.join(SCAN, "mesh", "mesh_raw.ply"), os.path.join(OUT, "mesh_raw_cleaned.ply"))

# --- LiDAR座標系の色付き点群 (ガウシアン中心 + RGB): ビューアと同じ座標で使う用 ---
gp = z["gs_pos"].astype(np.float32); gc = z["gs_rgb"]
k = ~S.erase_mask(gp.astype(np.float64), cfg); gp, gc = gp[k], gc[k]
p = os.path.join(OUT, "venue_points_lidar_frame.ply")
hdr = (f"ply\nformat binary_little_endian 1.0\ncomment LiDAR frame (metres, Z up); cleaned\n"
       f"element vertex {len(gp)}\nproperty float x\nproperty float y\nproperty float z\n"
       f"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode()
rec = np.zeros(len(gp), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
rec["x"], rec["y"], rec["z"] = gp.T; rec["r"], rec["g"], rec["b"] = gc.T
with open(p, "wb") as fh: fh.write(hdr); fh.write(rec.tobytes())
print(f"points (LiDAR frame): {len(gp)}  -> {p}  {os.path.getsize(p)/1e6:.1f} MB")
json.dump({"erase": cfg, "M_mesh_to_lidar": M.tolist()}, open(os.path.join(OUT, "cleaned_meta.json"), "w"), ensure_ascii=False, indent=1)
print("meta ->", os.path.join(OUT, "cleaned_meta.json"))
