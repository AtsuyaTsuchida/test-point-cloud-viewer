#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ouster OS-DOME-128 pcap → dynamic 3D point cloud viewer server

Reads a pcap recording and its sensor metadata JSON, decodes frames into
XYZ point clouds on demand, and serves them to the WebGL viewer
(viewer.html) in the browser.

On first launch the whole pcap is scanned to build a frame index, cached
as "<pcap>.viewidx.npz" (subsequent launches start instantly).

Usage:
    python3 server.py [--pcap PATH] [--meta PATH] [--port 8765] [--no-open]
"""
import argparse
import json
import mmap
import os
import struct
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))

RAW_CACHE = 48           # 生の距離画像のLRU段数 (時間中央値フィルタ用, 約1.2MB/枚)
TAN_MAX = 3.0            # 面とみなす最大入射角 tan (72°) — 法線の有効判定に使う
MESH_TAN = 12.0          # メッシュの面連続判定 tan (85°)。斜入射でも面は繋げる。
REL_NOISE = 0.02         # レンジノイズぶんの相対マージン
SIG_CAP_K = 0.010        # σの距離比上限。斜入射面の標本間隔は1m超になるため、
SIG_CAP_B = 0.005        # 忠実に覆うと筋状に伸びる。覆いきらず隙間を残す方を選ぶ。
                         # 値はフォールバック半径 rr*0.0092 に合わせてある。
MAX_ASPECT = 2.0         # 楕円の最大扁平率
STAGE_R = 10.0           # 会場距離による前景判定を適用する範囲 (センサーからの水平距離)
STAGE_Z = 2.6            # 同、高さ上限。天井や高所のスキャン欠損を前景にしないための空間事前
MIN_CLUSTER = 12         # 前景クラスタの最小画素数。単発のレンジノイズを落とす
GROW_VMIN = 8            # 成長がスキャン表面(会場距離がこれ未満)へ入るのを禁止する [cm]
LOG_R_BASE = 1.0182      # 半径のu8対数エンコード: r = 0.004 * BASE**k
LOG_R_MIN = 0.004

DEFAULT_DIR = os.path.join(
    os.path.dirname(HERE),
    "DATA", "20260701_可視化本映像３次元データ（動画）")
DEFAULT_PCAP = os.path.join(DEFAULT_DIR, "2026071_2007_OS-DOME-128_122333000992.pcap")
DEFAULT_META = os.path.join(DEFAULT_DIR, "2026071_2007_OS-DOME-128_122333000992.json")

PCAP_GLOBAL_HDR = 24
REC_HDR = 16
ETH_IP_UDP = 42          # Ethernet(14) + IPv4(20) + UDP(8)
PKT_HDR = 32             # Ouster eUDP パケットヘッダ
PKT_FTR = 32             # Ouster eUDP パケットフッタ
COL_HDR = 12             # timestamp(8) + measurement_id(2) + status(2)
PX_SIZE = 12             # RNG19_RFL8_SIG16_NIR16 の1ピクセル


class OusterPcap:
    def __init__(self, pcap_path, meta_path):
        self.pcap_path = pcap_path
        with open(meta_path, "r") as fh:
            self.meta = json.load(fh)

        fmt = self.meta["lidar_data_format"]
        prof = fmt["udp_profile_lidar"]
        if prof != "RNG19_RFL8_SIG16_NIR16":
            raise SystemExit(f"Unsupported UDP profile: {prof}")
        self.H = fmt["pixels_per_column"]        # 128
        self.W = fmt["columns_per_frame"]        # 1024
        self.cols_per_packet = fmt["columns_per_packet"]  # 16
        self.col_size = COL_HDR + self.H * PX_SIZE
        self.payload_size = PKT_HDR + self.cols_per_packet * self.col_size + PKT_FTR

        f = open(pcap_path, "rb")
        self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        self._bg = None                # 時間中央値の背景レンジ画像
        self._raw = {}                 # frame idx -> (rng, refl, sig, nir)
        self._raw_order = []
        self._raw_lock = threading.Lock()

        # OS-DOME-128 の beam_altitude_angles は空間順ではなく列内でインターリーブ
        # されている(隣接要素が5°飛ぶ、符号反転27回)。行を仰角順に並べ替えないと
        # 「隣の行」が空間的な隣にならず、法線も面の接続も成立しない。
        alt_raw = np.deg2rad(np.asarray(
            self.meta["beam_intrinsics"]["beam_altitude_angles"], np.float64))
        self.row_perm = np.argsort(alt_raw)[::-1].copy()    # 上(仰角大)から下へ
        alt = alt_raw[self.row_perm]

        # 深度エッジ判定用の角度ステップ(隣接±1本ぶん)。ビーム間隔は0.02°〜2.06°と
        # 一様でないので行ごとに持つ。
        sv = np.empty(self.H)
        sv[1:-1] = np.abs(alt[2:] - alt[:-2])
        sv[0] = abs(alt[1] - alt[0])
        sv[-1] = abs(alt[-1] - alt[-2])
        self.step_v = sv.astype(np.float32)
        self.step_u = np.float32(2.0 * (2.0 * np.pi / self.W))

        self._build_lut()
        self._index()

        dt = np.diff(self.frame_ts)
        dt = dt[(dt > 1e-4) & (dt < 1.0)]
        self.fps = float(1.0 / np.median(dt)) if len(dt) else 20.0

    # --- XYZ変換ルックアップテーブル (ouster-sdk make_xyz_lut と同じ式) ---
    def _build_lut(self):
        bi = self.meta["beam_intrinsics"]
        # 行は仰角順に並べ替えた順序で構築する (self.row_perm と一致させること)
        alt = np.deg2rad(np.asarray(bi["beam_altitude_angles"],
                                    np.float64))[self.row_perm]
        az = -np.deg2rad(np.asarray(bi["beam_azimuth_angles"],
                                    np.float64))[self.row_perm]
        b = np.asarray(bi["beam_to_lidar_transform"], np.float64).reshape(4, 4)
        n = float(np.hypot(b[0, 3], b[2, 3]))

        theta_e = 2.0 * np.pi * (1.0 - np.arange(self.W) / self.W)   # (W,)
        theta = theta_e[None, :] + az[:, None]                        # (H,W)
        cos_alt = np.cos(alt)[:, None]
        dirx = np.cos(theta) * cos_alt
        diry = np.sin(theta) * cos_alt
        dirz = np.broadcast_to(np.sin(alt)[:, None], (self.H, self.W)).copy()
        offx = np.cos(theta_e)[None, :] * b[0, 3] - n * dirx
        offy = np.sin(theta_e)[None, :] * b[0, 3] - n * diry
        offz = b[2, 3] - n * dirz

        L = np.asarray(self.meta["lidar_intrinsics"]["lidar_to_sensor_transform"],
                       np.float64).reshape(4, 4)
        R, t = L[:3, :3], L[:3, 3]
        d = np.einsum("ij,jhw->ihw", R, np.stack([dirx, diry, dirz]))
        o = np.einsum("ij,jhw->ihw", R, np.stack([offx, offy, offz])) + t[:, None, None]

        # xyz[m] = range[mm] * lut_dir + lut_off
        self.lut_dir = (d / 1000.0).astype(np.float32)
        self.lut_off = (o / 1000.0).astype(np.float32)

    # --- pcap 全体を1回スキャンしてフレーム索引を作る ---
    def _index(self):
        cache = self.pcap_path + ".viewidx.npz"
        st = os.stat(self.pcap_path)
        key = f"{st.st_size}:{int(st.st_mtime)}"

        if os.path.exists(cache):
            try:
                z = np.load(cache)
                if str(z["key"]) == key:
                    self.pkt_off = z["pkt_off"]
                    self.frame_start = z["frame_start"]
                    self.frame_count = z["frame_count"]
                    self.frame_ts = z["frame_ts"]
                    print(f"Index cache loaded: {len(self.frame_start)} frames")
                    return
                print("Index cache is stale, rebuilding")
            except Exception as e:
                print("Failed to load index cache:", e)

        print("Indexing pcap (first run only, may take a while)...")
        mm, N = self.mm, len(self.mm)
        expect_caplen = self.payload_size + ETH_IP_UDP
        unpack = struct.unpack_from
        offs, fids, tss = [], [], []
        pos = PCAP_GLOBAL_HDR
        t0 = time.time()
        while pos + REC_HDR <= N:
            ts, tus, caplen, _wire = unpack("<IIII", mm, pos)
            data_pos = pos + REC_HDR
            if data_pos + caplen > N:
                break
            if caplen == expect_caplen:
                p = data_pos + ETH_IP_UDP
                fids.append(unpack("<H", mm, p + 2)[0])
                offs.append(p)
                tss.append(ts + tus * 1e-6)
                if len(offs) % 100000 == 0:
                    print(f"  {len(offs)} packets / {pos/N*100:.0f}%")
            pos = data_pos + caplen

        fids = np.asarray(fids, np.int64)
        offs = np.asarray(offs, np.int64)
        tss = np.asarray(tss, np.float64)
        if len(fids) == 0:
            raise SystemExit("No LiDAR packets found")

        change = np.nonzero(np.diff(fids) != 0)[0] + 1
        starts = np.concatenate([[0], change])
        counts = np.diff(np.concatenate([starts, [len(fids)]]))
        if len(starts) > 2:   # 先頭・末尾の欠けたフレームを捨てる
            starts, counts = starts[1:-1], counts[1:-1]

        self.pkt_off = offs
        self.frame_start = starts
        self.frame_count = counts
        self.frame_ts = tss[starts]
        print(f"Indexing done: {len(starts)} frames, {len(offs)} packets "
              f"({time.time()-t0:.1f}s)")
        try:
            np.savez(cache, key=key, pkt_off=offs, frame_start=starts,
                     frame_count=counts, frame_ts=self.frame_ts)
            print("Index cache saved:", cache)
        except OSError as e:
            print("Failed to save index cache (will re-index next time):", e)

    @property
    def n_frames(self):
        return len(self.frame_start)

    # --- 1フレーム分の距離画像をデコード (LRUキャッシュ付き) ---
    def _decode_raw(self, i):
        with self._raw_lock:
            hit = self._raw.get(i)
            if hit is not None:
                self._raw_order.remove(i)
                self._raw_order.append(i)
                return hit

        s = int(self.frame_start[i])
        c = int(self.frame_count[i])
        H, W, CPP = self.H, self.W, self.cols_per_packet

        rng = np.zeros((H, W), np.uint32)
        refl = np.zeros((H, W), np.uint8)
        sig = np.zeros((H, W), np.uint16)
        nir = np.zeros((H, W), np.uint16)

        for k in range(s, s + c):
            off = int(self.pkt_off[k]) + PKT_HDR
            buf = np.frombuffer(self.mm, np.uint8, CPP * self.col_size, off)
            cols = buf.reshape(CPP, self.col_size)
            mids = cols[:, 8:10].copy().view("<u2")[:, 0]
            ok = mids < W
            if not ok.all():
                cols, mids = cols[ok], mids[ok]
            px = cols[:, COL_HDR:].reshape(-1, H, PX_SIZE)
            w0 = np.ascontiguousarray(px[:, :, 0:4]).view("<u4")[..., 0]
            rng[:, mids] = (w0 & 0x7FFFF).T
            refl[:, mids] = px[:, :, 4].T
            sig[:, mids] = np.ascontiguousarray(px[:, :, 6:8]).view("<u2")[..., 0].T
            nir[:, mids] = np.ascontiguousarray(px[:, :, 8:10]).view("<u2")[..., 0].T

        # 物理チャンネル順 → 仰角順。LUTも同じ並びで作ってある。
        p = self.row_perm
        out = (rng[p], refl[p], sig[p], nir[p])
        with self._raw_lock:
            self._raw[i] = out
            if i in self._raw_order:
                self._raw_order.remove(i)
            self._raw_order.append(i)
            while len(self._raw_order) > RAW_CACHE:
                self._raw.pop(self._raw_order.pop(0), None)
        return out

    # --- 時間方向の中央値でレンジノイズを落とす (静止センサー前提) ---
    def _median_range(self, i, tm):
        k = tm // 2
        idx = [j for j in range(i - k, i + k + 1) if 0 <= j < self.n_frames]
        st = np.stack([self._decode_raw(j)[0] for j in idx]).astype(np.float32)
        st[st == 0] = np.inf          # 無効点は末尾へ送り中央値の対象外にする
        st.sort(axis=0)
        valid = np.isfinite(st).sum(0)
        mid = (np.maximum(valid - 1, 0) // 2).astype(np.intp)
        med = np.take_along_axis(st, mid[None], 0)[0]
        med[valid == 0] = 0.0
        return med.astype(np.uint32)

    # --- 1フレームをデコードして点群バイナリを返す ---
    def decode(self, i, tm=1, seed_cm=0, grow_m=0.0):
        rng, refl, sig, nir = self._decode_raw(i)
        if tm > 1:
            rng = self._median_range(i, tm)

        xyz = self.lut_dir * rng[None].astype(np.float32) + self.lut_off  # (3,H,W)
        xyz[:, rng == 0] = 0.0   # 無効点は原点に置き、クライアント側で捨てる

        normal, rad_maj, rad_min, phi = self._surfels(xyz, rng)

        pts = np.ascontiguousarray(np.moveaxis(xyz, 0, -1).reshape(-1, 3),
                                   np.float32)
        quads = self._quads(rng)
        # 前景の証拠は2系統を合成する:
        #  (a) 時間差分: 背景(時間中央値)より手前に何cm出ているか。動くものに画素精度で効く。
        #      ただし座奏の演奏者や楽器のように動かないものは背景に吸収されて拾えない。
        #  (b) 会場距離: 3DGSスキャン表面から何cm離れているか。静止していても
        #      スキャンに無いものを拾う。天井などのスキャン欠損を誤検出しないよう
        #      ステージ域(STAGE_R, STAGE_Z)に限る。
        bg = self.background()
        rf = rng.astype(np.float32)
        fg_t = np.where((rng > 0) & (bg > 0), (bg - rf) / 10.0,
                        np.where(rng > 0, 255.0, 0.0))
        fg_t = np.clip(fg_t, 0.0, 255.0)
        venue_raw = venue_distance_cm(pts.astype(np.float64))
        fg_v = venue_raw.astype(np.float32).reshape(rng.shape)
        rxy = np.hypot(xyz[0], xyz[1])
        in_stage = (rxy < STAGE_R) & (xyz[2] < STAGE_Z)
        fg_v = np.where(in_stage, fg_v, 0.0)
        dyn = np.where(in_stage, np.maximum(fg_t, fg_v), 0.0).astype(np.uint8).reshape(-1)
        if seed_cm > 0 and grow_m > 0.0:
            grown = self._grow_dynamic(xyz, rng, dyn, seed_cm, grow_m,
                                       venue_cm=venue_raw).reshape(rng.shape)
            # 連結成分ごとに画素数を数え、小さすぎるものはレンジノイズとして捨てる
            lab, n_lab = ndimage.label(grown)
            if n_lab:
                sizes = np.bincount(lab.reshape(-1))
                keep = sizes >= MIN_CLUSTER
                keep[0] = False
                grown = keep[lab]
            # 成長は面の連続性だけを見るので、舞台上の種から壁や天井へ這い上がる。
            # 対象は舞台上の演奏者と楽器なので、舞台域の外は最終的に落とす。
            grown &= in_stage
            dyn = np.where(grown, 255, 0).astype(np.uint8).reshape(-1)
        t_rel = float(self.frame_ts[i] - self.frame_ts[0])
        header = struct.pack("<4sIff", b"OPC5", pts.shape[0], t_rel, self.fps)
        return b"".join([header, pts.tobytes(),
                         refl.reshape(-1).tobytes(),
                         sig.reshape(-1).astype("<u2").tobytes(),
                         nir.reshape(-1).astype("<u2").tobytes(),
                         normal.tobytes(), rad_maj.tobytes(),
                         rad_min.tobytes(), phi.tobytes(), dyn.tobytes(),
                         struct.pack("<I", quads.shape[0]), quads.tobytes()])

    # --- 背景モデル: 収録全体に散らした数十枚の画素ごと中央値 ---
    # 会場スキャンとの距離で動体を判定すると位置合わせ誤差に支配される
    # (静止フレームでも12.3%が30cm以上ずれ、S/N比が0.97しか出なかった)。
    # 同じセンサーの中央値なら画素単位で厳密に一致する。
    def background(self, k=31):
        if self._bg is None:
            idx = np.linspace(0, self.n_frames - 1, k).astype(int)
            st = np.stack([self._decode_raw(int(i))[0]
                           for i in idx]).astype(np.float32)
            st[st == 0] = np.nan
            with np.errstate(invalid="ignore"):
                med = np.nanmedian(st, axis=0)
            med[np.isnan(med)] = 0.0
            self._bg = med
        return self._bg

    # --- 動体の種から面を辿って連結成分を広げる ---
    # 会場からの距離だけで切ると、立っている人の足は必ず床の近くにあるので
    # 消える。確実な動体(種)からレンジ画像の隣接を辿り、測地距離が予算内なら
    # 会場からの距離に関わらず取り込む。床は斜入射で1ステップが0.8m級になり
    # 予算をすぐ超えるため、そのまま漏れ出しの歯止めになる。
    def _grow_dynamic(self, xyz, rng, dyn_cm, seed_cm, budget, iters=16, venue_cm=None):
        H, W = self.H, self.W
        V = rng > 0
        # スキャン済み表面(ピアノ・床・壁)の上へは広げない。演奏者がピアノに接して
        # いると成長がピアノ表面へ流れ込み、3DGSにある部分まで前景になる。
        if venue_cm is not None:
            V = V & (venue_cm.reshape(H, W) >= GROW_VMIN)
        rf = rng.astype(np.float32)
        INF = np.float32(1e9)

        # 方位方向の辺 (w -> w+1, 360°で巡回)
        rn = np.roll(rf, -1, axis=1)
        lu = np.linalg.norm(np.roll(xyz, -1, axis=2) - xyz, axis=0).astype(np.float32)
        oku = V & np.roll(V, -1, axis=1)
        oku &= np.abs(rn - rf) <= (MESH_TAN * self.step_u * 0.5 + REL_NOISE) \
            * np.maximum(np.minimum(rf, rn), 1.0)

        # ビーム方向の辺 (h -> h+1)
        lv = np.zeros((H, W), np.float32)
        okv = np.zeros((H, W), bool)
        lv[:-1] = np.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=0)
        thr_v = (MESH_TAN * self.step_v[:-1] * 0.5 + REL_NOISE)[:, None]
        okv[:-1] = (V[:-1] & V[1:]) & (np.abs(rf[1:] - rf[:-1])
                                       <= thr_v * np.maximum(
                                           np.minimum(rf[:-1], rf[1:]), 1.0))

        cost = np.where(V & (dyn_cm.reshape(H, W) >= seed_cm), 0.0, INF)
        cost = cost.astype(np.float32)
        okv_up = np.zeros_like(okv)
        okv_up[1:] = okv[:-1]
        for _ in range(iters):
            prev = cost
            c = np.roll(cost, -1, axis=1) + lu          # w+1 から w へ
            cost = np.where(oku & (c < cost), c, cost)
            c = np.roll(cost + lu, 1, axis=1)           # w から w+1 へ
            cost = np.where(np.roll(oku, 1, axis=1) & (c < cost), c, cost)
            c = np.full_like(cost, INF)
            c[:-1] = cost[1:] + lv[:-1]                 # h+1 から h へ
            cost = np.where(okv & (c < cost), c, cost)
            c = np.full_like(cost, INF)
            c[1:] = cost[:-1] + lv[:-1]                 # h から h+1 へ
            cost = np.where(okv_up & (c < cost), c, cost)
            if np.array_equal(cost, prev):
                break
        return ((cost <= budget) & V).reshape(-1)

    # --- レンジ画像の格子から、面として繋いでよい四角形パッチを列挙する ---
    # グリッド接続性を使うとビーム間の隙間がテクスチャ補間で埋まり、独立した
    # スプラットで生じる穴と棘の両方が同時に消える。
    def _quads(self, rng):
        H, W = self.H, self.W
        rf = rng.astype(np.float32)
        a = rf[:-1, :-1]                 # (h,w)      左上
        b = rf[:-1, 1:]                  # (h,w+1)    右上
        c = rf[1:, :-1]                  # (h+1,w)    左下
        d = rf[1:, 1:]                   # (h+1,w+1)  右下
        ok = (a > 0) & (b > 0) & (c > 0) & (d > 0)
        mx = np.maximum(np.maximum(a, b), np.maximum(c, d))
        mn = np.minimum(np.minimum(a, b), np.minimum(c, d))
        # 面の連続判定は法線の有効判定より緩くする。ドーム型センサーが天井や床を
        # 見ると仰角10°付近で隣接ビームのレンジ差が2.4mに達するが、これは滑らかな
        # 変化であって遮蔽境界ではない。ここで切ると面が短冊に割れる。
        thr = (MESH_TAN * (self.step_u + self.step_v[:-1]) * 0.5 + REL_NOISE)[:, None]
        ok &= (mx - mn) <= thr * np.maximum(mn, 1.0)
        # 方位の継ぎ目(w=W-1→0)はUVが逆走するので列ごと落とす
        hh, ww = np.nonzero(ok)
        return (hh.astype(np.uint32) * W + ww.astype(np.uint32))

    # --- 法線のoctahedronエンコード/デコード (GLSL octDecode と同一) ---
    @staticmethod
    def _oct_encode(n):
        ax, ay, az = np.abs(n)
        s = np.maximum(ax + ay + az, 1e-9)
        u, v = n[0] / s, n[1] / s
        neg = n[2] < 0
        su = np.where(u >= 0, 1.0, -1.0)
        sv = np.where(v >= 0, 1.0, -1.0)
        uo = np.where(neg, (1.0 - np.abs(v)) * su, u)
        vo = np.where(neg, (1.0 - np.abs(u)) * sv, v)
        o = np.clip(np.stack([uo, vo], -1) * 127.0, -127, 127)
        return np.round(o).astype(np.int8)

    @staticmethod
    def _oct_decode(oct8, shape):
        e = oct8.astype(np.float32) / 127.0
        x = e[..., 0].reshape(shape)
        y = e[..., 1].reshape(shape)
        z = 1.0 - np.abs(x) - np.abs(y)
        neg = z < 0
        sx = np.where(x >= 0, 1.0, -1.0)
        sy = np.where(y >= 0, 1.0, -1.0)
        nx = np.where(neg, (1.0 - np.abs(y)) * sx, x)
        ny = np.where(neg, (1.0 - np.abs(x)) * sy, y)
        n = np.stack([nx, ny, z]).astype(np.float32)
        return n / np.maximum(np.linalg.norm(n, axis=0), 1e-9)

    # --- 法線から一意な接線基底 (Duff et al. 2017, GLSL側と同一式) ---
    @staticmethod
    def _onb(n):
        s = np.where(n[2] >= 0.0, 1.0, -1.0).astype(np.float32)
        a = -1.0 / (s + n[2])
        b = n[0] * n[1] * a
        t1 = np.stack([1.0 + s * n[0] * n[0] * a, s * b, -s * n[0]])
        t2 = np.stack([b, s + n[1] * n[1] * a, -n[1]])
        return t1, t2

    @staticmethod
    def _log8(r):
        k = np.log(np.maximum(r, LOG_R_MIN) / LOG_R_MIN) / np.log(LOG_R_BASE)
        return np.clip(np.round(k), 0, 255).astype(np.uint8).reshape(-1)

    # --- 距離画像の隣接点からサーフェル(法線+半径)を推定 ---
    def _surfels(self, P, rng):
        H, W = self.H, self.W
        V = rng > 0

        du = np.roll(P, -1, axis=2) - np.roll(P, 1, axis=2)   # 方位方向(360°で巡回)
        dv = np.empty_like(P)                                  # ビーム方向
        dv[:, 1:-1] = P[:, 2:] - P[:, :-2]
        dv[:, 0] = P[:, 1] - P[:, 0]
        dv[:, -1] = P[:, -1] - P[:, -2]

        nrm = np.cross(du, dv, axisa=0, axisb=0, axisc=0)
        ln = np.linalg.norm(nrm, axis=0)
        flip = (nrm * P).sum(0) > 0        # センサ側を向くように反転
        nrm[:, flip] *= -1.0

        good = V & (ln > 1e-9)
        good &= np.roll(V, -1, axis=1) & np.roll(V, 1, axis=1)
        vv = np.zeros_like(V)
        vv[1:-1] = V[2:] & V[:-2]
        vv[0] = V[1]
        vv[-1] = V[-2]
        good &= vv
        rr = rng.astype(np.float32) / 1000.0
        # 深度エッジ判定はレンジの相対差で行う。3次元距離を見ると斜入射の面ほど
        # 隣接点が離れるため、このドーム型センサーでは面までエッジ扱いになり
        # 有効法線が全体の0.9%まで落ちていた。閾値は角度ステップ×tan(最大入射角)。
        rf = rng.astype(np.float32)
        den = np.maximum(rf, 1.0)
        dru = np.abs(np.roll(rf, -1, axis=1) - np.roll(rf, 1, axis=1)) / den
        drv = np.empty_like(rf)
        drv[1:-1] = np.abs(rf[2:] - rf[:-2])
        drv[0] = np.abs(rf[1] - rf[0])
        drv[-1] = np.abs(rf[-1] - rf[-2])
        drv /= den
        good &= (dru < TAN_MAX * self.step_u + REL_NOISE)
        good &= (drv < (TAN_MAX * self.step_v + REL_NOISE)[:, None])

        # 法線が取れない点はセンサ方向を向いたビルボード扱い
        pl = np.maximum(np.linalg.norm(P, axis=0), 1e-9)
        fb = -P / pl
        n_unit = np.where(good, nrm / np.maximum(ln, 1e-9), fb).astype(np.float32)

        # 法線は先に量子化する。シェーダは量子化後の値から接線基底を作るので、
        # 角度phiも同じ基底で測らないと楕円の向きがずれる。
        oct8 = self._oct_encode(n_unit)
        n_q = self._oct_decode(oct8, (H, W))
        t1, t2 = self._onb(n_q)

        # 方位/ビーム方向の半間隔ベクトルを接平面へ射影し、2次モーメントの
        # 主軸を楕円の軸に取る。標本間隔は方向で最大5.9倍違うため、等方円だと
        # 縦に隙間が残り横に過剰ににじむ。
        au, av = (du * t1).sum(0) * 0.5, (du * t2).sum(0) * 0.5
        bu, bv = (dv * t1).sum(0) * 0.5, (dv * t2).sum(0) * 0.5
        c00 = au * au + bu * bu
        c11 = av * av + bv * bv
        c01 = au * av + bu * bv
        disc = np.sqrt(np.maximum(0.25 * (c00 - c11) ** 2 + c01 * c01, 0.0))
        half = 0.5 * (c00 + c11)
        K = 0.7                              # 半間隔→σ (従来の 0.35*|d| と同スケール)
        r1 = K * np.sqrt(np.maximum(half + disc, 0.0))
        r2 = K * np.sqrt(np.maximum(half - disc, 0.0))
        phi = 0.5 * np.arctan2(2.0 * c01, c00 - c11)

        # 標本形状に忠実な扁平率は斜入射面で15倍を超え、描画では棘状に破綻する。
        # 穏やかな異方性に留め、覆いきれないぶんは隙間として残す。
        cap = rr * SIG_CAP_K + SIG_CAP_B
        r1 = np.minimum(r1, cap)
        r2 = np.minimum(np.maximum(r2, r1 / MAX_ASPECT), cap)
        r_fb = rr * (2.0 * np.pi / W) * 1.5
        r1 = np.where(good, r1, r_fb)
        r2 = np.where(good, r2, r_fb)
        phi = np.where(good, phi, 0.0)
        np.clip(r1, 0.004, 0.4, out=r1)
        np.clip(r2, 0.004, 0.4, out=r2)

        # 楕円はπ周期なので phi は [0,π) に畳んで u8 に載せる
        ph8 = np.clip(np.round(np.mod(phi, np.pi) / np.pi * 255.0), 0, 255)
        return (oct8.reshape(H * W, 2), self._log8(r1), self._log8(r2),
                ph8.astype(np.uint8).reshape(-1))


VENUE_DIST = None        # (dist grid, origin, cell) — 会場表面からの距離(ボクセル単位)


def venue_distance_cm(pts):
    """各点の会場表面からの距離をcmで返す。会場が無ければ全点255(=動体扱い)。"""
    if VENUE_DIST is None:
        return np.full(len(pts), 255, np.uint8)
    dist, origin, cell = VENUE_DIST
    dims = np.array(dist.shape)
    k = ((pts - origin) / cell).astype(np.int64)
    ok = np.all((k >= 0) & (k < dims), axis=1)
    out = np.full(len(pts), 255, np.int32)
    kk = k[ok]
    v = dist[kk[:, 0], kk[:, 1], kk[:, 2]].astype(np.int32)
    out[ok] = np.where(v == 255, 255,
                       np.minimum(v * int(round(cell * 100)), 254))
    return out.astype(np.uint8)


VENUE_POINTS = None      # 会場を点群で描くモード用のバイナリ (ガウシアン中心 + 色)


def load_venue():
    """bake_venue.py が焼いた会場メッシュ (LiDAR座標系, 頂点カラー付き)。"""
    global VENUE_DIST, VENUE_POINTS
    p = os.path.join(HERE, "venue.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    if "dist" in z:
        VENUE_DIST = (z["dist"], z["dist_origin"].astype(np.float64),
                      float(z["dist_cell"]))
        print(f"Venue distance grid: {z['dist'].shape} @"
              f"{float(z['dist_cell'])*100:.0f}cm")
    if "gs_pos" in z:
        gp = np.ascontiguousarray(z["gs_pos"].astype(np.float32)); gc = np.ascontiguousarray(z["gs_rgb"])
        VENUE_POINTS = b"".join([struct.pack("<4sI", b"VPT1", len(gp)), gp.tobytes(), gc.tobytes()])
        print(f"Venue points: {len(gp)}")
    v = z["verts"].astype(np.float32)
    c = np.clip(z["colors"] * 255.0, 0, 255).astype(np.uint8)
    f = z["faces"].astype(np.uint32)
    print(f"Venue mesh: {len(v)} verts / {len(f)} tris "
          f"(yaw {float(z['yaw_deg']):.2f} deg)")
    return b"".join([struct.pack("<4sII", b"VNU1", len(v), len(f)),
                     v.tobytes(), np.ascontiguousarray(c).tobytes(),
                     f.tobytes()])


VIDEOS = {}              # name -> path (会場を撮った実写。射影テクスチャの素材)
CALIB_DIR = os.path.join(HERE, "calib_data")   # ブラウザから受けた較正素材の置き場


def find_videos():
    d = os.path.dirname(DEFAULT_DIR)
    if not os.path.isdir(d):
        return
    for fn in sorted(os.listdir(d)):
        if fn.lower().endswith((".mp4", ".mov")):
            VIDEOS[fn] = os.path.join(d, fn)
    if VIDEOS:
        print("Videos:", ", ".join(VIDEOS))


def make_handler(src: OusterPcap):
    venue = load_venue()
    info = {
        "n_frames": src.n_frames,
        "fps": src.fps,
        "n_points": src.H * src.W,
        "rows": src.H,
        "cols": src.W,
        "duration_s": float(src.frame_ts[-1] - src.frame_ts[0]),
        "sensor": src.meta.get("sensor_info", {}).get("prod_line", "?"),
        "serial": src.meta.get("sensor_info", {}).get("prod_sn", "?"),
        "mode": src.meta.get("config_params", {}).get("lidar_mode", "?"),
        "pcap": os.path.basename(src.pcap_path),
        # 前景判定は venue.npz と背景モデルに依存する。起動ごとに変わる番号をURLに
        # 付けさせ、ブラウザが前回起動時のフレームをキャッシュから返さないようにする。
        "gen": int(time.time()),
    }
    info_json = json.dumps(info).encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):   # 静かに
            pass

        def _send(self, code, ctype, body, cache=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if cache:
                self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        # 動画は大きいのでRangeで部分送信する (シークに必須)
        def _send_video(self, name):
            path = VIDEOS.get(name)
            if not path or not os.path.exists(path):
                self._send(404, "text/plain", b"unknown video")
                return
            size = os.path.getsize(path)
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng and rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                if a:
                    start = int(a)
                if b:
                    end = min(int(b), size - 1)
                end = min(end, start + 4 * 1024 * 1024 - 1)
            length = max(0, end - start + 1)
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as fh:
                fh.seek(start)
                remain = length
                while remain > 0:
                    chunk = fh.read(min(262144, remain))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remain -= len(chunk)

        # 較正用: ブラウザから映像の生画素(グレースケール)や動き系列を受け取る
        def do_POST(self):
            try:
                u = urlparse(self.path)
                n = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(n)
                if u.path == "/calib/save":
                    d = json.loads(body.decode("utf-8"))
                    if not (isinstance(d.get("P"), list) and len(d["P"]) == 3):
                        raise ValueError("bad P")
                    with open(os.path.join(HERE, "calib.json"), "w") as fh:
                        json.dump(d, fh, indent=1)
                    self._send(200, "text/plain", b"calibration saved")
                elif u.path == "/calib/upload":
                    name = os.path.basename(parse_qs(u.query).get("name", ["blob"])[0])
                    os.makedirs(CALIB_DIR, exist_ok=True)
                    with open(os.path.join(CALIB_DIR, name), "wb") as fh:
                        fh.write(body)
                    self._send(200, "text/plain", f"saved {n} bytes".encode())
                else:
                    self._send(404, "text/plain", b"not found")
            except Exception as e:
                self._send(500, "text/plain", str(e).encode())

        def do_GET(self):
            try:
                if self.path in ("/", "/index.html"):
                    with open(os.path.join(HERE, "viewer.html"), "rb") as fh:
                        self._send(200, "text/html; charset=utf-8", fh.read())
                elif self.path == "/info":
                    self._send(200, "application/json", info_json)
                elif urlparse(self.path).path == "/calib":
                    p = os.path.join(HERE, "calib.json")
                    if os.path.exists(p):
                        with open(p, "rb") as fh:
                            self._send(200, "application/json", fh.read())
                    else:
                        self._send(404, "text/plain", b"no calibration yet")
                elif self.path == "/videos":
                    self._send(200, "application/json",
                               json.dumps(sorted(VIDEOS)).encode())
                elif self.path.startswith("/video/"):
                    self._send_video(unquote(os.path.basename(urlparse(self.path).path)))
                elif urlparse(self.path).path == "/venue_points":
                    if VENUE_POINTS is None:
                        self._send(404, "text/plain", b"run bake_venue.py first")
                    else:
                        self._send(200, "application/octet-stream", VENUE_POINTS, cache=True)
                elif urlparse(self.path).path == "/venue":
                    if venue is None:
                        self._send(404, "text/plain", b"run bake_venue.py first")
                    else:
                        self._send(200, "application/octet-stream", venue,
                                   cache=True)
                elif self.path.startswith("/frame/"):
                    u = urlparse(self.path)
                    i = int(u.path.split("/")[2])
                    q = parse_qs(u.query)
                    tm = int(q.get("tm", ["1"])[0])
                    tm = max(1, min(15, tm | 1))    # 中央値を取るため奇数に丸める
                    seed = max(0, min(200, int(q.get("seed", ["0"])[0])))
                    grow = max(0.0, min(3.0, float(q.get("grow", ["0"])[0])))
                    if 0 <= i < src.n_frames:
                        self._send(200, "application/octet-stream",
                                   src.decode(i, tm, seed, grow), cache=True)
                    else:
                        self._send(404, "text/plain", b"frame out of range")
                else:
                    self._send(404, "text/plain", b"not found")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self._send(500, "text/plain", str(e).encode())
                except Exception:
                    pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Ouster pcap 3D point cloud viewer")
    ap.add_argument("--pcap", default=DEFAULT_PCAP)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true",
                    help="do not open the browser automatically")
    args = ap.parse_args()

    for p, label in [(args.pcap, "pcap"), (args.meta, "metadata JSON")]:
        if not os.path.exists(p):
            raise SystemExit(f"{label} not found: {p}")

    find_videos()
    src = OusterPcap(args.pcap, args.meta)
    print(f"{src.meta['sensor_info']['prod_line']}  "
          f"{src.n_frames} frames  {src.fps:.1f} fps  "
          f"{src.frame_ts[-1]-src.frame_ts[0]:.1f} s")

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(src))
    url = f"http://127.0.0.1:{args.port}/"
    print("Viewer:", url, " (Ctrl+C to quit)")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()
