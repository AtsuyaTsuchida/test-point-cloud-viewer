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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

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

        self._build_lut()
        self._index()

        dt = np.diff(self.frame_ts)
        dt = dt[(dt > 1e-4) & (dt < 1.0)]
        self.fps = float(1.0 / np.median(dt)) if len(dt) else 20.0

    # --- XYZ変換ルックアップテーブル (ouster-sdk make_xyz_lut と同じ式) ---
    def _build_lut(self):
        bi = self.meta["beam_intrinsics"]
        alt = np.deg2rad(np.asarray(bi["beam_altitude_angles"], np.float64))
        az = -np.deg2rad(np.asarray(bi["beam_azimuth_angles"], np.float64))
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

    # --- 1フレームをデコードして点群バイナリを返す ---
    def decode(self, i):
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

        xyz = self.lut_dir * rng[None].astype(np.float32) + self.lut_off  # (3,H,W)
        xyz[:, rng == 0] = 0.0   # 無効点は原点に置き、クライアント側で捨てる

        normal, radius = self._surfels(xyz, rng)

        pts = np.ascontiguousarray(np.moveaxis(xyz, 0, -1).reshape(-1, 3),
                                   np.float32)
        t_rel = float(self.frame_ts[i] - self.frame_ts[0])
        header = struct.pack("<4sIff", b"OPC2", pts.shape[0], t_rel, self.fps)
        return b"".join([header, pts.tobytes(),
                         refl.reshape(-1).tobytes(),
                         sig.reshape(-1).astype("<u2").tobytes(),
                         nir.reshape(-1).astype("<u2").tobytes(),
                         normal.tobytes(), radius.tobytes()])

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
        lu_n = np.linalg.norm(du, axis=0)
        lv_n = np.linalg.norm(dv, axis=0)
        rr = rng.astype(np.float32) / 1000.0
        # 期待される隣接間隔(≈0.012×距離)の数倍を超えたら深度エッジとみなす
        good &= (lu_n < rr * 0.06) & (lv_n < rr * 0.08)

        # 法線が取れない点はセンサ方向を向いたビルボード扱い
        pl = np.maximum(np.linalg.norm(P, axis=0), 1e-9)
        fb = -P / pl
        n_unit = np.where(good, nrm / np.maximum(ln, 1e-9), fb).astype(np.float32)

        r = 0.35 * np.maximum(lu_n, lv_n)
        r = np.minimum(r, rr * 0.02 + 0.01)     # 距離に応じた上限で巨大化を防ぐ
        r_fb = rr * (2.0 * np.pi / W) * 1.5
        r = np.where(good, r, r_fb)
        np.clip(r, 0.004, 0.3, out=r)

        # 法線: octahedronエンコードでint8×2 / 半径: 対数スケールでu8
        ax, ay, az = np.abs(n_unit)
        s = np.maximum(ax + ay + az, 1e-9)
        u = n_unit[0] / s
        v = n_unit[1] / s
        neg = n_unit[2] < 0
        su = np.where(u >= 0, 1.0, -1.0)
        sv = np.where(v >= 0, 1.0, -1.0)
        uo = np.where(neg, (1.0 - np.abs(v)) * su, u)
        vo = np.where(neg, (1.0 - np.abs(u)) * sv, v)
        oct8 = np.clip(np.stack([uo, vo], -1) * 127.0, -127, 127)
        oct8 = np.round(oct8).astype(np.int8).reshape(H * W, 2)

        k = np.log(r / 0.004) / np.log(1.0182)
        rad8 = np.clip(np.round(k), 0, 255).astype(np.uint8).reshape(-1)
        return oct8, rad8


def make_handler(src: OusterPcap):
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

        def do_GET(self):
            try:
                if self.path in ("/", "/index.html"):
                    with open(os.path.join(HERE, "viewer.html"), "rb") as fh:
                        self._send(200, "text/html; charset=utf-8", fh.read())
                elif self.path == "/info":
                    self._send(200, "application/json", info_json)
                elif self.path.startswith("/frame/"):
                    i = int(self.path.split("/")[2])
                    if 0 <= i < src.n_frames:
                        self._send(200, "application/octet-stream",
                                   src.decode(i), cache=True)
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
