# LiDAR Point Cloud Viewer (Ouster OS-DOME-128)

A browser-based viewer that plays back an Ouster pcap recording (plus its
sensor metadata JSON) as a dynamic 3D point cloud. The only dependency is
**Python 3 + numpy** — the viewer itself is plain WebGL2 with no libraries.

## Getting started

```sh
./run.sh
# or
python3 server.py
```

Your browser opens automatically at http://127.0.0.1:8765/.
By default the server reads the pcap under
`DATA/20260701_可視化本映像３次元データ（動画）/`.
To use different files, pass `--pcap` / `--meta`:

```sh
python3 server.py --pcap /path/to/rec.pcap --meta /path/to/meta.json
```

- On first launch the whole pcap is indexed (a few minutes for 11 GB).
  The index is cached as `<pcap>.viewidx.npz`, so subsequent launches
  start instantly.
- Frames are decoded on demand and prefetched around the playhead, so
  even an 11 GB capture uses little memory.

## Controls

| Input | Action |
|---|---|
| Drag | Orbit |
| Right-drag / Shift+drag | Pan |
| Wheel | Zoom |
| Space | Play / pause |
| ← → | Step one frame (±20 with Shift) |

The panel lets you switch rendering (splat / points), color source
(reflectivity / signal / near-IR / height / range), colormap
(Magma / Inferno / Viridis / Turbo / gray), point size / splat radius /
opacity, auto-leveling, and range rings.

- **Splat rendering (default)**: each point is drawn as a surfel — an
  oriented elliptical Gaussian whose normal is estimated from the range
  image neighborhood — giving a continuous, Gaussian-Splatting-like
  surface appearance. Splats are depth-sorted and alpha-blended.
- **"PLY書き出し" (Export PLY)**: saves the current frame as a plain
  binary PLY (readable by TouchDesigner's Point File In, etc.).
- **"3DGS PLY書き出し" (Export 3DGS PLY)**: saves the current frame in the
  standard **3D Gaussian Splatting PLY format** (with f_dc / opacity /
  scale / rot), loadable directly in SuperSplat
  (https://playcanvas.com/supersplat/editor) and other gsplat viewers.
  The current color settings are baked in.

## Implementation notes

- `server.py` — mmaps the pcap and decodes the UDP payloads
  (eUDP `RNG19_RFL8_SIG16_NIR16`: 32 B header + 16 columns ×
  (12 B + 128 px × 12 B) + 32 B footer) with numpy.
  The XYZ conversion uses the same formula as ouster-sdk's
  `make_xyz_lut` (beam_intrinsics + lidar_to_sensor_transform),
  precomputed as a lookup table. It also estimates per-point surfel
  normals (oct-encoded int8×2) and radii (log-encoded u8) from the
  range-image neighborhood. One frame (131,072 points) takes ~40 ms.
- `viewer.html` — WebGL2. Positions are float32; reflectivity u8,
  signal u16, near-IR u16, normal, and radius are vertex attributes.
  Colormaps are polynomial approximations evaluated in the shader.
  Splat footprints are computed by intersecting the view ray with each
  surfel's tangent plane, composited with a CPU radix sort
  (back-to-front) and premultiplied alpha blending.
  Frames go through an LRU cache (~280 frames) with 4 parallel prefetches.
