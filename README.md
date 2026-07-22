# LiDAR 点群ビューア (Ouster OS-DOME-128)

Ousterのpcap録画 + センサJSONメタデータを、ブラウザ上で動的3D点群として
再生・操作できるビューアです。依存は **Python 3 + numpy** のみ
(ビューア側はライブラリなしのWebGL2)。

## 起動

```sh
./run.sh
# または
python3 server.py
```

ブラウザが自動で開きます (http://127.0.0.1:8765/)。
デフォルトで `DATA/20260701_可視化本映像３次元データ（動画）/` のpcapを読みます。
別ファイルは `--pcap` / `--meta` で指定:

```sh
python3 server.py --pcap /path/to/rec.pcap --meta /path/to/meta.json
```

- 初回のみpcap全体を索引化します(11GBで数分)。結果は
  `<pcap>.viewidx.npz` にキャッシュされ、2回目以降は即起動します。
- フレームは再生位置に合わせてオンデマンドにデコード・先読みされるので、
  11GBでもメモリを圧迫しません。

## 操作

| 操作 | 内容 |
|---|---|
| ドラッグ | 回転 |
| 右ドラッグ / Shift+ドラッグ | 平行移動 |
| ホイール | ズーム |
| Space | 再生 / 一時停止 |
| ← → | コマ送り (Shiftで±20フレーム) |

パネルから 描画(スプラット/点)、カラー(反射率/信号強度/近赤外/高さ/距離)、
カラーマップ(Magma/Inferno/Viridis/Turbo/グレー)、点サイズ/スプラット径/不透明度、
自動レベル補正、距離リングを切り替えられます。

- **スプラット描画(デフォルト)**: 各点を距離画像の隣接関係から推定した
  法線付きサーフェル(楕円ガウシアン)として描画。Gaussian Splatting風の
  連続した面の見た目になります。深度ソート+αブレンドで正しく合成。
- 「PLY書き出し」: 表示中フレームを素のバイナリPLYで保存
  (TouchDesignerのPoint File In等で読めます)。
- 「3DGS PLY書き出し」: 表示中フレームを**3D Gaussian Splatting標準の
  PLY形式**(f_dc/opacity/scale/rot付き)で保存。SuperSplat
  (https://playcanvas.com/supersplat/editor)やgsplat系ビューアに
  そのまま読み込めます。色は現在のカラー設定が焼き込まれます。

## 実装メモ

- `server.py` — pcapをmmapし、UDPペイロード(eUDP `RNG19_RFL8_SIG16_NIR16`,
  32Bヘッダ + 16列×(12B + 128px×12B) + 32Bフッタ)をnumpyでデコード。
  XYZ変換は ouster-sdk の `make_xyz_lut` と同じ式
  (beam_intrinsics + lidar_to_sensor_transform) を事前計算LUT化。
  さらに距離画像の隣接点からサーフェル法線(oct圧縮int8×2)と
  半径(対数u8)を推定して配信。1フレーム(131,072点)約40ms。
- `viewer.html` — WebGL2。位置float32 + 反射率u8/信号u16/近赤外u16/法線/半径を
  頂点属性として保持し、カラーマップはシェーダ内多項式近似で計算。
  スプラットは接平面とレイの交点で楕円フットプリントを算出し、
  CPU基数ソート(奥→手前)+前乗算αブレンドで合成。
  フレームはLRUキャッシュ(約280フレーム)+先読み4並列。
