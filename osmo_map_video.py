#!/usr/bin/env python3
"""
osmo_map_video.py

Created on 2026/10/04 by NIM with Local LLM

OSMO Action 4 / 6 の動画に埋め込まれた GPS から、関東の地図上に軌跡を描いた
2048x2048 の動画を生成する。

- 出力は入力と同じフレーム数・同じフレームレート（＝同じ長さ）
- 出力には入力と同じ開始タイムコードを埋め込む
    * 入力に TC があればそれをコピー
    * 無ければ GPS 時刻（現地時刻）から開始 TC を算出し、
      入力側にも同じ TC を付けたコピー（*_tc.mov）を作る（--no-tag-input で無効）
- OSMO の位置は常に画面中心から下に 1/4（x=1024, y=1536）
- アイコン色・軌跡色・縮尺（m/px）はパラメータで指定

必要なもの:
  ffmpeg / ffprobe（PATH 上）
  exiftool（Osmo Action 6 対応の新しい版）
  pip install numpy pillow pmtiles mapbox-vector-tile
  pip install opencv-python-headless   # --lite を使う場合のみ

######################
地図データ作成コマンド（Protomaps / OSM, ODbL）の準備（一度だけ）:
  pmtiles extract https://build.protomaps.com/YYYYMMDD.pmtiles kanto.pmtiles \
      --bbox=138.4,34.8,140.95,37.2 --maxzoom=15
  YYYYMMDD指定日付なので、現在日付で良い。
######################

例:
  python osmo_map_video.py DJI_0001.MP4 --pmtiles kanto.pmtiles --mpp 2 \
      --icon-color "#ff3b30" --heading-up
  --mpp 2 は地図の縮尺で、1px が 2m に相当する。小さいほど拡大される。
  歩きなら、--mpp 0.2〜0.05くらいにするとよい。
  --lite を付けると高速化版で処理する（地図の回転・移動を OpenCV で行い、
  libx264 の preset 既定を veryfast にする）。見た目はほぼ同じ。
"""
import argparse
import gzip
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
import mapbox_vector_tile
from pmtiles.reader import Reader, MmapSource

try:
    import cv2                                    # --lite のときだけ使う
except ImportError:
    cv2 = None

# ---------------------------------------------------------------- 定数
OUT = 2048
ICON_X, ICON_Y = OUT // 2, OUT * 3 // 4          # 中心から下に 1/4
KANTO_BBOX = (138.4, 34.8, 140.95, 37.2)         # lon_min, lat_min, lon_max, lat_max
M_PER_PX_Z0 = 156543.03392                       # 赤道・zoom0・256px タイルの m/px

try:
    BILINEAR = Image.Resampling.BILINEAR
    LANCZOS = Image.Resampling.LANCZOS
    AFFINE = Image.Transform.AFFINE
except AttributeError:  # 古い Pillow
    BILINEAR, LANCZOS, AFFINE = Image.BILINEAR, Image.LANCZOS, Image.AFFINE

# ---------------------------------------------------------------- 地図スタイル
WATER = (170, 211, 223)
EARTH = (242, 239, 233)
PARK = (205, 235, 176)
WOOD = (173, 209, 158)
FARM = (238, 240, 213)
BUILDING = (217, 208, 201)
CASING = (185, 175, 165)

WOOD_KINDS = {"forest", "wood"}
GREEN_KINDS = {"park", "grass", "grassland", "garden", "golf_course", "nature_reserve",
               "meadow", "recreation_ground", "cemetery", "scrub", "wetland",
               "national_park", "protected_area", "village_green", "playground"}
FARM_KINDS = {"farmland", "orchard", "vineyard", "allotments", "farmyard"}

# kind: (描画順, 色, 幅[m], 最小幅[px])
ROAD_STYLE = {
    "highway":    (6, (232, 146, 162), 18, 3),
    "major_road": (5, (252, 214, 164), 12, 2),
    "minor_road": (4, (255, 255, 255), 7, 1),
    "path":       (3, (200, 190, 180), 2, 1),
    "rail":       (3, (150, 150, 150), 3, 1),
}
ROAD_DEFAULT = (4, (255, 255, 255), 5, 1)


def poly_style(layer, kind, mpp):
    if layer == "earth":
        return (0, EARTH)
    if layer in ("landcover", "landuse"):
        if kind in WOOD_KINDS:
            return (1, WOOD)
        if kind in GREEN_KINDS:
            return (1, PARK)
        if kind in FARM_KINDS:
            return (1, FARM)
        return None
    if layer == "water":
        return (2, WATER)
    if layer == "buildings" and mpp <= 3.0:
        return (3, BUILDING)
    return None


def line_style(layer, kind, mpp):
    """(描画順, 色, 幅px, ケーシング有無)"""
    if layer == "roads":
        o, c, m, mn = ROAD_STYLE.get(kind, ROAD_DEFAULT)
        return (o, c, max(mn, m / mpp), True)
    if layer == "water":
        return (2, WATER, max(1.0, (15 if kind == "river" else 4) / mpp), False)
    return None


# ---------------------------------------------------------------- 外部コマンド
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"コマンド失敗: {' '.join(map(str, cmd))}\n{r.stderr}")
    return r.stdout


def probe_video(path):
    """(fps分子, fps分母, フレーム数, 既存TC or None, creation_time or None)"""
    j = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=r_frame_rate,nb_read_packets",
                        "-of", "json", str(path)]))
    st = j["streams"][0]
    num, den = map(int, st["r_frame_rate"].split("/"))
    nframes = int(st["nb_read_packets"])

    j2 = json.loads(run(["ffprobe", "-v", "error", "-show_entries",
                         "stream_tags=timecode:format_tags=timecode,creation_time",
                         "-of", "json", str(path)]))
    tc = None
    for s in j2.get("streams", []):
        tc = tc or s.get("tags", {}).get("timecode")
    ftags = j2.get("format", {}).get("tags", {})
    tc = tc or ftags.get("timecode")
    return num, den, nframes, tc, ftags.get("creation_time")


# ---------------------------------------------------------------- GPS
GPS_DT_RE = re.compile(r"(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)")


def parse_gps_dt(s):
    """ExifTool の GPSDateTime（UTC）→ UNIX 秒"""
    m = GPS_DT_RE.search(str(s))
    if not m:
        return None
    y, mo, d, h, mi = map(int, m.groups()[:5])
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp() + float(m.group(6))


def extract_gps(path, fps):
    """[(映像先頭からの秒, lat, lon, GPS UTC秒 or None), ...]"""
    out = run(["exiftool", "-ee", "-n", "-j", "-G3", "-api", "LargeFileSupport=1",
               "-GPSLatitude", "-GPSLongitude", "-GPSDateTime", "-SampleTime", str(path)])
    data = json.loads(out)[0]
    docs = {}
    for key, val in data.items():
        if ":" not in key:
            continue
        grp, tag = key.split(":", 1)
        m = re.fullmatch(r"Doc(\d+)", grp)
        if m:
            docs.setdefault(int(m.group(1)), {})[tag] = val
    rows = []
    for n in sorted(docs):
        d = docs[n]
        if "GPSLatitude" not in d or "GPSLongitude" not in d:
            continue
        t = d.get("SampleTime")
        # SampleTime が無い場合は「1フレーム1サンプル」とみなす
        t = float(t) if isinstance(t, (int, float)) else (n - 1) / fps
        rows.append((t, float(d["GPSLatitude"]), float(d["GPSLongitude"]),
                     parse_gps_dt(d.get("GPSDateTime", ""))))
    return rows


def clean_track(rows):
    lo0, la0, lo1, la1 = KANTO_BBOX
    good = [r for r in rows if la0 <= r[1] <= la1 and lo0 <= r[2] <= lo1]
    if len(good) < 2:
        sys.exit("関東範囲内の有効な GPS サンプルがありません"
                 "（リモコン接続と GPS 測位、exiftool のバージョンを確認してください）")
    if len(rows) - len(good):
        print(f"注意: 関東範囲外・未測位の GPS サンプル {len(rows) - len(good)} 件を除外")
    t = np.array([r[0] for r in good])
    lat = np.array([r[1] for r in good])
    lon = np.array([r[2] for r in good])
    # 受信機の更新レートより記録レートが高く階段状になるので、値が変わった点だけ残す
    keep = np.ones(len(t), bool)
    keep[1:] = (np.diff(lat) != 0) | (np.diff(lon) != 0)
    keep[-1] = True
    return t[keep], lat[keep], lon[keep], good


def to_world_px(lat, lon, zf):
    size = 256.0 * 2.0 ** zf
    x = (lon + 180.0) / 360.0 * size
    y = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2.0 * size
    return x, y


def compute_heading(fx, fy, fps, mpp):
    """各フレームの進行方位（北=0, 時計回り, rad）。停止中は直前の方位を保持。"""
    n = len(fx)
    k = max(1, round(fps * 0.5))
    idx = np.arange(n)
    i0, i1 = np.clip(idx - k, 0, n - 1), np.clip(idx + k, 0, n - 1)
    dx, dy = fx[i1] - fx[i0], fy[i1] - fy[i0]
    valid = np.hypot(dx, dy) * mpp > 1.0          # 1 秒で 1 m 以上動いているとき
    if not valid.any():
        return np.zeros(n)
    h = np.arctan2(dx, -dy)
    src = np.maximum.accumulate(np.where(valid, idx, -1))
    src[src < 0] = int(np.argmax(valid))
    h = h[src]
    w = max(1, round(fps))                        # 約 1 秒の円周移動平均
    ker = np.ones(w) / w
    pad = (w // 2, w - 1 - w // 2)
    c = np.convolve(np.pad(np.cos(h), pad, mode="edge"), ker, mode="valid")
    s = np.convolve(np.pad(np.sin(h), pad, mode="edge"), ker, mode="valid")
    return np.arctan2(s, c)


# ---------------------------------------------------------------- タイムコード
def frames_to_tc(n, nominal, drop):
    if drop:
        d = 2 if nominal == 30 else 4
        per10 = nominal * 600 - 9 * d
        per1 = nominal * 60 - d
        tens, rem = divmod(n, per10)
        n += 9 * d * tens + (d * ((rem - d) // per1) if rem > d else 0)
    ff, s = n % nominal, n // nominal
    sep = ";" if drop else ":"
    return f"{(s // 3600) % 24:02d}:{(s // 60) % 60:02d}:{s % 60:02d}{sep}{ff:02d}"


def clock_to_tc(sec_of_day, num, den):
    """時刻（0時からの秒）→ TC。29.97/59.94 はドロップフレームで実時刻に追従させる。"""
    nominal = round(num / den)
    drop = den == 1001 and nominal in (30, 60)
    n = round(sec_of_day * num / den) if drop else round(sec_of_day * nominal)
    return frames_to_tc(n, nominal, drop)


def start_clock_from_gps(good, tz_hours, gps_offset):
    """映像先頭フレームの現地時刻（0時からの秒）"""
    pairs = [(r[0] + gps_offset, r[3]) for r in good if r[3] is not None]
    if not pairs:
        return None
    # GPS 時刻が切り替わった瞬間のサンプルを使い、秒単位の階段による誤差を抑える
    cand = [g - t for (t, g), (_, g0) in zip(pairs[1:], pairs[:-1]) if g != g0]
    if not cand:
        cand = [pairs[0][1] - pairs[0][0]]
    return (float(np.median(cand)) + tz_hours * 3600.0) % 86400.0


# ---------------------------------------------------------------- 地図描画
class VectorMap:
    """PMTiles（Protomaps）のベクトルタイルを指定縮尺でラスタ化し、チャンク単位でキャッシュする"""

    def __init__(self, pmtiles_path, zf, mpp, track_xy, track_rgb,
                 chunk=1024, ss=2, cache_chunks=64):
        self._f = open(pmtiles_path, "rb")
        self.reader = Reader(MmapSource(self._f))
        hdr = self.reader.header()
        self.maxzoom = int(hdr.get("max_zoom", 15))
        self.zf, self.mpp = zf, mpp
        self.z = max(0, min(self.maxzoom, math.ceil(zf)))
        self.k = 256.0 * 2.0 ** (zf - self.z)      # 1 タイルあたりの出力 px
        self.C, self.ss = chunk, max(1, ss)
        self.track, self.track_rgb = track_xy, track_rgb
        self.tile = lru_cache(maxsize=512)(self._tile)
        self.chunk = lru_cache(maxsize=cache_chunks)(self._render_chunk)

    def _tile(self, x, y):
        n = 2 ** self.z
        if not 0 <= y < n:
            return {}
        data = self.reader.get(self.z, x % n, y)
        if not data:
            return {}
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        try:
            return mapbox_vector_tile.decode(data, default_options={"y_coord_down": True})
        except TypeError:  # mapbox-vector-tile 1.x
            return mapbox_vector_tile.decode(data, y_coord_down=True)

    def _render_chunk(self, cx, cy):
        C, ss, k = self.C, self.ss, self.k
        W = C * ss
        ox, oy = cx * C, cy * C
        polys, lines = {}, []

        for ty in range(math.floor(oy / k), math.floor((oy + C - 1) / k) + 1):
            for tx in range(math.floor(ox / k), math.floor((ox + C - 1) / k) + 1):
                for lname, layer in self.tile(tx, ty).items():
                    ext = layer.get("extent", 4096)

                    def tr(ring, tx=tx, ty=ty, ext=ext):
                        a = np.asarray(ring, dtype=float)
                        if a.ndim != 2 or len(a) < 2:
                            return None
                        px = ((tx + a[:, 0] / ext) * k - ox) * ss
                        py = ((ty + a[:, 1] / ext) * k - oy) * ss
                        return np.column_stack([px, py]).ravel().tolist()

                    for feat in layer.get("features", []):
                        g = feat.get("geometry") or {}
                        gt, co = g.get("type"), g.get("coordinates")
                        kind = (feat.get("properties") or {}).get("kind", "")
                        if gt in ("Polygon", "MultiPolygon"):
                            st = poly_style(lname, kind, self.mpp)
                            if st is None:
                                continue
                            for p in ([co] if gt == "Polygon" else co):
                                rings = [r for r in map(tr, p) if r and len(r) >= 6]
                                if rings:
                                    polys.setdefault(st, []).append(rings)
                        elif gt in ("LineString", "MultiLineString"):
                            st = line_style(lname, kind, self.mpp)
                            if st is None:
                                continue
                            for ln in ([co] if gt == "LineString" else co):
                                f = tr(ln)
                                if f and len(f) >= 4:
                                    lines.append((st, f))

        img = Image.new("RGB", (W, W), WATER)      # 背景は海、earth レイヤが陸
        for st in sorted(polys, key=lambda s: s[0]):
            mask = Image.new("L", (W, W), 0)
            md = ImageDraw.Draw(mask)
            for rings in polys[st]:
                md.polygon(rings[0], fill=255)
                for hole in rings[1:]:
                    md.polygon(hole, fill=0)
            img.paste(st[1], (0, 0, W, W), mask)

        d = ImageDraw.Draw(img)
        lines.sort(key=lambda x: x[0][0])
        for (o, c, w, casing), f in lines:
            if casing:
                d.line(f, fill=CASING, width=max(1, round((w + 2) * ss)), joint="curve")
        for (o, c, w, casing), f in lines:
            d.line(f, fill=c, width=max(1, round(w * ss)), joint="curve")

        # 全行程の軌跡（薄い色）を地図に焼き込む
        if self.track is not None and len(self.track) >= 2:
            px = (self.track[:, 0] - ox) * ss
            py = (self.track[:, 1] - oy) * ss
            d.line(np.column_stack([px, py]).ravel().tolist(),
                   fill=self.track_rgb, width=round(6 * ss), joint="curve")

        return img.resize((C, C), LANCZOS) if ss > 1 else img

    def region(self, x0, y0, w, h):
        C = self.C
        img = Image.new("RGB", (w, h), WATER)
        for cy in range(math.floor(y0 / C), math.floor((y0 + h - 1) / C) + 1):
            for cx in range(math.floor(x0 / C), math.floor((x0 + w - 1) / C) + 1):
                img.paste(self.chunk(cx, cy), (cx * C - x0, cy * C - y0))
        return img


# ---------------------------------------------------------------- フレーム描画
def load_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def nice_length(m):
    p = 10 ** math.floor(math.log10(m))
    for f in (5, 2, 1):
        if f * p <= m:
            return f * p
    return p


def warp_affine_cv2(src, data):
    """Image.transform(AFFINE, BILINEAR) と同じ写像を OpenCV（マルチスレッド）で行う"""
    a, b, c0, d, e, f0 = data
    # PIL は画素中心 (x+0.5, y+0.5) で写像するので、OpenCV の整数座標系に合わせて平行移動を補正
    M = np.array([[a, b, c0 + 0.5 * (a + b) - 0.5],
                  [d, e, f0 + 0.5 * (d + e) - 0.5]])
    out = cv2.warpAffine(np.asarray(src), M, (OUT, OUT),
                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return Image.fromarray(out)


def render_frame(vm, X, Y, theta, heading, trav_xy, icon_rgb, track_rgb, font, mpp,
                 lite=False):
    c, s = math.cos(theta), math.sin(theta)
    # 出力の四隅に対応する地図側の範囲
    corners = [(-ICON_X, -ICON_Y), (OUT - ICON_X, -ICON_Y),
               (-ICON_X, OUT - ICON_Y), (OUT - ICON_X, OUT - ICON_Y)]
    dxs = [c * u - s * v for u, v in corners]
    dys = [s * u + c * v for u, v in corners]
    x0, y0 = math.floor(X + min(dxs)) - 2, math.floor(Y + min(dys)) - 2
    x1, y1 = math.ceil(X + max(dxs)) + 2, math.ceil(Y + max(dys)) + 2
    src = vm.region(x0, y0, x1 - x0, y1 - y0)
    Xr, Yr = X - x0, Y - y0
    # 出力(u,v) → 地図(x,y)。回転と端数ピクセルの移動を 1 回の補間で行う
    data = (c, -s, Xr - c * ICON_X + s * ICON_Y,
            s, c, Yr - s * ICON_X - c * ICON_Y)
    if lite:
        frame = warp_affine_cv2(src, data)
    else:
        frame = src.transform((OUT, OUT), AFFINE, data, resample=BILINEAR)
    d = ImageDraw.Draw(frame)

    # 走行済みの軌跡
    if len(trav_xy) >= 2:
        dx, dy = trav_xy[:, 0] - X, trav_xy[:, 1] - Y
        u = c * dx + s * dy + ICON_X
        v = -s * dx + c * dy + ICON_Y
        d.line(np.column_stack([u, v]).ravel().tolist(), fill=track_rgb, width=9, joint="curve")

    # OSMO アイコン（進行方向の矢印＋円）
    a = heading - theta
    ux, uy = math.sin(a), -math.cos(a)
    px, py = -uy, ux
    tri = [(ICON_X + ux * 56, ICON_Y + uy * 56),
           (ICON_X + ux * 12 + px * 26, ICON_Y + uy * 12 + py * 26),
           (ICON_X + ux * 12 - px * 26, ICON_Y + uy * 12 - py * 26)]
    d.polygon(tri, fill=icon_rgb)
    d.line(tri + [tri[0]], fill=(255, 255, 255), width=4, joint="curve")
    r = 26
    d.ellipse([ICON_X - r, ICON_Y - r, ICON_X + r, ICON_Y + r],
              fill=icon_rgb, outline=(255, 255, 255), width=6)

    # 縮尺バー（左下）
    L = nice_length(400 * mpp)
    Lpx = L / mpp
    bx, by = 40, OUT - 50
    d.rectangle([bx - 4, by - 14, bx + Lpx + 4, by + 4], fill=(255, 255, 255))
    d.rectangle([bx, by - 10, bx + Lpx, by], fill=(40, 40, 40))
    label = f"{L / 1000:g} km" if L >= 1000 else f"{L:g} m"
    d.text((bx, by - 20), label, fill=(40, 40, 40), font=font,
           stroke_width=3, stroke_fill=(255, 255, 255), anchor="ld")

    # 出典表示（ODbL）
    d.text((OUT - 30, OUT - 24), "© OpenStreetMap contributors", fill=(60, 60, 60),
           font=font, stroke_width=3, stroke_fill=(255, 255, 255), anchor="rd")
    return frame


# ---------------------------------------------------------------- メイン
def main():
    ap = argparse.ArgumentParser(description="OSMO Action 4/6 の GPS から TC 同期可能な地図動画を作る（関東）")
    ap.add_argument("input", help="OSMO Action 4/6 の動画ファイル")
    ap.add_argument("-o", "--output", help="出力ファイル（既定: <入力名>_map.mov）")
    ap.add_argument("--pmtiles", required=True, help="関東を切り出した .pmtiles")
    ap.add_argument("--mpp", type=float, default=2.0, help="地図の縮尺 [m/px]（既定 2.0）")
    ap.add_argument("--icon-color", default="#ff3b30", help="OSMO アイコンの色（既定 #ff3b30）")
    ap.add_argument("--track-color", default="#1e66ff", help="軌跡の色（既定 #1e66ff）")
    ap.add_argument("--heading-up", action="store_true", help="進行方向を上にする（既定は北が上）")
    ap.add_argument("--tz", type=float, default=9.0, help="TC に使う時差 [h]（既定 +9 = JST）")
    ap.add_argument("--gps-offset", type=float, default=0.0,
                    help="GPS を映像に対して遅らせる秒数（同期の微調整用）")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", help="libx264 の preset（既定 medium、--lite 時は veryfast）")
    ap.add_argument("--supersample", type=int, default=2, help="地図描画のアンチエイリアス倍率")
    ap.add_argument("--no-tag-input", action="store_true",
                    help="入力に TC が無い場合でも、TC 付きコピーを作らない")
    ap.add_argument("--lite", action="store_true",
                    help="高速化版で処理する（OpenCV で地図を変換し、preset 既定を veryfast にする）")
    args = ap.parse_args()
    if args.lite and cv2 is None:
        sys.exit("--lite には OpenCV が必要です（pip install opencv-python-headless）")
    preset = args.preset or ("veryfast" if args.lite else "medium")

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_name(inp.stem + "_map.mov")
    icon_rgb = ImageColor.getrgb(args.icon_color)[:3]
    track_rgb = ImageColor.getrgb(args.track_color)[:3]
    track_light = tuple(int(v + (255 - v) * 0.55) for v in track_rgb)

    # --- 入力の解析
    num, den, nframes, tc_in, ctime = probe_video(inp)
    fps = num / den
    print(f"入力: {nframes} フレーム, {fps:.2f} ({num}/{den}) fps, 総時間({nframes / fps:.2f} 秒), 既存TC: {tc_in}")

    rows = extract_gps(inp, fps)
    t, lat, lon, good = clean_track(rows)

    lat0 = float(np.mean(lat))
    zf = math.log2(M_PER_PX_Z0 * math.cos(math.radians(lat0)) / args.mpp)
    wx, wy = to_world_px(lat, lon, zf)

    ft = np.arange(nframes) / fps
    fx = np.interp(ft, t + args.gps_offset, wx)
    fy = np.interp(ft, t + args.gps_offset, wy)
    heading = compute_heading(fx, fy, fps, args.mpp)

    step = max(1, round(fps / 5))                 # 軌跡描画用に約 5Hz に間引き
    dec = np.column_stack([fx[::step], fy[::step]])

    # --- タイムコード決定
    tag_input = False
    if tc_in:
        start_tc = tc_in
        print(f"入力の TC をそのまま使用: {start_tc}")
    else:
        clock = start_clock_from_gps(good, args.tz, args.gps_offset)
        if clock is None and ctime:
            print("注意: GPS 時刻が無いため creation_time から TC を算出します（精度は低め）")
            dt = datetime.fromisoformat(ctime.replace("Z", "+00:00"))
            clock = (dt.timestamp() + args.tz * 3600) % 86400
        if clock is None:
            print("注意: 時刻情報が無いため TC は 00:00:00:00 から始めます")
            clock = 0.0
        start_tc = clock_to_tc(clock, num, den)
        tag_input = not args.no_tag_input
        print(f"GPS 時刻から算出した開始 TC: {start_tc}（UTC{args.tz:+g}）")

    if tag_input:
        tagged = inp.with_name(inp.stem + "_tc.mov")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(inp),
             "-map", "0:v:0", "-map", "0:a?", "-c", "copy",
             "-timecode", start_tc, str(tagged)])
        print(f"入力に TC を付けたコピーを作成: {tagged}")

    # --- 描画とエンコード
    vm = VectorMap(args.pmtiles, zf, args.mpp, dec, track_light, ss=args.supersample)
    print(f"縮尺 {args.mpp} m/px（タイル z{vm.z}, 実効ズーム {zf:.2f}）")
    if args.lite:
        print(f"--lite: OpenCV で地図変換, libx264 preset {preset}")
    font = load_font(30)

    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT}x{OUT}",
         "-framerate", f"{num}/{den}", "-i", "-",
         "-c:v", "libx264", "-preset", preset, "-crf", str(args.crf),
         "-pix_fmt", "yuv420p", "-timecode", start_tc, str(out)],
        stdin=subprocess.PIPE)

    t_start = time.time()
    try:
        for i in range(nframes):
            theta = float(heading[i]) if args.heading_up else 0.0
            trav = np.vstack([dec[: i // step + 1], [[fx[i], fy[i]]]])
            frame = render_frame(vm, float(fx[i]), float(fy[i]), theta, float(heading[i]),
                                 trav, icon_rgb, track_rgb, font, args.mpp, lite=args.lite)
            enc.stdin.write(frame.tobytes())
            if i % 30 == 0 or i == nframes - 1:
                el = time.time() - t_start
                rate = (i + 1) / el if el > 0 else 0
                eta = (nframes - i - 1) / rate if rate else 0
                print(f"\r{i + 1}/{nframes} フレーム  {rate:.1f} fps  残り {eta / 60:.1f} 分",
                      end="", flush=True)
    except BrokenPipeError:
        sys.exit("\nffmpeg が異常終了しました")
    finally:
        if enc.stdin:
            enc.stdin.close()
        enc.wait()
    print()
    if enc.returncode != 0:
        sys.exit("ffmpeg のエンコードに失敗しました")
    print(f"出力: {out}（{nframes} フレーム, TC {start_tc}）")


if __name__ == "__main__":
    main()
