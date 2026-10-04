#!/usr/bin/env python3
"""DJI Osmo Action の動画から GPS 記録を抜き出し、NMEA と KML (Google マップ用) を出力する。

使い方:
    extract_gps.py /path/to/DJI_xxxx.MP4 [--hz 10]

動画と同じディレクトリに以下を出力する:
    DJI_xxxx.nmea  ... $GPRMC / $GPGGA 形式
    DJI_xxxx.kml   ... Google マイマップ / Google Earth にインポートできる軌跡
"""

import argparse
import math
import sys
from datetime import timedelta
from pathlib import Path
from xml.sax.saxutils import escape

from pyosmogps import OsmoGps


def assign_precise_times(points, frame_rate):
    """GPS の時刻は秒単位なので、フレーム番号から秒未満の時刻を補う。

    秒が切り替わった最初のフレームを基準にして、各フレームに
    基準時刻 + (フレーム差 / フレームレート) を割り当てる。
    """
    anchor = next(
        (i for i in range(1, len(points))
         if points[i]["timeinfo"] != points[i - 1]["timeinfo"]),
        0,
    )
    t_anchor = points[anchor]["timeinfo"]
    for i, p in enumerate(points):
        p["time"] = t_anchor + timedelta(seconds=(i - anchor) / frame_rate)


def resample(points, hz):
    """1/hz 秒ごとに最も近いフレームを採用する。"""
    if hz <= 0:
        return points
    step = timedelta(seconds=1 / hz)
    out = []
    next_t = points[0]["time"]
    for p in points:
        if p["time"] >= next_t:
            out.append(p)
            while next_t <= p["time"]:
                next_t += step
    return out


def distance_m(a, b):
    r = 6371008.8
    la1, la2 = math.radians(a["latitude"]), math.radians(b["latitude"])
    dla = la2 - la1
    dlo = math.radians(b["longitude"] - a["longitude"])
    h = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def bearing_deg(a, b):
    la1, la2 = math.radians(a["latitude"]), math.radians(b["latitude"])
    dlo = math.radians(b["longitude"] - a["longitude"])
    x = math.sin(dlo) * math.cos(la2)
    y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlo)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def nmea_sentence(body):
    cs = 0
    for c in body:
        cs ^= ord(c)
    return f"${body}*{cs:02X}"


def nmea_coord(value, is_lat):
    hemi = ("N" if value >= 0 else "S") if is_lat else ("E" if value >= 0 else "W")
    value = abs(value)
    deg = int(value)
    minutes = (value - deg) * 60
    if is_lat:
        return f"{deg:02d}{minutes:07.4f}", hemi
    return f"{deg:03d}{minutes:07.4f}", hemi


def write_nmea(points, path):
    lines = []
    for i, p in enumerate(points):
        t = p["time"]
        hhmmss = t.strftime("%H%M%S") + f".{t.microsecond // 10000:02d}"
        ddmmyy = t.strftime("%d%m%y")
        lat, ns = nmea_coord(p["latitude"], True)
        lon, ew = nmea_coord(p["longitude"], False)

        # 速度・進行方向は前後の点から計算する
        a = points[max(i - 1, 0)]
        b = points[min(i + 1, len(points) - 1)]
        dt = (b["time"] - a["time"]).total_seconds()
        dist = distance_m(a, b)
        speed_knots = dist / dt * 1.943844 if dt > 0 else 0.0
        course = bearing_deg(a, b) if dist > 0 else 0.0

        lines.append(nmea_sentence(
            f"GPRMC,{hhmmss},A,{lat},{ns},{lon},{ew},"
            f"{speed_knots:.2f},{course:.1f},{ddmmyy},,,A"
        ))
        lines.append(nmea_sentence(
            f"GPGGA,{hhmmss},{lat},{ns},{lon},{ew},1,,,"
            f"{p['altitude']:.1f},M,,M,,"
        ))
    path.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")


def write_kml(points, path, name):
    coords = "\n".join(
        f"{p['longitude']:.7f},{p['latitude']:.7f},{p['altitude']:.1f}" for p in points
    )
    start, end = points[0], points[-1]

    def placemark(label, p, style):
        return f"""    <Placemark>
      <name>{label}</name>
      <description>{p['time']:%Y-%m-%d %H:%M:%S} UTC</description>
      <styleUrl>#{style}</styleUrl>
      <Point><coordinates>{p['longitude']:.7f},{p['latitude']:.7f},{p['altitude']:.1f}</coordinates></Point>
    </Placemark>"""

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{escape(name)}</name>
    <Style id="track"><LineStyle><color>ff0000ff</color><width>4</width></LineStyle></Style>
    <Style id="start"><IconStyle><color>ff00ff00</color></IconStyle></Style>
    <Style id="end"><IconStyle><color>ff0000ff</color></IconStyle></Style>
    <Placemark>
      <name>{escape(name)}</name>
      <description>{start['time']:%Y-%m-%d %H:%M:%S} - {end['time']:%H:%M:%S} UTC</description>
      <styleUrl>#track</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>
{coords}
        </coordinates>
      </LineString>
    </Placemark>
{placemark("Start", start, "start")}
{placemark("End", end, "end")}
  </Document>
</kml>
"""
    path.write_text(kml, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Osmo Action の動画から GPS を抽出して NMEA / KML を出力")
    ap.add_argument("video", type=Path, help="入力動画 (.MP4)")
    ap.add_argument("--hz", type=float, default=10,
                    help="NMEA の出力頻度 [Hz] (0 で全フレーム, デフォルト 10)")
    args = ap.parse_args()

    video = args.video.expanduser().resolve()
    if not video.is_file():
        sys.exit(f"ファイルが見つかりません: {video}")

    gps = OsmoGps([str(video)])
    # 測位前 (0,0) の点は捨てる
    points = [p for p in gps.gps_data if p["latitude"] != 0 or p["longitude"] != 0]
    if not points:
        sys.exit("GPS データが見つかりませんでした (リモコンの GPS が未接続/未測位の可能性)")

    assign_precise_times(points, gps.input_frame_rate)
    points = resample(points, args.hz)

    nmea_path = video.with_suffix(".nmea")
    kml_path = video.with_suffix(".kml")
    write_nmea(points, nmea_path)
    write_kml(points, kml_path, video.stem)

    total = sum(distance_m(a, b) for a, b in zip(points, points[1:]))
    print(f"{len(points)} 点 / {points[0]['time']:%H:%M:%S} - {points[-1]['time']:%H:%M:%S} UTC"
          f" / 距離 {total:.0f} m")
    print(f"NMEA: {nmea_path}")
    print(f"KML : {kml_path}")


if __name__ == "__main__":
    main()
