import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic
from matplotlib.lines import Line2D
import numpy as np
import os
import time

def parse_event_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    latitude = float(root.find("Latitude").text.strip())
    longitude = float(root.find("Longitude").text.strip())

    pick_stations = set()
    max_pick_distance = 0.0

    picks = root.find("Picks")
    if picks is not None:
        for pick in picks.findall("Pick"):
            net = pick.find("Network").text.strip().upper() if pick.find("Network") is not None else "Unknown"
            sta = pick.find("Station").text.strip().upper() if pick.find("Station") is not None else "Unknown"
            pick_stations.add((net, sta))

            pick_lat = pick.find("Latitude").text.strip() if pick.find("Latitude") is not None else None
            pick_lon = pick.find("Longitude").text.strip() if pick.find("Longitude") is not None else None

            if pick_lat is not None and pick_lon is not None:
                pick_lat = float(pick_lat)
                pick_lon = float(pick_lon)
                dist = geodesic((latitude, longitude), (pick_lat, pick_lon)).km
                max_pick_distance = max(max_pick_distance, dist)

    return latitude, longitude, pick_stations, max_pick_distance

def load_station_coordinates(csv_path):
    df = pd.read_csv(csv_path)
    station_coords = {}
    for _, row in df.iterrows():
        net = str(row["net"]).strip().upper()
        sta = str(row["sta"]).strip().upper()
        lat = float(row["lat"])
        lon = float(row["long"])
        station_coords[(net, sta)] = (lat, lon)
    return station_coords

def plot_event_and_stations(xml_path, csv_path):
    event_lat, event_lon, pick_stations, max_pick_distance = parse_event_xml(xml_path)
    station_coords = load_station_coordinates(csv_path)

    in_range_no_pick = []
    picked = []
    out_of_range = []

    for key, (lat, lon) in station_coords.items():
        dist_km = geodesic((event_lat, event_lon), (lat, lon)).km
        dist_deg = dist_km / 111.32

        if key in pick_stations:
            picked.append((key, lat, lon, dist_deg))
        elif dist_deg <= max_pick_distance:
            in_range_no_pick.append((key, lat, lon, dist_deg))
        else:
            out_of_range.append((key, lat, lon, dist_deg))

    locs = [(lat, lon) for (_, lat, lon, _) in picked]
    longspicked = [lon for (_, lon) in locs]
    latspicked = [lat for (lat, _) in locs]

    max_lat = max(latspicked + [event_lat])
    min_lat = min(latspicked + [event_lat])
    max_lon = max(longspicked + [event_lon])
    min_lon = min(longspicked + [event_lon])

    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([min_lon - 2, max_lon + 2, min_lat - 2, max_lat + 2], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines(draw_labels=True)

    ax.plot(event_lon, event_lat, marker='*', color='red', markersize=15, label='Event', zorder=5)

    for (key, lat, lon, _) in picked:
        ax.plot(lon, lat, marker='^', color='yellow', markersize=8, transform=ccrs.PlateCarree(),
                zorder=4, markeredgecolor='black', markeredgewidth=0.8)
        ax.plot([event_lon, lon], [event_lat, lat], color='orange', linestyle='--', linewidth=1,
                transform=ccrs.PlateCarree(), alpha=0.8)
        ax.text(lon + 0.1, lat + 0.1, f"{key[0]}.{key[1]}", fontsize=8, transform=ccrs.PlateCarree())

    for (_, lat, lon, _) in out_of_range:
        ax.plot(lon, lat, marker='o', color='lightgray', markersize=5, transform=ccrs.PlateCarree(), zorder=1)

    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='Event', markerfacecolor='red', markersize=12),
        Line2D([0], [0], marker='^', color='w', label='Picked Station', markerfacecolor='yellow', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='No Pick', markerfacecolor='lightgray', markersize=6),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.title("Seismic Event & Station Map")
    plt.tight_layout()
    print("绘图完成")
    return fig

def process_and_log_files(xml_root, plot_root, csv_path, log_file):
    # 读取已处理的文件
    if os.path.exists(log_file):
        with open(log_file, 'r') as log:
            processed_files = set(log.read().splitlines())
    else:
        processed_files = set()

    # 扫描并处理已有的 XML 文件
    for root, _, files in os.walk(xml_root):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root, file)
                if xml_path not in processed_files:
                    try:
                        fig = plot_event_and_stations(xml_path, csv_path)
                        if fig:
                            # 获取文件夹路径并创建目录
                            rel_path = os.path.relpath(xml_path, xml_root)
                            rel_dir = os.path.dirname(rel_path)
                            prefix = os.path.splitext(file)[0].split("-")[0]
                            output_dir = os.path.join(plot_root, rel_dir)
                            os.makedirs(output_dir, exist_ok=True)

                            # 保存图像
                            fig.savefig(os.path.join(output_dir, f"{prefix}.png"))
                            fig.savefig(os.path.join(output_dir, f"{prefix}.pdf"))
                            plt.close(fig)

                            # 记录处理过的文件
                            with open(log_file, 'a') as log:
                                log.write(f"{xml_path}\n")
                            print(f"已保存: {prefix} 到 {output_dir}")
                    except Exception as e:
                        print(f"处理失败: {xml_path}，错误信息：{e}")

    # 实时监控新文件
    print("开始监控新文件...")
    while True:
        for root, _, files in os.walk(xml_root):
            for file in files:
                if file.endswith(".xml"):
                    xml_path = os.path.join(root, file)
                    if xml_path not in processed_files:
                        try:
                            fig = plot_event_and_stations(xml_path, csv_path)
                            if fig:
                                # 获取文件夹路径并创建目录
                                rel_path = os.path.relpath(xml_path, xml_root)
                                rel_dir = os.path.dirname(rel_path)
                                prefix = os.path.splitext(file)[0].split("_")[0]
                                output_dir = os.path.join(plot_root, rel_dir)
                                os.makedirs(output_dir, exist_ok=True)

                                # 保存图像
                                fig.savefig(os.path.join(output_dir, f"{prefix}.png"))
                                fig.savefig(os.path.join(output_dir, f"{prefix}.pdf"))
                                plt.close(fig)

                                # 记录处理过的文件
                                with open(log_file, 'a') as log:
                                    log.write(f"{xml_path}\n")
                                print(f"已保存: {prefix} 到 {output_dir}")
                        except Exception as e:
                            print(f"处理失败: {xml_path}，错误信息：{e}")

        time.sleep(10)  # 每10秒检查一次新文件

# ============================== 
# ✅ 主入口
csv_path = "station_coordinates.csv"
plot_root = "./plot"
xml_path = "/home/zm/seiscomp/DiTing/CHINA/events/xml/20250515/IGP2025jkhz_20250515014436.xml"
fig = plot_event_and_stations(xml_path, csv_path)
fig.savefig("./plot/GP2025jkhz.png")
fig.savefig("./plot/GP2025jkhz.pdf")
plt.close(fig)

