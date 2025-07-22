"""
# DiTing Foreshock Detection and Analysis auto-workflow
# Usage: python main.py --config-file [path to config files]
"""
import obspy
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import argparse
import os
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import time
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.geodesic import Geodesic

def readxml(xmlpath):
    
    tree = ET.parse(xmlpath)  # 替换为你的文件路径
    root = tree.getroot()

    # 设置固定值
    agency_id = "IGP"
    author = "DTFlow"
    phase = "P"  # 你可以修改为 "P" 或自动识别
    pick_type = "Pick"

    # 提取 Pick 数据
    output = []
    pick_times = []
    for pick in root.find('Picks').findall('Pick'):
        network = pick.find('Network').text
        station = pick.find('Station').text
        location = pick.find('Location').text
        channel = pick.find('Channel').text
        time_raw = pick.find('Time').text
        utc_time = obspy.UTCDateTime(time_raw)
        time = utc_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]+"Z"  # 保留三位小数

        pick_times.append(time)
        pick_id = f"{network}.{station}.{location}.{channel}.{time}"

        item = {
            "Site": {
                "Network": network,
                "Station": station
            },
            "Type": pick_type,
            "Phase": phase,
            "Time": time,
            "ID": pick_id,
            "Source": {
                "AgencyID": agency_id,
                "Author": author
            }
        }

        output.append(item)
    
    return output, pick_times

def plot_detection_event_cartopy(json_file, nastation_unique, figsavepath):
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取事件 ID 和震中位置
    event_id = data.get("ID", "unknown_event")
    hypo = data.get("Hypocenter", {})
    ev_lon = hypo.get("Longitude")
    ev_lat = hypo.get("Latitude")

    # 提取 Pick 信息
    picks = data.get("Data", [])
    pick_info = []
    for pick in picks:
        if pick.get("Type") != "Pick":
            continue
        site = pick.get("Site", {})
        assoc = pick.get("AssociationInfo", {})
        pick_info.append({
            "net": site.get("Network"),
            "sta": site.get("Station"),
            "residual": assoc.get("Residual", np.nan),  # 保留原始残差值
            "distance": assoc.get("Distance", np.nan),
            "azimuth": assoc.get("Azimuth", np.nan)
        })

    pick_df = pd.DataFrame(pick_info)
    # 合并台站坐标信息
    merged = pd.merge(pick_df, nastation_unique, how='left', on=['net', 'sta'])
    merged.dropna(subset=['long', 'lat'], inplace=True)

    # 设置地图范围
    lon_min = min(merged['long'].min(), ev_lon) - 1
    lon_max = max(merged['long'].max(), ev_lon) + 1
    lat_min = min(merged['lat'].min(), ev_lat) - 1
    lat_max = max(merged['lat'].max(), ev_lat) + 1

    # 创建 Cartopy 图像
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # 添加地图要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3)

    # 添加网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # 残差颜色映射（保留正负）
    residual_min = merged['residual'].min()
    residual_max = merged['residual'].max()
    abs_max = max(abs(residual_min), abs(residual_max))  # 对称范围
    norm = Normalize(vmin=-abs_max, vmax=abs_max)
    cmap = cm.get_cmap('coolwarm')

    # 绘制台站
    sc = ax.scatter(merged['long'], merged['lat'], 
                    c=merged['residual'], cmap=cmap, norm=norm,
                    s=100, marker='^', edgecolor='black', label='Stations',
                    transform=ccrs.PlateCarree())

    # 台站标注和虚线连接
    for _, row in merged.iterrows():
        ax.text(row['long'] + 0.05, row['lat'] + 0.05, row['net'] + "." + row['sta'],
                fontsize=8, transform=ccrs.PlateCarree())

        ax.plot([ev_lon, row['long']], [ev_lat, row['lat']],
                linestyle='--', color='yellow', linewidth=1,
                transform=ccrs.PlateCarree(), alpha=0.8)

    # 绘制震中
    ax.scatter(ev_lon, ev_lat, s=200, color='red', marker='*', edgecolor='black',
               label='Hypocenter', transform=ccrs.PlateCarree())

    # 添加颜色条
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.6, pad=0.02)
    cbar.set_label("Residual (s)")

    # 标题和图例
    ax.set_title(f"Seismic Detection Map: {event_id}", fontsize=16)
    ax.legend(loc='upper right')

    # 保存图像
    plt.savefig(figsavepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像已保存：{figsavepath}")


def glass3(cfgs, pickxml):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))   
    p = Path(pickxml)
    # 提取日期目录
    date_folder = p.parent.name  # '20250512'
    # 提取文件前缀（去掉扩展名）
    filename = p.stem  # 'IGP2025jflu_20250512100735'
    # 提取根目录路径（到 events/）
    # root_dir = str(p.parents[2]) + '/'  # '/home/zm/seiscomp/DiTing/CHINA/events/'
    
    workdir_ = filename.split("_")[0]
    job_folder = f"workdir/{workdir_}"
    if not os.path.exists(job_folder):
        os.mkdir(job_folder)
        
    glass_folder = job_folder + os.sep + cfgs['glass.d']['ConfigDirectory']
    
    if not os.path.exists(glass_folder):
        os.mkdir(glass_folder)
        
    cfgs['glass.d']['ConfigDirectory'] = glass_folder
    # glass file
    with open(glass_folder + os.sep + 'glass.d','w') as f:
        f.write('%s' % json.dumps(cfgs['glass.d']))
    # input.d
    input_file = cfgs['glass.d']['InputConfig']
    if input_file not in cfgs:
        print('Input file is required!', flush=True)
        exit()
    input_folder = glass_folder + os.sep + cfgs[input_file]['InputDirectory']
    if not os.path.exists(input_folder):
        os.mkdir(input_folder)
    cfgs[input_file]['InputDirectory'] = input_folder
    archive_folder = glass_folder + os.sep + cfgs[input_file]['ArchiveDirectory']
    if not os.path.exists(archive_folder):
        os.mkdir(archive_folder)
    cfgs[input_file]['ArchiveDirectory'] = archive_folder
    with open(glass_folder + os.sep + input_file,'w') as f:
        f.write('%s' % json.dumps(cfgs[input_file]))
    # output.d
    output_file = cfgs['glass.d']['OutputConfig']
    if output_file not in cfgs:
        print('Output file is required!', flush=True)
        exit()
    output_folder = glass_folder + os.sep + cfgs[output_file]['OutputDirectory']
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    cfgs[output_file]['OutputDirectory'] = output_folder
    with open(glass_folder + os.sep + output_file,'w') as f:
        f.write('%s' % json.dumps(cfgs[output_file]))
    # process each grid file
    for grid_file in cfgs['glass.d']['GridFiles']:
        if grid_file not in cfgs:
            print('Grid file: ' + grid_file + ' is required!', flush=True)
            exit()
        with open(glass_folder + os.sep + grid_file,'w') as f:
            f.write('%s' % json.dumps(cfgs[grid_file]))
    # initialized.d
    initialize_file = cfgs['glass.d']['InitializeFile']
    if initialize_file not in cfgs:
        print('Initialize file is required!', flush=True)
        exit()
    with open(glass_folder + os.sep + initialize_file,'w') as f:
        f.write('%s' % json.dumps(cfgs[initialize_file]))
        
    stationlist = pd.read_csv("./station_coordinates.csv")
    nastation_unique = stationlist.drop_duplicates(subset=["net", "sta"])
    # process stations
    json_stations = {"Type":"StationInfoList","StationList":[]}
    for index in range(len(nastation_unique)):
        net = nastation_unique.iloc[index]["net"]
        sta = nastation_unique.iloc[index]["sta"]
        longnitude = nastation_unique.iloc[index]["long"]
        latitude = nastation_unique.iloc[index]["lat"]
        elev = nastation_unique.iloc[index]["elev"]
        stt = {'Network':net,'Station':sta,\
               'Latitude':latitude,\
                'Longitude':longnitude,'Elevation':elev}
        json_stations['StationList'].append({"Type":"StationInfo","Site":stt,\
                      "Quality":1.0,"Enable":True,"Use":True,"UseForTeleseismic":True})
    # write stations
    #print(json_stations)
    with open(glass_folder + os.sep + cfgs['glass.d']['StationList'],'w') as f:
        f.write('%s' % json.dumps(json_stations))

    picks, pick_times = readxml(pickxml)
    # write picks
    # sort by pick time
    n_picks = len(picks)
    idx = sorted(range(n_picks), key=lambda k:pick_times[k])
    with open(input_folder+os.sep+'picks.'+cfgs[cfgs['glass.d']['InputConfig']]['Format'],'w') as f:
        for i_pick in range(n_picks):
            pick_idx = idx[i_pick]
            f.write('%s\n' % json.dumps(picks[pick_idx]))

    # run glass
    print('Starting Phase Association', flush=True)
    os.system('./glass-app '+glass_folder+os.sep+'glass.d')

    # post-process
    # convert to csv
    event_lat = []
    event_lon = []
    event_dep = []
    events = []
    event_file_list = list(Path(output_folder).glob('*.jsondetect'))
    for index, event_file in enumerate(event_file_list):
        with open(event_file,'r') as f:
            event_dict = json.load(f)
        event_lat.append(event_dict['Hypocenter']['Latitude'])
        event_lon.append(event_dict['Hypocenter']['Longitude'])
        event_dep.append(event_dict['Hypocenter']['Depth'])
        event_time = event_dict['Hypocenter']['Time']
        events.append({'Time':event_time,'Latitude':event_dict['Hypocenter']['Latitude'],\
                'Longitude':event_dict['Hypocenter']['Longitude'],'Depth':event_dict['Hypocenter']['Depth'],\
                'Pick':len(event_dict['Data'])})
        figsavepath = os.path.join("plot", filename+f"_glass_{index}.png")
        plot_detection_event_cartopy(event_file, nastation_unique, figsavepath)
    
if __name__ == '__main__':
    # get the absolute path of the directory containing the current python script
    
    cfgs = yaml.load(open("./GlassConfig.yaml"), Loader=yaml.SafeLoader)
    glass3(cfgs, "IGP2025kvlv_20250604093332.xml")