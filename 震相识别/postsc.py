import os
import pytz
from datetime import datetime
import obspy
import matplotlib.pyplot as plt
import time

def plotPick(st, eventID, p_time, pP, s_time, sP, net = None, sta = None):
    """
    绘制三通道图像，并将 p_time 标注在图中，保存结果到 results/plot 文件夹。
    
    Parameters:
    st : obspy.Stream
        输入的地震波形流数据。
    eventID : str
        事件 ID，用于文件命名。
    p_time : obspy.UTCDateTime
        P波时间，用于截取数据和标注。
    """
    # 截取从 P波前后 20 秒的时间段
    if s_time == "nan":
        st_trim = st.trim(starttime=p_time - 30, endtime=p_time + 30)
            # 创建一个图形对象
        fig, ax = plt.subplots(nrows=3, figsize=(10, 8), sharex=True)
        # 遍历每个通道并绘制
        for i, tr in enumerate(st_trim):
            ax[i].plot(tr.times(), tr.data, label=tr.id, color='black')
            ax[i].axvline(p_time - st_trim[0].stats.starttime, color='red', linestyle='--', label=f"pP={pP}")
            ax[i].set_ylabel(tr.stats.channel)
            ax[i].legend(loc='upper right')
        
        # 设置x轴标签
        ax[2].set_xlabel("Time (s)")
        # 添加标题
        fig.suptitle(f"Waveform for Event: {eventID}_{p_time}", fontsize=16)
        if pP < 0.5:
            prefix = "lq"
        elif 0.5<=pP<0.75:
            prefix = "mq"
        elif 0.59<=pP<0.9:
            prefix = "hq"
        else:
            prefix = "eq"
        # 保存图像
        output_dir = f"./results/plot/{eventID[0:4]}/{eventID[4:6]}/{eventID[6:8]}/{net}/{sta}"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"文件夹 '{output_dir}' 已创建。")
        else:
            print(f"文件夹 '{output_dir}' 已存在。")
            
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, f"{eventID}_waveform.png")
        plt.savefig(plot_filename)
        plt.close(fig)
        
        output_dir_streams = f"./results/eventstreams/{eventID[0:4]}/{eventID[4:6]}/{eventID[6:8]}/{net}/{sta}"
        if not os.path.exists(output_dir_streams):
            os.makedirs(output_dir_streams)
            print(f"文件夹 '{output_dir_streams}' 已创建。")
        else:
            print(f"文件夹 '{output_dir_streams}' 已存在。")
        stream_filename = os.path.join(output_dir_streams, f"{eventID}.mseed")
        st_trim.write(stream_filename, format="MSEED") 
    
    else:
        st_trim = st.trim(starttime=p_time - 15, endtime=s_time + 30)
            # 创建一个图形对象
        fig, ax = plt.subplots(nrows=3, figsize=(10, 8), sharex=True)
        # 遍历每个通道并绘制
        for i, tr in enumerate(st_trim):
            ax[i].plot(tr.times(), tr.data, label=tr.id, color='black')
            ax[i].axvline(p_time - st_trim[0].stats.starttime, color='red', linestyle='--', label=f"pP={pP}")
            ax[i].axvline(s_time - st_trim[0].stats.starttime, color='purple', linestyle='--', label=f"sP={sP}")
            ax[i].set_ylabel(tr.stats.channel)
            ax[i].legend(loc='upper right')
        
        # 设置x轴标签
        ax[2].set_xlabel("Time (s)")
        # 添加标题
        fig.suptitle(f"{eventID}_{p_time}", fontsize=12)
        if pP < 0.5:
            prefix = "lq"
        elif 0.5<=pP<0.75:
            prefix = "mq"
        elif 0.59<=pP<0.9:
            prefix = "hq"
        else:
            prefix = "eq"
        # 保存图像
        workerfloder = None
        if workerfloder == None:
            output_dir = f"./results/plot/{eventID[0:4]}/{eventID[4:6]}/{eventID[6:8]}/{net}/{sta}"
        else:
            output_dir = workerfloder + f"./results/plot/{eventID[0:4]}/{eventID[4:6]}/{eventID[6:8]}/{net}/{sta}"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"文件夹 '{output_dir}' 已创建。")
        else:
            print(f"文件夹 '{output_dir}' 已存在。")
           
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, f"{eventID}_waveform.png")
        plt.savefig(plot_filename)
        plt.close(fig)
        
        output_dir_streams = f"./results/eventstreams/{eventID[0:4]}/{eventID[4:6]}/{eventID[6:8]}/{net}/{sta}"
        if not os.path.exists(output_dir_streams):
            os.makedirs(output_dir_streams)
            print(f"文件夹 '{output_dir_streams}' 已创建。")
        else:
            print(f"文件夹 '{output_dir_streams}' 已存在。")
        stream_filename = os.path.join(output_dir_streams, f"{eventID}.mseed")
        st_trim.write(stream_filename, format="MSEED") 
    
    print(f"Plot saved as {plot_filename}")