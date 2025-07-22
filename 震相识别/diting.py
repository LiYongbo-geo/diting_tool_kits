import os
import requests
import obspy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from obspy.signal.trigger import trigger_onset
from sklearn.cluster import AgglomerativeClustering
from requests.auth import HTTPBasicAuth
import math
import time

def clean_json(obj):
    if isinstance(obj, float):
        return float(np.nan_to_num(obj, nan=0.0, posinf=1e10, neginf=-1e10))
    elif isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(item) for item in obj]
    else:
        return obj
data = {
    "value": float('nan'),
    "another": 1.0,
    "list": [1.0, float('inf')]
}

def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    modified from https://github.com/smousavi05/EQTransformer
    Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.

    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.

    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).

    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.

    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).

    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.

    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.
    Returns
    ---------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Modified from
   ----------------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind

def postprocesser_ev_center(yh1, yh2, yh3, det_th=0.3, p_th=0.3, p_mpd=10, s_th=0.3, s_mpd=10, ev_tolerance = 100, p_tolerance = 500):

    """
    modified from https://github.com/smousavi05/EQTransformer
    Postprocessing to detection and phase picking
    """
    detection = trigger_onset(yh1, det_th, det_th)
    pp_arr = _detect_peaks(yh2, mph=p_th, mpd=p_mpd)
    ss_arr = _detect_peaks(yh3, mph=s_th, mpd=s_mpd)

    P_PICKS = {}
    S_PICKS = {}
    EVENTS = {}
    matches = list()

    # P
    if len(pp_arr) > 0:
        for pick in range(len(pp_arr)):
            pauto = pp_arr[pick]
            if pauto:
                P_prob = np.round(yh2[int(pauto)], 3)
                P_PICKS.update({pauto : [P_prob]})
    # S
    if len(ss_arr) > 0:
        for pick in range(len(ss_arr)):
            sauto = ss_arr[pick]
            if sauto:
                S_prob = np.round(yh3[int(sauto)], 3)
                S_PICKS.update({sauto : [S_prob]})

    if len(detection) > 0:
        # merge close detections
        for ev in range(1,len(detection)):
            if detection[ev][0] - detection[ev-1][1] < ev_tolerance:
                detection[ev-1][1] = detection[ev][1]
                detection[ev][0] = -1
                detection[ev][1] = -1

        for ev in range(len(detection)):
            D_prob = np.mean(yh1[detection[ev][0]:detection[ev][1]])
            D_prob = np.round(D_prob, 3)
            EVENTS.update({ detection[ev][0] : [D_prob, detection[ev][1]]})

    # matching the detection and picks
    for ev in EVENTS:
        bg = ev
        ed = EVENTS[ev][1]

        if int(ed-bg) >= ev_tolerance:
            candidate_Ps = list()
            for Ps, P_val in P_PICKS.items():
                if Ps > bg - p_tolerance and Ps < bg + p_tolerance:
                    candidate_Ps.append([Ps, P_val[0]])
            candidate_Ss = list()
            for Ss, S_val in S_PICKS.items():
                if Ss > bg and Ss < ed:
                    candidate_Ss.append([Ss, S_val[0]])
            if len(candidate_Ps) == 0:
                continue
            else:
                #keep the max prob P pick
                max_prob = -10
                for Ps, P_val in candidate_Ps:
                    if P_val > max_prob:
                        max_prob = P_val
                candidate_Ps = [[Ps, P_val] for Ps, P_val in candidate_Ps if P_val == max_prob]
                candidate_Ps = [candidate_Ps[0]]

            if len(candidate_Ss) == 0:
                candidate_Ss.append([np.nan, np.nan])
            else:
                #keep the max prob S pick
                max_prob = -10
                for Ss, S_val in candidate_Ss:
                    if S_val > max_prob:
                        max_prob = S_val
                candidate_Ss = [[Ss, S_val] for Ss, S_val in candidate_Ss if S_val == max_prob]
                candidate_Ss = [candidate_Ss[0]]

            if len(candidate_Ps) != 0 and len(candidate_Ss) != 0:
                matches.append([bg, candidate_Ps, candidate_Ss])
    return matches

def detect_peaks_post_process(yh1, yh2, p_th=0.3, p_mpd=10, s_th=0.3, s_mpd=10):

    pp_arr = _detect_peaks(yh1, mph=p_th, mpd=p_mpd)
    ss_arr = _detect_peaks(yh2, mph=s_th, mpd=s_mpd)
    P_PICKS = {}
    S_PICKS = {}
    # P
    if len(pp_arr) > 0:
        for pick in range(len(pp_arr)):
            pauto = pp_arr[pick]
            if pauto:
                P_prob = np.round(yh1[int(pauto)], 3)
                P_PICKS.update({pauto : [P_prob]})
    # S
    if len(ss_arr) > 0:
        for pick in range(len(ss_arr)):
            sauto = ss_arr[pick]
            if sauto:
                S_prob = np.round(yh2[int(sauto)], 3)
                S_PICKS.update({sauto : [S_prob]})

    return P_PICKS, S_PICKS

def DiTing_EQDet_PhasePick_predict(stream, window_length=10000, step_size=10000, p_th=0.1, s_th=0.1, det_th=0.50):
    data_len = stream[0].data.shape[0]

    # 一次性提取三分量数据，避免重复 select()
    tmp_waveform = np.zeros((data_len, 3))
    tmp_waveform[:, 0] = stream.select(channel='*Z')[0].data
    tmp_waveform[:, 1] = stream.select(channel='*[N1]')[0].data if stream.select(channel='*[N1]') else 0
    tmp_waveform[:, 2] = stream.select(channel='*[E2]')[0].data if stream.select(channel='*[E2]') else 0

    # 计算窗口数
    if data_len < window_length:
        num_windows = 1
        count = np.zeros((1, 3, window_length))
        confidence = np.zeros((1, 3, window_length))
    else:
        num_windows = (data_len - window_length) // step_size + 1
        count = np.zeros((1, 3, data_len))
        confidence = np.zeros((1, 3, data_len))

    for i in tqdm(range(num_windows)):
        start = i * step_size
        end = start + window_length
        window = tmp_waveform[start:end].copy()

        # 标准化向量化处理（避免循环）
        means = window.mean(axis=0)
        stds = window.std(axis=0)
        stds[stds == 0] = 1  # 防止除0
        window = (window - means) / stds
        # 补0（如果末尾不足 window_length）
        if window.shape[0] < window_length:
            window = np.pad(window, ((0, window_length - window.shape[0]), (0, 0)), mode='constant')

        # 转置格式一次性操作
        window_array = window.T[np.newaxis, :, :]  # shape: (1, 3, 10000)
        # 避免慢的 tolist() 和 clean_json（如你能控制服务端，推荐直接传 numpy）
        t_req0 = time.time()
        #cleaned_data = clean_json(window_array.tolist())
        cleaned_data = np.nan_to_num(window_array, nan=0.0, posinf=1e10, neginf=-1e10).tolist()
        #print("数据清洗：", time.time()-t_req0)

        # 记录请求时间
        t_req = time.time()
        try:
            response = requests.post(
                "http://124.17.4.221:30589/ybmodel01/dpk",
                json={'array_data':cleaned_data},
            )
            print("API请求时间：", time.time()-t_req)
        except:
            print("input data:", cleaned_data)

        try:
            output_np = np.array(response.json()['result'])
            confidence[:, :, start:end] = output_np
            count[:, :, start:end] += 1
        except Exception as e:
            print("响应错误：", response.text)

        # 平均值处理
    confidence = np.divide(confidence, count, out=np.zeros_like(confidence), where=(count != 0))
    events = postprocesser_ev_center(
        yh1=confidence[0, 0, :], yh2=confidence[0, 1, :], yh3=confidence[0, 2, :], 
        p_th=p_th, s_th=s_th, det_th=det_th
    )
    num = len(events)
    if not events:
        events = [[np.nan, [[np.nan, np.nan]], [[np.nan, np.nan]]]]
    return events, confidence, num
