
from __future__ import annotations

from scipy.io import loadmat
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from itertools import groupby
from scipy.signal import find_peaks, detrend
from statsmodels.robust.scale import mad
from scipy import interpolate
import warnings
warnings.filterwarnings("ignore")
import glob
import mne
from mne.filter import filter_data, notch_filter, resample
from mne.time_frequency import psd_array_multitaper
from tqdm import tqdm 
import datetime
from datetime import timedelta
import h5py
import argparse
from pathlib import Path
from tqdm import tqdm
from filelock import FileLock

VALID_BELT_MODES = ('raw_mean', 'robust_mean', 'summary_mean')

def load_prepared_data(file_path:str, signals_to_load:list=None, annotations_to_load:list=None):
    """ 
    load prepared data format (post March 22, 2023). Currently, all signals are read. 
    Inputs:
    file_path: path to prepared .h5 file.
    Returns Xy, params
    """

    signals = pd.DataFrame([])
    annotations = pd.DataFrame([])

    with h5py.File(file_path, "r") as f:
        
        group = f['signals']
        signals_contained = list(group.keys())
        # if 'c4-m1' is not in signals_to_load but only "eeg" is in signals_contained, then we load "eeg

        if signals_to_load is None:
            signals_to_load = signals_contained
            rename_eeg_to_c4m1 = False
            if 'c4-m1' not in signals_contained and 'eeg' in signals_contained:
                rename_eeg_to_c4m1 = True
                signals_to_load = [x.replace('c4-m1', 'eeg') for x in signals_to_load]
            dataset_names = signals_contained
        else:
            rename_eeg_to_c4m1 = False
            if 'c4-m1' in signals_to_load and ('eeg' in signals_contained and 'c4-m1' not in signals_contained):
                # replace 'c4-m1' with 'eeg':
                rename_eeg_to_c4m1 = True
                signals_to_load = [x.replace('c4-m1', 'eeg') for x in signals_to_load]
            dataset_names = signals_to_load
                
        for dataset_name in dataset_names:
            dataset = group[dataset_name][:]
            assert dataset.shape[1] == 1, "Only one-dimensional datasets expected"
            signals[dataset_name] = dataset.flatten()
        if rename_eeg_to_c4m1:
            signals.rename(columns={'eeg': 'c4-m1'}, inplace=True)
        
        if 'annotations' in f.keys():
            group = f['annotations']
            if annotations_to_load is None:
                dataset_names = list(group.keys())
            else:
                dataset_names = annotations_to_load
            for dataset_name in dataset_names:
                dataset = group[dataset_name][:]
                assert dataset.shape[1] == 1, "Only one-dimensional datasets expected"
                annotations[dataset_name] = dataset.flatten()

        params = {}
        params['fs'] = f.attrs['sampling_rate']
        params['unit_voltage'] = f.attrs['unit_voltage']

        signals = signals.astype(float)
        annotations = annotations.astype(float)
        
    return signals, annotations, params


def isNaN(num):
    return num != num

def envelope(trace, fs=200, r=1000):
    '''
    fs: sampling rate
    r: min distance of peaks, in samples
    '''
    peaks = find_peaks(trace, distance=r)[0]
    troughs = find_peaks(-trace, distance=r)[0]
    
    mvg_avg_dur = 10 # in seconds
    trace_mvg_avg = pd.Series(trace)
    trace_mvg_avg = trace_mvg_avg.rolling(mvg_avg_dur*fs, center=True).mean().values
    trace_mvg_avg[np.isnan(trace_mvg_avg)]=0
    
    peaks = peaks[np.where(trace[peaks] > trace_mvg_avg[peaks]*1.1)[0]]
    troughs =  troughs[np.where(trace[troughs] < trace_mvg_avg[troughs]*0.9)[0]]

    if peaks.shape[0] == 0:
        return None, None

    try:
        f_interp = interpolate.interp1d(peaks, trace[peaks], kind='cubic', fill_value="extrapolate")
        envelope_up = f_interp(range(len(trace)))
    except:
        envelope_up = None
    try:
        f_interp = interpolate.interp1d(troughs, trace[troughs], kind='cubic', fill_value="extrapolate")
        envelope_lo = f_interp(range(len(trace)))
    except:
        envelope_lo = None

    return envelope_up, envelope_lo


def create_env(trace, fs, r = 5):
    'r: in seconds'
    
    r = r*fs; # 5 seconds distance for peaks.
    up,lo =  envelope(trace, fs, r=r)

    if np.any([up is None, lo is None]):
        return up, lo

    ind = np.where(up<lo)[0]
    up[ind] = 0
    lo[ind] = 0
    ind = np.where(up<(0.8*trace))[0]
    for i in range(4*fs,len(ind)):
        up[ind[i]] = np.mean(up[(ind[i]-3*fs):ind[i]])
    ind = np.where(lo>(0.9*trace))[0]
    for i in range(4*fs,len(ind)):
        lo[ind[i]] = np.mean(lo[(ind[i]-3*fs):ind[i]])
    return up,lo

def connect_apneas(th,d,fs):
    Nt = len(d)
    n = round(fs*5000/200); # max time before giving up search for successor

    # forward
    ct = float("inf")
    inOne = 0
    z = np.full([ len(d)], np.nan)
    for i in range(Nt):
        if d[i]<th : 
            ct=0
        if d[i]>=th:
            ct=ct+1
        if ct<n:
            z[i] = -1000

    #backward
    dd = np.fliplr([d])[0]
    ct = float("inf")
    inOne = 0
    zz=np.full([ len(dd)], np.nan)
    for i in range(Nt):
        if dd[i]<th:
            ct=0;
        if dd[i]>=th:
            ct=ct+1; 
        if ct<n:
            zz[i] = -1000  
    zz = np.fliplr([zz])[0]
    ind = np.where(~isNaN(zz))
    z[ind] = -1000;
    return z


def remove_outliers(t,trace,fs, min_height=0.01, do_plots=False):
    
    tPeaks,peaks,tTroughs,troughs = get_peaks_troughs_2(t,trace,min_height,fs)
    t1 = tPeaks.copy()
    t2 = tTroughs.copy()
    y1 = peaks.copy()
    y2 = troughs.copy()
    ThresholdFactor = 5
    tf1 = is_outlier(y1,ThresholdFactor=ThresholdFactor)
    ind1 = list(np.nonzero(tf1)[0]); 
    tf2 = is_outlier(y2,ThresholdFactor=ThresholdFactor)
    ind2 = list(np.nonzero(tf2)[0]); 

    yy1 = y1.copy()
    yy2 = y2.copy()
    yy1 = fill_outliers(yy1,ThresholdFactor=ThresholdFactor)
    yy2 = fill_outliers(yy2,ThresholdFactor=ThresholdFactor)

    #get envelope by interpolation
    f1 = interpolate.interp1d(t1,yy1,bounds_error=False)
    f2 = interpolate.interp1d(t2,yy2,bounds_error=False)
    up = f1(t)
    lo = f2(t)
    lo = [ lo[i] if (lo[i] <= up[i]) else up[i] for i in range(len(lo))]
    up = [ lo[i] if (lo[i] >= up[i]) else up[i] for i in range(len(up))]
    m = np.add(up,lo)/2

    idx = [] # keep track of points that have been "corrected"
    ind = [ i for i in range(len(trace)) if (trace[i] > up[i]) ]
    idx = idx + ind
    trace = [ up[i] if (trace[i] > up[i]) else trace[i] for i in range(len(trace))]
    ind = [ i for i in range(len(trace)) if (trace[i] < lo[i]) ]
    idx = idx + ind
    trace = [ lo[i] if (trace[i] < lo[i]) else trace[i] for i in range(len(trace))]
    idx = np.sort(idx)

    traceN = trace-m; 
#     d = np.array(up) - np.array(lo); 
#     print(np.nanmedian(d))
#     traceN = 600*traceN/np.nanmedian(d); 

    traceN[np.isnan(traceN)] = 0 # 500
    

    if do_plots:
        fig, ax = plt.subplots(5,1, sharex=True, sharey=True, figsize=(16,20), dpi=80)
        ax[0].plot(t, trace,linewidth=1)
        ax[0].scatter(tPeaks, peaks, c='r',s=8)
        ax[0].scatter(tTroughs, troughs, c='r',s=8)
        ax[0].set_xlim([0,1000])
        ax[0].set_ylim([-11,11])
        ax[1].scatter(t1,y1,c='r',s=8)
        ax[1].scatter(t2, y2, c='r',s=8)
        ax[1].scatter(t1[ind1],y1[ind1],c='blue',s=8)
        ax[1].scatter(t2[ind2],y2[ind2],c='blue',s=8)    

    #     ax[2].scatter(t1,yy1,c='r',s=8)
    #     ax[2].scatter(t2,yy2,c='r',s=8)
    #     ax[2].scatter(t1[ind1],yy1[ind1],c='blue',s=8)
    #     ax[2].scatter(t2[ind2],yy2[ind2],c='blue',s=8)
        ax[2].plot(t,trace,linewidth=1)
        ax[2].plot(t,up,c='peru',linewidth=1)
        ax[2].plot(t,lo,c='gold',linewidth=1)

        ax[3].plot(t,trace,linewidth=1)
        ax[3].plot(t,m,c='r',linewidth=1)

        ax[4].plot(t, traceN, linewidth=1)
    
    return idx,traceN

# replaces outliers with the previous non-outlier element
def fill_outliers(y,ThresholdFactor=5):
    outliers = is_outlier(y,ThresholdFactor=ThresholdFactor)
    outliers_indices = np.nonzero(outliers);
    for outlier_index in list(outliers_indices[0]):
        i=1
        while (outliers[outlier_index-i] == 1):
            i += 1
        y[outlier_index] = y[outlier_index-i] 
    return y

def is_outlier(x_array,ThresholdFactor=5):
    MAD = mad(x_array)
    median = np.median(x_array)
    return [1 if ((x > (median+MAD*ThresholdFactor))|(x < (median-MAD*ThresholdFactor))) else 0 for x in x_array] 




def get_peaks_troughs_4(t,trace,mph,mpd):
    i0,properties = find_peaks(trace,height=mph,distance=mpd)
    i1 = [];
    for i in range(0,len(i0)-1):
        ind = list(np.arange(i0[i],i0[i+1]))
        ii = min(trace[ind])
        jj = list(trace[ind]).index(ii) 
        i1.append(ind[jj])

    if len(i0) == 0:
        return None, None, None, None, None, None

    ind = list(np.arange(i0[-1],len(trace)))
    ii = min(trace[ind])
    jj = list(trace[ind]).index(ii) 
    i1.append(ind[jj])
    i1=np.asarray(i1)

    tPeaks =  t[i0] 
    peaks = trace[i0]; 
    tTroughs = t[i1]; 
    troughs = trace[i1]; 

    killIt = np.zeros(len(tPeaks))
    for i in range(0,len(tPeaks)):
        if (peaks[i]/(troughs[i]+.1)<3.5): #3  Eline: 5
            killIt[i]=1
        if troughs[i]>0.9: #1.2 Eline: 100 (based on absolute amplitude)
            killIt[i]=1
        if (abs(tPeaks[i]-tTroughs[i])>100):
            killIt[i]=1

    ind = list(np.where(killIt == 0)[0])
    peaks = peaks[ind] 
    troughs= troughs[ind]
    tPeaks=tPeaks[ind]
    tTroughs=tTroughs[ind]
    i0 = i0[ind]
    i1 = i1[ind]
    return tPeaks,peaks,tTroughs,troughs,i0,i1

def get_peaks_troughs_2(t,trace,mph,mpd):
    i0,properties = find_peaks(trace,height=mph,distance=mpd)
    i2 = [];
    for i in range(0,len(i0)-1):
        ind = list(np.arange(i0[i],i0[i+1]))
        ii = min(trace[ind]); 
        jj = list(trace[ind]).index(ii) 
        i2.append(ind[jj]); 

    ind = list(np.arange(i0[-1],len(trace)))
    ii = min(trace[ind]); 
    jj = list(trace[ind]).index(ii) 
    i2.append(ind[jj]);

    tPeaks =  t[i0].copy()
    peaks = trace[i0].copy(); 
    tTroughs = t[i2].copy(); 
    troughs = trace[i2].copy(); 
    return tPeaks,peaks,tTroughs,troughs


def load_mgh_matlab_data(data_path, fs):
    
    data = loadmat(data_path)
    channels = [data['hdr'][0][x][0][0] for x in range(len(data['hdr'][0]))]
    abd_channel_no = channels.index('ABD')
    chest_channel_no = channels.index('CHEST')
    trace = data['s'][abd_channel_no,:] + data['s'][chest_channel_no,:]
    t = np.arange(len(trace))/fs
    
    return t, trace


def infer_effort_channel_role(channel_name):
    name = str(channel_name).lower()
    if any(token in name for token in ['abd', 'abdom']):
        return 'abd'
    if any(token in name for token in ['chest', 'thorax', 'thoracic', 'tho']):
        return 'chest'
    if 'effort' in name:
        return 'effort'
    return None


def select_effort_signals(signals, channel_names=None):
    signals = np.atleast_2d(np.asarray(signals, dtype=float))
    if signals.shape[0] == 0:
        raise ValueError('No effort signals available for breathing stability computation.')

    if channel_names is None:
        channel_names = [f'effort_{i+1}' for i in range(signals.shape[0])]
    if len(channel_names) != signals.shape[0]:
        raise ValueError('channel_names must match the number of signals.')

    preferred = {}
    fallback = []
    for idx, channel_name in enumerate(channel_names):
        role = infer_effort_channel_role(channel_name)
        if role in ['abd', 'chest'] and role not in preferred:
            preferred[role] = idx
        else:
            fallback.append(idx)

    selected_idx = [preferred[x] for x in ['abd', 'chest'] if x in preferred]
    selected_labels = [x for x in ['abd', 'chest'] if x in preferred]

    if len(selected_idx) == 0:
        selected_idx = fallback if len(fallback) > 0 else [0]
        if len(selected_idx) == 1:
            selected_labels = ['effort']
        else:
            selected_labels = [f'effort_{i+1}' for i in range(len(selected_idx))]

    return signals[selected_idx], selected_labels


def preprocess_effort_signals(signals, fs, notch_freq=60, bandpass_freq=[0, 10], n_jobs=1, new_fs=10):
    traces = np.atleast_2d(np.asarray(signals, dtype=float))
    if notch_freq is not None and np.max(bandpass_freq) >= notch_freq:
        traces = notch_filter(traces, fs, notch_freq, n_jobs=n_jobs, verbose='ERROR')
    traces = filter_data(detrend(traces, axis=1), fs, bandpass_freq[0], bandpass_freq[1], n_jobs=n_jobs, verbose='ERROR')
    traces = resample(traces, up=new_fs, down=fs, axis=1)
    return np.atleast_2d(np.asarray(traces, dtype=float))


def clip_z_normalize(signal):
    signal_clipped = np.clip(signal, np.percentile(signal,2), np.percentile(signal,98))

    med = np.median(signal_clipped)
    iqr = np.percentile(signal_clipped, 75) - np.percentile(signal_clipped, 25)
    signal = (signal - med) / iqr

    return signal


def build_belt_mode_traces(signals, belt_mode='raw_mean'):
    if belt_mode not in VALID_BELT_MODES:
        raise ValueError(f'Invalid belt_mode={belt_mode!r}. Expected one of {VALID_BELT_MODES}.')

    traces = np.atleast_2d(np.asarray(signals, dtype=float))
    normalized_traces = np.vstack([clip_z_normalize(trace.copy()) for trace in traces])

    if belt_mode == 'raw_mean':
        reference_trace = clip_z_normalize(traces.mean(axis=0))
        component_traces = normalized_traces
    else:
        reference_trace = clip_z_normalize(normalized_traces.mean(axis=0))
        component_traces = normalized_traces

    return reference_trace, component_traces


def summarize_stability_by_stage(stability_array, fs, stage, stability_wl):
    if stage is None:
        stage_selections = ['ALL']
    else:
        stage_selections = ['ALL', 'SLEEP', 'N1', 'N2', 'N3', 'NREM', 'REM', 'WAKE']

    stability_summary = {}
    stability_episodes = {}
    for stage_selection in stage_selections:
        if stage_selection == 'ALL':
            stability_array_stage = stability_array.copy()
        else:
            if stage_selection == 'SLEEP':
                vals_stage = [1, 2, 3, 4]
            elif stage_selection == 'NREM':
                vals_stage = [1, 2, 3]
            elif stage_selection == 'N1':
                vals_stage = [3]
            elif stage_selection == 'N2':
                vals_stage = [2]
            elif stage_selection == 'N3':
                vals_stage = [1]
            elif stage_selection == 'REM':
                vals_stage = [4]
            elif stage_selection == 'WAKE':
                vals_stage = [5]
            stability_array_stage = stability_array.copy()
            stability_array_stage = stability_array_stage[np.isin(stage, vals_stage)]

        stability_summary_stage = stability_summary_features(stability_array_stage, fs)
        stability_episodes_stage = stability_episodes_features(stability_array_stage, fs, window_length_min=stability_wl)
        stability_summary.update({f'{stage_selection}_{k}': v for k, v in stability_summary_stage.items()})
        stability_episodes.update({f'{stage_selection}_{k}': v for k, v in stability_episodes_stage.items()})

    return stability_summary, stability_episodes


def compute_trace_outputs(trace, fs, stage, stability_wl, stability_overlap):
    trace = np.asarray(trace, dtype=float).flatten()
    t = np.arange(len(trace))/fs
    _, trace = remove_outliers(t, trace, fs, min_height=0.01)

    up, lo = create_env(trace, fs)
    apneas, hypopneas = detect_central_events(lo, up, fs)
    similarity_array = compute_self_similarity(up, lo, fs)
    apneas, hypopneas = post_processing_detections(apneas, hypopneas, similarity_array, fs)
    stability_array = compute_stability(up, lo, fs, window_length_min=stability_wl, overlap=stability_overlap)
    stability_summary, stability_episodes = summarize_stability_by_stage(stability_array, fs, stage, stability_wl)

    return {
        'trace': trace,
        'apneas': apneas,
        'hypopneas': hypopneas,
        'stability_array': stability_array,
        'stability_summary': stability_summary,
        'stability_episodes': stability_episodes,
    }


def average_metric_dicts(dicts):
    if len(dicts) == 0:
        return {}
    if len(dicts) == 1:
        return dict(dicts[0])

    averaged = {}
    for key in dicts[0].keys():
        values = [d.get(key, np.nan) for d in dicts]
        averaged[key] = float(np.nanmean(values))
    return averaged


def detect_central_events(lo,up,fs):

    d = up - lo 
    centr_detected = np.empty(d.shape)
    peakenv= np.empty(d.shape)
    wdw=3*60*fs #window of 3 minutes to find value for upper 90% of the difference between upper and lower envelope 
    for i in range(wdw, len(d)-wdw, wdw):
        peakenv[i:i+wdw]=np.percentile(d[i-wdw+1:i],90)

    # central apnea when difference between upper and lower envelope is smaller than 0.2 times the 90% percentile peakenv (difference of upper and lower envelope) of the 3 minutes before
    centr_detected[np.where(d<0.2*peakenv)[0]]=1;

    # HYPOPNEAS
    centr_hypo=np.empty(d.shape)
    centr_hypo[np.where(d<0.7*peakenv)[0]]=1;
    
    # %it is only a central apnea when it lasts 10 sec
    # %9 seconds is used now, because the envelope causes a more smooth decrease
    # %of the trace so have to give it a bit more margin
    centr_detected_final=np.zeros(centr_detected.shape);
    for i in np.arange(4.5*fs, len(centr_detected)-4.5*fs, fs):
        if sum(centr_detected[int(i-4.5*fs):int(i+4.5*fs)])>8.5*fs:
            centr_detected_final[int(i-4.5*fs):int(i+4.5*fs)] = 1


    centr_hypo_final=np.zeros(centr_hypo.shape);
    for i in np.arange(4.5*fs, len(centr_hypo)-4.5*fs, fs):
        if sum(centr_hypo[int(i-4.5*fs):int(i+4.5*fs)])>8.5*fs:
            centr_hypo_final[int(i-4.5*fs):int(i+4.5*fs)] = 1
          

    return centr_detected_final, centr_hypo_final

def generate_report(trace, centr_detected_final, centr_hypo_final, fs, save=False):
    
    state_switches = centr_detected_final[1:] - centr_detected_final[:-1]
    apnea_start = np.where(state_switches==1)[0]+1
    apnea_end = np.where(state_switches==-1)[0]+1
    state_switches = centr_hypo_final[1:] - centr_hypo_final[:-1]
    hypopnea_start = np.where(state_switches==1)[0]+1
    hypopnea_end = np.where(state_switches==-1)[0]+1
    
    hypopnea_report = pd.DataFrame(columns=['start_sample','end_sample','start_second','end_second','event'])
    hypopnea_report.start_sample = hypopnea_start
    hypopnea_report.end_sample = hypopnea_end
    hypopnea_report.start_second = np.round(hypopnea_start/fs,1)
    hypopnea_report.end_second = np.round(hypopnea_end/fs,1)
    hypopnea_report.event  = ['central hypopnea']*hypopnea_report.shape[0]
    
    apnea_report = pd.DataFrame(columns=['start_sample','end_sample','start_second','end_second','event'])
    apnea_report.start_sample = apnea_start
    apnea_report.end_sample = apnea_end
    apnea_report.start_second = np.round(apnea_start/fs,1)
    apnea_report.end_second = np.round(apnea_end/fs,1)
    apnea_report.event  = ['central apnea']*apnea_report.shape[0]
    
    report_events = pd.concat([hypopnea_report,apnea_report],ignore_index=True).sort_values(by='start_sample').reset_index()
    report_events.drop('index', axis=1,inplace=True)
    
    summary_report = pd.DataFrame([],columns=['signal duration (h)', 'detected central apnea events', 'detected central hypopnea events'])
    summary_report['signal duration (h)'] = [np.round(len(trace)/fs/3600,2)]
    summary_report['detected central apnea events'] = [len(apnea_start)]
    summary_report['detected central hypopnea events'] = [len(hypopnea_start)]
    
    full_report = pd.concat([report_events,summary_report],axis=1)
    if save:
        full_report.to_csv('report_central_resp_events.csv',index=False)

    return full_report
    

# replace all zeros by nan's

def plot_central_events(trace, centr_detected_final, centr_hypo_final, savepath = 'figure_central_resp_events'):

    central_events = np.zeros(centr_detected_final.shape)
    central_events[centr_hypo_final.astype('bool')] = 2
    central_events[centr_detected_final.astype('bool')] = 1
    central_events.astype('float')
    central_events[central_events==0] = float('nan')

    assert(central_events.shape[0] == trace.shape[0])


    # use seg_start_pos to convert to the nonoverlapping signal
    # y = ytrue                               # shape = (N, 4100)
    # yp = apnea_prediction                   # shape = (N, 4100)
    # yp_smooth = apnea_prediction_smooth     # shape = (N, 4100)

    # define the ids each row
    nrow = 10
    row_ids = np.array_split(np.arange(len(trace)), nrow)
    row_ids.reverse()

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    row_height = 10
    label_color = [None, 'g', 'c']

    # here, we get clip-normalized signal. we should not need to plot >3 STD values:
    trace[np.abs(trace > 3)] = np.nan

    for ri in range(nrow):
        # plot signal
        ax.plot(trace[row_ids[ri]]+ri*row_height, c='k', lw=0.2)


        y2 = central_events         

        yi=0

        # run over each plot row
        for ri in range(nrow):
            # plot tech annonation
    #         ax.axhline(ri*row_height-3*(2**yi), c=[0.5,0.5,0.5], ls='--', lw=0.2)  # gridline
            loc = 0

            # group all labels and plot them
            for i, j in groupby(y2[row_ids[ri]]):
                len_j = len(list(j))
                if not np.isnan(i) and label_color[int(i)] is not None:
                    # i is the value of the label
                    # list(j) is the list of labels with same value
                    ax.plot([loc, loc+len_j], [ri*row_height-3*(2**yi)]*2, c=label_color[int(i)], lw=1)
                loc += len_j

    # plot layout setup
    ax.set_xlim([0, max([len(x) for x in row_ids])])
    ax.axis('off')
    plt.tight_layout()
    # plt.title(test_info[si])
    # save the figure
    plt.savefig(savepath + '.pdf')
    # plt.savefig(savepath + '.png')


def compute_self_similarity(up, lo, fs):
    
    env_diff = up-lo
    similarity_array = np.zeros(env_diff.shape)

    env_diff[env_diff<0] = 0
    th = np.percentile(env_diff, 95)
    
    apneas2 = connect_apneas(th, env_diff, fs)
    apneas2[np.where(np.isnan(apneas2))] = 0
    apneas2[np.where(apneas2==-1000)] = 2

    env_diff = env_diff-np.percentile(env_diff,5)
    # upper and lower envelopes of envelope differences
    [tmp,lo2] = create_env(env_diff, fs=fs, r=25) # 25 8000/fs
    if lo2 is None:
        return similarity_array

    # subtract baseline
    env_diff = env_diff - lo2 
    [up2,tmp]=create_env(env_diff, fs=fs, r=25) # 25 8000/fs
    if up2 is None:
        return similarity_array

    env_diff = env_diff/(up2+0.00001)
    t = np.arange(len(env_diff))/fs
    
    [tePeaks,epeaks,teTroughs,etroughs,i0,i1] = get_peaks_troughs_4(t,abs(up-lo),2.51,10*fs) # get_peaks_troughs_4(t,abs(up-lo),3.51,5000)

    if tePeaks is None:
        return similarity_array

    # get similarities
    # get time of crescendo-diminuendo patterns

    # as done in Eline's script. peakTimes are start/end points of waves/segments of signals of interest.
    # peakTimes are simply mean points between two peaks found in envelope differnece.
    # for first peak, it's -15/+15 seconds of 
    # new implemention, with samples:
    pattern_boundary = np.zeros((len(i0),2), dtype=np.int64) # this is called peakTimes in Eline's code.
    similarity = []
    if pattern_boundary.shape[0] > 0:
        pattern_boundary[0,:] = [max(0,i0[0]-15*fs), i0[0]+15*fs]
        for i in range(len(tePeaks)-1):
            pattern_boundary[i+1,:] = [np.mean(i0[i:i+2]), np.mean(i0[i+1:i+3])]
        if len(tePeaks)-1 > 0:
            pattern_boundary[i+1,1] = min(pattern_boundary[i+1,1]+15*fs, len(env_diff))    


        # do convolution:

        for [pattern_start, pattern_end] in pattern_boundary:
            if pattern_start == 0: continue
            # get first pattern (wave) and normalize it:
            s1 = env_diff[pattern_start:pattern_end].copy()
            s1 = (s1-np.mean(s1))/np.std(s1+0.00001)
            len_pattern = pattern_end-pattern_start
            # get wave ahead (wave-1)|
            s0 = env_diff[max(1,pattern_start-len_pattern):pattern_start].copy()
            s0 = (s0-np.mean(s0))/np.std(s0+0.00001)

            # get wave behind (wave+1)
            s2 = env_diff[pattern_end: min(pattern_end+len_pattern,len(env_diff))]
            s2 = (s2-np.mean(s2))/np.std(s2+0.00001)
            # convolution
            if len(s2)==0:
        #             s2 = np.zeros(s1.shape)
                average_conv = max(np.convolve(s1,s0,'same'))/len_pattern
            else:
                conv1 = np.convolve(s1,s0,'same')
                conv2 = np.convolve(s1,s2,'same')
        #         average_conv = (conv1 + conv2)/2/len_pattern
                average_conv = (np.percentile(conv1,99)/len_pattern +  np.percentile(conv2,99)/len_pattern)/2

        #     similarity.append(max(average_conv))
            similarity.append(average_conv)
            similarity_array[pattern_start:pattern_end] = average_conv
            #     plt.figure()
            #     plt.plot(s1)
            #     plt.plot(s0)
            #     plt.plot(s2)
            #     plt.legend(['center wave', 'wave ahead', 'wave behind'])
            #     plt.title('current implementation, similarity: ' + str(average_conv))
        similarity = np.array(similarity)

    return similarity_array
    

def post_processing_detections(apneas, hypopneas, similarity_array, fs):

    # if apnea in hypopnea, keep only apnea.
    hypo_on = np.where(np.diff(hypopneas)==1)[0]
    hypo_off = np.where(np.diff(hypopneas)==-1)[0]
    for [hypo_on_tmp, hypo_off_tmp] in list(zip(hypo_on, hypo_off)):
        if any(apneas[hypo_on_tmp:hypo_off_tmp]==1):
            hypopneas[hypo_on_tmp:hypo_off_tmp+1] = 0

    # remove detections with similarity less than 0.5
    apneas[similarity_array<0.5] = 0
    hypopneas[similarity_array<0.5] = 0
    
    # need to last more than 9 seconds:
    hypo_on = np.where(np.diff(hypopneas)==1)[0]
    hypo_off = np.where(np.diff(hypopneas)==-1)[0]
    diff=[]
    for [hypo_on_tmp, hypo_off_tmp] in list(zip(hypo_on, hypo_off)):
        if hypo_off_tmp - hypo_on_tmp < 9*fs:
            hypopneas[hypo_on_tmp:hypo_off_tmp+1] = 0
    
    # if less than 5 seconds gap between two detections, connect them
    
    hypo_on = np.where(np.diff(hypopneas)==1)[0]
    hypo_off = np.where(np.diff(hypopneas)==-1)[0]
    apnea_on = np.where(np.diff(apneas)==1)[0]
    apnea_off = np.where(np.diff(apneas)==-1)[0]
        
    for [hypo_on_tmp, hypo_off_previous] in list(zip(hypo_on[1:], hypo_off[:-1])):
        if hypo_on_tmp - hypo_off_previous < 5*fs:
            hypopneas[hypo_off_previous:hypo_on_tmp+1] = 1

    for [apnea_on_tmp, apnea_off_previous] in list(zip(apnea_on[1:], apnea_off[:-1])):
        if apnea_on_tmp - apnea_off_previous < 5*fs:
            apneas[apnea_off_previous:apnea_on_tmp+1] = 1

    return apneas, hypopneas


def compute_stability(up: np.ndarray,
                    lo: np.ndarray,
                    fs: float,
                    window_length_min: float = 2.0,
                    overlap: float = 0.5
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a breathing stability index based on the sum of mean absolute gradients
    of the upper and lower envelopes over sliding windows.

    Parameters
    ----------
    up : np.ndarray
        Upper‐envelope of the breathing signal (1D array).
    lo : np.ndarray
        Lower‐envelope of the breathing signal (1D array), same shape as `up`.
    fs : float
        Sampling frequency in Hz.
    window_length_min : float, optional
        Window length in minutes (default 2.0).
    overlap : float, optional
        Fractional overlap between windows (0 ≤ overlap < 1, default 0.5 for 50%).

    Returns
    -------
    stability_full : np.ndarray
        Full‐length array (same shape as `up`) with the computed stability
        assigned at each window’s center sample, NaN elsewhere.
    centers_s : np.ndarray
        1D array of center‐times (in seconds) for each window.
    stability_vals : np.ndarray
        1D array of stability index values (gradient sums) for each window.
    """
    # Convert window and step to samples
    window_len = int(window_length_min * 60 * fs)
    if window_len < 2:
        raise ValueError("window_length_min is too small for the given fs.")
    overlap = min(overlap, 0.99)
    step = int(window_len * (1 - overlap))
    if step < 1:
        raise ValueError("overlap must be < 1 to make positive step size.")

    n = up.size
    stability_full = np.full(n, np.nan, dtype=float)

    for start in range(0, n - window_len + 1, step):
        end = start + window_len
        
        # 1) raw gradient sum
        grad_up = np.mean(np.abs(np.diff(up[start:end])))
        grad_lo = np.mean(np.abs(np.diff(lo[start:end])))
        s_raw = grad_up + grad_lo
        
        # 2) window-local amplitude baseline
        mean_env = np.nanmean((np.abs(up[start:end]) + np.abs(lo[start:end])) / 2)
        s_norm = s_raw / (mean_env + 1e-4)
        # scale to %:
        s_norm = s_norm * 100
        
        # 3) assign at center
        center = start + window_len // 2
        stability_full[center] = s_norm

    # 4) fill forward/backward
    stability_full = pd.Series(stability_full).fillna(method='ffill')\
                                        .fillna(method='bfill').values
    
    return stability_full


def stability_summary_features(stability_full: np.ndarray, fs: float) -> tuple[float, float]:
    """
    Compute summary metrics for the stability index.
    1. Median stability
    2. Area under the curve (AUC) normalized by time (in seconds).
    3. 10-bin Histogram of stability values.
    4. 5-bin Histogram of stability values.
    5. list of quantiles (0.01, 0.025, 0.05, 0.075, 0.1, 0.175, 0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75, 0.825, 0.9, 0.95, 0.975, 0.99).
    """
    
    qs = [1, 2.5, 5, 7.5, 10, 17.5, 25, 35, 45, 50, 55, 65, 75, 82.5, 90, 95, 97.5, 99]

    if len(stability_full) == 0: # empty array
        return {
            'median_stability': np.nan,
            'stability_auc': np.nan,
            'hist_12_bin_0': np.nan, 'hist_12_bin_1': np.nan, 'hist_12_bin_2': np.nan, 
            'hist_12_bin_3': np.nan, 'hist_12_bin_4': np.nan, 'hist_12_bin_5': np.nan, 
            'hist_12_bin_6': np.nan, 'hist_12_bin_7': np.nan, 'hist_12_bin_8': np.nan, 
            'hist_12_bin_9': np.nan, 'hist_12_bin_10': np.nan, 'hist_12_bin_11': np.nan,
            'hist_5_bin_0': np.nan, 'hist_5_bin_1': np.nan, 'hist_5_bin_2': np.nan,
            'hist_5_bin_3': np.nan, 'hist_5_bin_4': np.nan,
            **{f'quantile_{q}': np.nan for q in qs}
        }
        
    assert stability_full.ndim == 1, "stability_full must be 1D array"

    # clip extreme outliers, defined by deviation from IQR:
    q1 = np.nanpercentile(stability_full, 25)
    q3 = np.nanpercentile(stability_full, 75)
    iqr = q3 - q1
    lower_bound = q1 - 2 * iqr
    upper_bound = q3 + 2 * iqr
    stability_clipped = np.clip(stability_full, lower_bound, upper_bound)
    
    stability_summary_metrics = {}
    
    # 1. median stability:
    median_stability = np.nanmedian(stability_clipped)
    stability_summary_metrics['median_stability'] = median_stability
    
    # 2. area under the curve normalized by time (in seconds):
    stability_auc = np.nansum(stability_clipped) / len(stability_clipped) * fs
    stability_summary_metrics['stability_auc'] = stability_auc
        
    # 3. 12-bin histogram of stability values:
    bin_edges_12 = np.array([0. , 0.25, 0.5, 0.75, 1., 1.25, 1.5, 2. , 2.5, 3. , 3.5, 4. , 8. ])
    hist_12, bin_edges_12 = np.histogram(stability_full, bins=bin_edges_12)
    hist_12 = hist_12 / np.sum(hist_12) # normalize histogram
    for i in range(len(hist_12)):
        stability_summary_metrics[f'hist_12_bin_{i}'] = hist_12[i]

    # 4. 5-bin histogram of stability values:
    bin_edges_5 = np.array([0. , 0.35, 0.70, 1.40 , 2.80, 8.00 ])
    hist_5, bin_edges_5 = np.histogram(stability_full, bins=bin_edges_5)
    hist_5 = hist_5 / np.sum(hist_5) # normalize histogram
    for i in range(len(hist_5)):
        stability_summary_metrics[f'hist_5_bin_{i}'] = hist_5[i]

    # 5. list of quantiles (0.01, 0.025, 0.05, 0.075, 0.1, 0.175, 0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75, 0.825, 0.9, 0.95, 0.975, 0.99):
    quantiles = np.nanpercentile(stability_full, qs)
    for i, q in enumerate(qs):
        stability_summary_metrics[f'quantile_{q}'] = quantiles[i]
    
    return stability_summary_metrics


def stability_episodes_features(stability_full: np.ndarray,
                                    fs: float,
                                    window_length_min: float = 5.0,
                                    low_thr: float = 0.50,
                                    high_thr: float = 1.5) -> dict:
    """
    Episode‐based features with dynamic window‐counts:
      - within each contiguous episode, how many full 5 min windows fit
    """

    # Helper to get contiguous episodes and durations (sec)
    def get_episodes(mask):
        padded = np.concatenate(([False], mask, [False]))
        d = np.diff(padded.astype(int))
        starts = np.where(d == 1)[0]
        ends   = np.where(d == -1)[0]
        durs = (ends - starts) / fs  # in seconds
        return list(zip(starts, ends, durs))

    if len(stability_full) == 0:
        return {
            'n_5min_windows_stable': np.nan,
            'n_5min_windows_unstable': np.nan,
        }
    # Build masks
    stable_mask   = stability_full < low_thr
    unstable_mask = stability_full > high_thr

    stable_eps = get_episodes(stable_mask)
    unstable_eps = get_episodes(unstable_mask)

    # DYNAMIC window‐counts within each episode
    def dynamic_window_count(eps):
        # sum floor(duration / window_length_min)
        return sum(int(d // (window_length_min * 60)) for (_, _, d) in eps)

    cnt_stable_dyn   = dynamic_window_count(stable_eps)
    cnt_unstable_dyn = dynamic_window_count(unstable_eps)
    if pd.isna(cnt_stable_dyn):
        cnt_stable_dyn = 0
    if pd.isna(cnt_unstable_dyn):
        cnt_unstable_dyn = 0

    return {
        'n_5min_windows_stable': cnt_stable_dyn,
        'n_5min_windows_unstable': cnt_unstable_dyn,
    }
    

def _get_row_indices(n_samples: int, n_rows: int):
    """
    Split a 1D index array [0…n_samples) into `n_rows` roughly equal chunks,
    then reverse so that chunk[0] is the bottom row when plotting.
    """
    row_ids = np.array_split(np.arange(n_samples), n_rows)
    row_ids.reverse()
    return row_ids


def plot_stability_index(trace: np.ndarray,
                         stability_index: np.ndarray,
                         fs: float,
                         n_rows: int = 10,
                         row_height: float = 10,
                         cmap_name: str = 'coolwarm',
                         bg_height_frac: float = 0.10,
                         vmax = 3,
                         save_name: str = 'stability_index',
                         title=None):
    """
    Plot `trace` in stacked rows, color-coding a fraction of each row's vertical space
    with a divergent colormap representing `stability_index` (normalized between 0 and 1).

    Parameters
    ----------
    trace : np.ndarray
        1D breathing signal (unitless or z-scored) of length N.
    stability_index : np.ndarray
        1D array of same length N with values between 0 and 1.
    fs : float
        Sampling frequency (Hz). Unused here but kept for API symmetry.
    n_rows : int
        Number of stacked rows.
    row_height : float
        Vertical spacing between rows.
    cmap_name : str
        Name of Matplotlib colormap (divergent).
    bg_height_frac : float
        Fraction [0,1] of each row's height to fill with stability background (from top) [1: full background, default: 0.10 top line above signal]
    save_name : str
        Base filename to save PDF/PNG (no extension).
    """
    from matplotlib.colors import Normalize

    assert trace.shape == stability_index.shape, "trace & stability must align"
    assert 0 < bg_height_frac <= 1.0, "bg_height_frac must be in (0,1]"

    # 1) Row splits
    row_ids = _get_row_indices(len(trace), n_rows)
    max_len = max(len(r) for r in row_ids)

    # 2) Build 2D array of stability (NaN-padded)
    bg = np.full((n_rows, max_len), np.nan, dtype=float)
    for i, idx in enumerate(row_ids):
        bg[i, :len(idx)] = stability_index[idx]

    # 3) Set up figure
    fig, ax = plt.subplots(figsize=(12, 8))
    norm = Normalize(vmin=0.001, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # 4) Draw background per row (or full) with controlled height
    masked_bg = np.ma.masked_invalid(bg)
    if bg_height_frac >= 1.0:
        # full-row background
        extent = [0, max_len, -row_height/2, row_height * n_rows - row_height/2]
        ax.imshow(masked_bg,
                    aspect='auto', origin='lower', extent=extent,
                    cmap=cmap, norm=norm, interpolation='none')
    else:
        # draw each row's background only in top fraction
        for row_i in range(n_rows):
            y_off = (row_i - 0.55) * row_height
            y0 = y_off + (1 - bg_height_frac) * row_height
            y1 = y_off + row_height
            row_slice = masked_bg[row_i:row_i+1, :]
            ax.imshow(row_slice,
                        aspect='auto', origin='lower',
                        extent=[0, max_len, y0, y1],
                        cmap=cmap, norm=norm, interpolation='none')

    # 5) Over-plot signal
    for row_i, idx in enumerate(row_ids):
        y_off = row_i * row_height
        x = np.arange(len(idx))
        ax.plot(x, trace[idx] * 1.25 + y_off, c='k', lw=0.2)

    if title is not None:
        y = 1
        if 1:
            ax.text(0, 1.005, title, fontsize=10, ha='left', va='bottom', transform=ax.transAxes)
            title = "Sample Stability Index for Breathing Signal - Full Night\n"
            y = 1.01
        ax.set_title(title, fontsize=12, y=y)
        
    # 6) Style & save
    ax.axis('off')
    if 1:
        cbar_ax = fig.add_axes([0.93, 0.70, 0.015, 0.25])  # [left, bottom, width, height] in figure fraction
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
            cax=cbar_ax,
            label='Stability Index'
        )
        plt.subplots_adjust(left=0.01, right=0.9, bottom=0, top=0.93, hspace=0)
    else:
        plt.tight_layout()
    plt.savefig(save_name + '.pdf')
    # plt.savefig(save_name + '.png')
    plt.close(fig)

    
def process_file_ss_bsi(file_path, notch_freq=60, bandpass_freq=[0, 10], n_jobs=1, new_fs=10,
                                 stability_wl=5, stability_overlap=0.90, filetitle=None, no_plot=False,
                                 belt_mode='raw_mean'):
    """Process a single file for self-similarity (SS) and breathing stability (BSI) analysis

    Parameters
    ----------
    file_path : str
        Path to the input file (EDF or H5).
    notch_freq : float
        Frequency for notch filtering (default: 60 Hz).
    bandpass_freq : list
        Frequency range for bandpass filtering (default: [0, 10] Hz).
    n_jobs : int
        Number of parallel jobs for filtering (default: 1).
    new_fs : float
        New sampling frequency after resampling (default: 10 Hz).
    stability_wl : float
        Window length in minutes for stability index (default: 5 min).
    stability_overlap : float
        Fractional overlap between windows for stability index (default: 0.90).
    belt_mode : str
        Belt combination mode for abdominal and chest effort channels.
        `raw_mean` preserves the legacy trace average, `robust_mean` robust-normalizes
        each belt before averaging, and `summary_mean` computes BSI per belt first and
        averages the nightly/stage summaries across belts.
    """

    import time
    starttime = time.time()

    # savedir = '/media/wolfgang/badgerhd/breathing_stability'
    savedir = '.'
    # self_sim_results_dir = os.path.join(savedir, 'central_resp_events')
    stability_results_dir = os.path.join(savedir, 'stability_index')

    # if result folder does not exist, create
    # if not os.path.exists(self_sim_results_dir):
        # os.mkdir(self_sim_results_dir)
    if not os.path.exists(stability_results_dir):
        os.mkdir(stability_results_dir)

    try:
    # if 1:
        fileid = os.path.basename(file_path).replace('.edf', '').replace('.h5', '')

        if belt_mode not in VALID_BELT_MODES:
            raise ValueError(f'Invalid belt_mode={belt_mode!r}. Expected one of {VALID_BELT_MODES}.')

        if file_path.endswith('.edf'):

            respiratory_effort_channels = ['Abdomen', 'Abdominal', 'Chest', 'ABDOMEN', 'CHEST', 'ABDOMINAL', 'Effort THO']
            edf = mne.io.read_raw_edf(file_path, stim_channel=None, preload=False, verbose=False)
            edf_channels = edf.info['ch_names']
            fs = int(edf.info['sfreq'])

            try:
                start_time = datetime.datetime.fromtimestamp(edf.info['meas_date'][0]) + timedelta(seconds=time.altzone)
            except:
                start_time = edf.info['meas_date'] + timedelta(seconds=time.altzone)

            # find respiratory effort channels

            if not sum(np.isin(respiratory_effort_channels, edf_channels)) > 0:
                print(f'{file_path}:')
                print(f'Expected Channel Name not found. Try flexible search for anything with "effort", "chest", "tho", or "abd" in it.')
                respiratory_effort_channels = [x for x in edf_channels if any([y in x.lower() for y in ['effort', 'chest', 'tho', 'abd']])]
                if len(respiratory_effort_channels) > 0:
                    print(f'Success. Channel used: {respiratory_effort_channels}')
            if not sum(np.isin(respiratory_effort_channels, edf_channels)) > 0:
                print(f'Code cannot be performed: No effort belt channel found in the EDF file. \nThe file contains:{edf_channels}.')
                return None
            
            respiratory_effort_channels = [x for x in respiratory_effort_channels if x in edf_channels]

            signals = edf.get_data(picks=respiratory_effort_channels)  # signals.shape=(#channel, T)
            channel_names = respiratory_effort_channels

            stage = None
            
        elif file_path.endswith('.h5'):
            # 'prepared data' format file:
            signals_df, annotations, params = load_prepared_data(file_path)
            channel_names = [x for x in signals_df.columns if infer_effort_channel_role(x) in ['abd', 'chest', 'effort']]
            if len(channel_names) == 0:
                raise ValueError('Code cannot be performed: No effort belt channel found in the H5 file.')
            signals = signals_df[channel_names].values.T  # shape: (n_channels, n_samples)
            fs = params['fs']
            stage = annotations['stage'].values.flatten()
            fs_ratio = fs/new_fs
            assert fs_ratio.is_integer(), f"fs_ratio {fs_ratio} is not an integer. Cannot resample stage to {new_fs} Hz."
            stage = stage[::int(fs_ratio)]

        signals, channel_names = select_effort_signals(signals, channel_names)
        signals = preprocess_effort_signals(
            signals,
            fs,
            notch_freq=notch_freq,
            bandpass_freq=bandpass_freq,
            n_jobs=n_jobs,
            new_fs=new_fs,
        )

        if stage is not None:
            assert signals.shape[1] == len(stage), f"Length of resp trace ({signals.shape[1]}) and stage ({len(stage)}) do not match after resampling."
            
        fs = new_fs

        trace, component_traces = build_belt_mode_traces(signals, belt_mode=belt_mode)
        reference_outputs = compute_trace_outputs(trace, fs, stage, stability_wl, stability_overlap)

        apneas = reference_outputs['apneas']
        hypopneas = reference_outputs['hypopneas']
        trace = reference_outputs['trace']

        if belt_mode == 'summary_mean' and len(component_traces) > 1:
            component_outputs = [
                compute_trace_outputs(component_trace, fs, stage, stability_wl, stability_overlap)
                for component_trace in component_traces
            ]
            stability_array = np.mean(np.vstack([x['stability_array'] for x in component_outputs]), axis=0)
            stability_summary = average_metric_dicts([x['stability_summary'] for x in component_outputs])
            stability_episodes = average_metric_dicts([x['stability_episodes'] for x in component_outputs])
        else:
            stability_array = reference_outputs['stability_array']
            stability_summary = reference_outputs['stability_summary']
            stability_episodes = reference_outputs['stability_episodes']

        if 'SLEEP_stability_auc' in stability_summary:
            stability_auc = stability_summary['SLEEP_stability_auc']
            stability_median = stability_summary['SLEEP_median_stability']
            stability_q25 = stability_summary['SLEEP_quantile_25']
            stability_q75 = stability_summary['SLEEP_quantile_75']
        else:
            stability_auc = stability_summary['ALL_stability_auc']
            stability_median = stability_summary['ALL_median_stability']
            stability_q25 = stability_summary['ALL_quantile_25']
            stability_q75 = stability_summary['ALL_quantile_75']
            
        # saves report
        if not no_plot:
            # savepath_report = os.path.join(self_sim_results_dir, fileid + '_report.csv')
            report = generate_report(trace, apneas, hypopneas, fs)
            # report.to_csv(savepath_report, index=False)

        # png and pdf plots of central events:
        # savepath_plot = os.path.join(self_sim_results_dir, fileid + '_figure')
        # if not no_plot:
        #     plot_central_events(trace, apneas, hypopneas, savepath_plot)

        ### Save stability results (plots and summary):
        # Plot:
        # savepath_stability = os.path.join(stability_results_dir, fileid + '_stability_index')
        if not os.path.exists(stability_results_dir):
            os.mkdir(stability_results_dir)
        savepath_stability = os.path.join(stability_results_dir, fileid + '_stability_index')

        if stability_wl == 5: vmax = 3
        elif stability_wl == 2: vmax = 3
        elif stability_wl == 1: vmax = 3
        
        if 0: 
            filetitle = filetitle + f' ||| stability index: {stability_auc:.1f}'
        if 1:
            hours_recording = np.round(len(trace) / fs / 3600, 1)
            title_appendix = f'Recording time: {hours_recording} h, '
            if stage is not None:
                hours_sleep = np.round(np.isin(stage, [1, 2, 3, 4]).sum() / fs / 3600, 1)
                title_appendix += f'sleep time: {hours_sleep} h, '
                
            filetitle = title_appendix + filetitle \
                + f', median (IQR) stability index: {stability_median:.1f} ({stability_q25:.1f}-{stability_q75:.1f})'

        if not no_plot:
            plot_stability_index(trace, stability_array, fs, save_name=savepath_stability, vmax=vmax,
                                title=filetitle)
        
    
        # Save stability summary to summary csv file:
        new_row = {
            'file': fileid,
            'window_length': stability_wl,
            'overlap': stability_overlap,
            'belt_mode': belt_mode,
            'belt_channels_used': '|'.join(channel_names),
            'n_belts_used': len(channel_names),
        }
        new_row.update(stability_summary)
        new_row.update(stability_episodes)
            
        # if 0: # Old way, read the csv file first, then append the new row.
        #     file_summary_stability = os.path.join(stability_results_dir, 'summary_stability.csv')
        #     if not os.path.exists(file_summary_stability):
        #         summary_stability = pd.DataFrame(columns=['file', 'window_length', 'overlap'] + list(stability_summary.keys()) + list(stability_episodes.keys()))
        #     else:
        #         summary_stability = pd.read_csv(file_summary_stability)
                
        #     summary_stability = pd.concat([summary_stability, pd.DataFrame([new_row])], ignore_index=True)
        #     summary_stability.drop_duplicates(inplace=True, keep='first')
        #     # all features that are not integers, can be rounded to 3 decimal places
        #     for col in summary_stability.columns:
        #         if summary_stability[col].dtype == 'float64':
        #             summary_stability[col] = np.round(summary_stability[col], 3)
        #     summary_stability.to_csv(file_summary_stability, index=False)
            
        # Append to csv file, no need to read it first (unstable for parallel processing)
        df_new_entry = pd.DataFrame([new_row])
        for col in df_new_entry.columns:
            if df_new_entry[col].dtype == 'float64':
                df_new_entry[col] = np.round(df_new_entry[col], 3)
        
        # Only save most important columns for main use cases
        cols_to_save = ['file'] + [x for x in df_new_entry.columns if any([y in x for y in ['median_stability', 'quantile', 'n_5min_windows']])]
        # only 'ALL_' and remove 'ALL_' naming
        if 0:
            cols_to_save = ['file'] + [x for x in cols_to_save if x.startswith('ALL_')]
            df_new_entry = df_new_entry[cols_to_save]
            df_new_entry.columns = [x.replace('ALL_', '') for x in df_new_entry.columns]
            
        file_summary_stability = os.path.join(stability_results_dir, 'summary_stability.csv')
        lock_path = file_summary_stability + '.lock'
        with FileLock(lock_path):
            header = not os.path.exists(file_summary_stability)
            df_new_entry.to_csv(file_summary_stability, mode='a', header=header, index=False)
            
        # HLG summary csv
        # try:
        if 0:
        # if 1:
            path_hlg_summary = os.path.join('summary_HLG_percentage.csv')
            if os.path.exists(path_hlg_summary):
                summary_hlg = pd.read_csv(path_hlg_summary)
            else:
                summary_hlg = pd.DataFrame(columns = ['file', 'HLG_percentage (%)'])

            summary_hlg_tmp = pd.DataFrame(columns = ['file', 'HLG_percentage (%)'])
            summary_hlg_tmp.loc[0, 'file'] = os.path.basename(file_path)

            hlg_perc = (sum(apneas) + sum(hypopneas)) / (len(trace) * 0.9) # small scaler because we consider HLG area also between two annotated apneas.

            summary_hlg_tmp.loc[0, 'HLG_percentage (%)'] = np.round(100*hlg_perc, 1)

            summary_hlg = pd.concat([summary_hlg, summary_hlg_tmp], axis=0, ignore_index=True)
            summary_hlg.drop_duplicates(inplace=True)

            summary_hlg.to_csv(path_hlg_summary, index=False)


        # if 1: save the signals for a plot later: original resp trace, envelopes, stability index

        # except Exception as e:
        #     print('Error in summary file computation/creation:')
        #     print(e)
        #     print('Continue with next file.')
        #     return None
            
    #     # print('runtime (sec):')
    #     # print(time.time() - starttime)

    except Exception as e:
        print(f'Error for {file_path}:')
        print(e)
        print('Continue with next file.')
        return None

    
def main():
    
    p = argparse.ArgumentParser(
        description="Compute self‑similarity & stability for EDF/H5 files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "-i","--input-path",
        type=Path,
        default=None,
        help="either a directory with edf/h5 files or a CSV with a 'file_path' column; if omitted, scan cwd for .edf/.h5"
    )
    p.add_argument(
        "-w","--stability-window-length",
        type=float,
        default=2.0,
        help="window length in minutes"
    )
    p.add_argument(
        "-o","--stability-overlap",
        type=float,
        default=0.9,
        help="fractional overlap (0–1)"
    )
    p.add_argument(
        "-np","--noplot",
        action='store_true',
        help="do not plot results"
    )
    p.add_argument(
        "--belt-mode",
        choices=VALID_BELT_MODES,
        default='raw_mean',
        help="how abdominal and chest effort signals are combined before BSI summarization"
    )
    args = p.parse_args()
    
    input_path = args.input_path
    stability_wl = args.stability_window_length
    stability_overlap = args.stability_overlap
    no_plot = args.noplot
    belt_mode = args.belt_mode
    is_inputtable = False
    
    if 1:
        # Randomly switch on no_plot in 1% of the cases. Create just a few sample plot when running on many files.
        if np.random.rand() < 0.01:
            no_plot = False
        else:
            no_plot = True
            
    if input_path is None:
        # if no input path, use current folder and process all edf and h5 files in it.
        files_folder = '.'
        # get all edf and h5 files paths in this folder
        file_paths = glob.glob(os.path.join(files_folder, '*.edf')) + glob.glob(os.path.join(files_folder, '*.h5'))
    else:
        input_path = str(input_path)
        if os.path.isdir(input_path):
            # if input path is a folder, process all edf and h5 files in it.
            files_folder = input_path
            file_paths = glob.glob(os.path.join(files_folder, '*.edf')) + glob.glob(os.path.join(files_folder, '*.h5'))
        elif os.path.isfile(input_path):
            is_inputtable = True
            # if input table, read it and get the file paths
            input_table = pd.read_csv(input_path)
            assert 'file_path' in input_table.columns, "Input table must contain a 'file_path' column."
            file_paths = input_table['file_path'].tolist()
            
    # process each file
    # for file_path in tqdm(file_paths, desc="Processing files"):
    for file_path in file_paths:
        filetitle = ''
        if is_inputtable is True:
            if all([x in input_table.columns for x in ['f_ahi', 'f_oai', 'f_cai']]):
                row = input_table[input_table['file_path'] == file_path]
                f_ahi = int(np.round(row['f_ahi']))
                f_oai = int(np.round(row['f_oai']))
                f_cai = int(np.round(row['f_cai']))
                filetitle = f'AHI: {f_ahi}, OAI: {f_oai}, CAI: {f_cai}'

        process_file_ss_bsi(file_path, stability_wl=stability_wl, stability_overlap=stability_overlap,
                                        filetitle=filetitle, no_plot=no_plot, belt_mode=belt_mode)
        
    # print('All files processed.')
    
if __name__ == "__main__":
    main()


