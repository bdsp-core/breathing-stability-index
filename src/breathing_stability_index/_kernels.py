"""Numerical kernels extracted verbatim from the revised development code.

See paper/provenance/source_manifest.json and docs/scientific-contract.md.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, detrend
from scipy import interpolate
from statsmodels.robust.scale import mad
from mne.filter import filter_data, notch_filter, resample
VALID_BELT_MODES = ('raw_mean', 'robust_mean', 'summary_mean')


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
