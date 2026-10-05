import argparse
import os
import datetime
from pathlib import Path
from typing import Dict, List, Optional


def _configure_runtime_environment() -> None:
    """Prefer stable, headless defaults for shared batch environments."""
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_USE_NUMBA", "false")
    os.environ.setdefault("MPLBACKEND", "Agg")


_configure_runtime_environment()

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import groupby
from scipy.signal import savgol_filter, convolve, find_peaks
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
from filelock import FileLock

import warnings
warnings.filterwarnings("ignore")

verbose = False  # deactivate all prints by default


def _import_mne():
    """Import MNE lazily so H5-only shards avoid EDF-only import/runtime issues."""
    import mne

    return mne


def vprint(*args, **kwargs):
    """Verbose-aware print helper."""
    if verbose:
        print(*args, **kwargs)


import contextlib

@contextlib.contextmanager
def suppress_stderr():
    """Temporarily suppress C-level stderr (e.g. MKL DGELSD errors)."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            
class EnvelopeComputationFailed(Exception):
    """Raised when ventilation envelope/smoothing cannot be computed reliably."""
    pass


ERROR_LOG_DEFAULT = 'self_similarity_errors.log'


def _normalize_dirname(name: str) -> str:
    return name.lower().replace('_', ' ').strip()


def _collect_data_files(directory: Path) -> List[Path]:
    """
    Collect EDF/H5 files from `directory` and an optional 'edf files' sub-folder (case-insensitive).
    """
    patterns = ('*.edf', '*.EDF', '*.h5', '*.H5')
    files: List[Path] = []

    for pattern in patterns:
        files.extend(directory.glob(pattern))

    try:
        for child in directory.iterdir():
            if child.is_dir() and _normalize_dirname(child.name) == 'edf files':
                for pattern in patterns:
                    files.extend(child.glob(pattern))
    except FileNotFoundError:
        pass

    unique: List[Path] = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return sorted(unique)


def _resolve_error_log_path(path: Optional[Path]) -> Path:
    if path is None:
        base = Path.cwd() / ERROR_LOG_DEFAULT
    else:
        base = Path(path).expanduser()
        if base.exists() and base.is_dir():
            base = base / ERROR_LOG_DEFAULT
    return base.resolve()


def _write_error_log(log_path: Path, errors: List[tuple[Path, str]]) -> None:
    if not errors:
        if log_path.exists():
            log_path.unlink()
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    with log_path.open('w', encoding='utf-8') as fh:
        fh.write(f'# Self-similarity processing errors ({timestamp})\n')
        for file_path, message in errors:
            fh.write(f'{file_path}\t{message}\n')


def _prepare_self_similarity_paths(base_dir: Path) -> tuple[Path, Path, Path]:
    """
    Ensure output directories exist and return (summary_csv, plots_dir, reports_dir).
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = base_dir / 'plots_self_similarity'
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = base_dir / 'self_similarity_reports'
    # reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = base_dir / 'results_self_similarity.csv'
    return summary_path, plots_dir, reports_dir


def _load_hypoxic_burden_map(results_dir: Path) -> dict[str, float]:
    summary_path = results_dir / 'results_hypoxic_burden.csv'
    if not summary_path.exists():
        return {}
    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return {}
    if 'hb_sleep' not in df.columns:
        return {}
    hb_map: dict[str, float] = {}
    file_cols = [c for c in ['file', 'fileid'] if c in df.columns]
    if not file_cols:
        return {}
    for _, row in df.iterrows():
        hb_value = row['hb_sleep']
        if pd.isna(hb_value):
            continue
        for col in file_cols:
            key = str(row[col]).strip()
            if key and key.lower() != 'nan':
                hb_map[key] = float(hb_value)
                hb_map[Path(key).stem] = float(hb_value)
    return hb_map

def show_credits():
    # header
    print("\n\n*** Notice: Ownership and Patent Information ***")
    credit = 'This program, including its underlying algorithms and methodologies '

    # names
    credit += 'is owned and patented by: T. Nassi, E. Oppersma, M.B. Westover, and R.J Thomas. '
    # credit += 'Beth Israel Deaconess Medical Center and the University of Twente. '

    # warning
    credit += 'Any unauthorized use, reproduction, or distribution of this program '
    credit += 'or its components is strictly prohibited and may result in legal action. '
    
    # info
    credit += 'For licensing information, please contact the corresponding authors.\n\n'

    print(credit)
    
def load_prepared_data(file_path: str, signals_to_load: Optional[List[str]] = None,
                       annotations_to_load: Optional[List[str]] = None) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Load respiratory prepared-data (.h5) files and return signals, annotations, and metadata.

    Parameters
    ----------
    file_path : str
        Path to the prepared-data file.
    signals_to_load : list[str] | None
        Names of signal datasets to load. If ``None``, load all available signals.
    annotations_to_load : list[str] | None
        Names of annotation datasets to load. If ``None``, load all available annotations.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict]
        DataFrames containing the requested signals and annotations, and a dict with metadata.
    """

    signals = pd.DataFrame([])
    annotations = pd.DataFrame([])

    with h5py.File(file_path, "r") as f:
        signal_group = f['signals']
        signals_contained = list(signal_group.keys())

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
                rename_eeg_to_c4m1 = True
                signals_to_load = [x.replace('c4-m1', 'eeg') for x in signals_to_load]
            dataset_names = signals_to_load

        for dataset_name in dataset_names:
            dataset = signal_group[dataset_name][:]
            assert dataset.shape[1] == 1, "Only one-dimensional datasets expected"
            signals[dataset_name] = dataset.flatten()
        if rename_eeg_to_c4m1:
            signals.rename(columns={'eeg': 'c4-m1'}, inplace=True)

        if 'annotations' in f.keys():
            annotation_group = f['annotations']
            if annotations_to_load is None:
                dataset_names = list(annotation_group.keys())
            else:
                dataset_names = annotations_to_load
            for dataset_name in dataset_names:
                dataset = annotation_group[dataset_name][:]
                assert dataset.shape[1] == 1, "Only one-dimensional datasets expected"
                annotations[dataset_name] = dataset.flatten()

        params = {
            'fs': f.attrs['sampling_rate'],
            'unit_voltage': f.attrs['unit_voltage'],
        }

        signals = signals.astype(float)
        annotations = annotations.astype(float)

    return signals, annotations, params


def extract_breathing_channels(signals: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """Select respiratory channels from a prepared-data signals table."""

    search_channels = [
        ('sum effort', 'effort'),
        ('effort', 'effort'),
        ('abd', 'abd'),
        ('abdom', 'abd'),
        ('thorax', 'chest'),
        ('chest', 'chest'),
        ('ptaf', 'ptaf'),
        ('npt', 'ptaf'),
        ('therm', 'airflow'),
        ('airflow', 'airflow'),
        ('cflow', 'cflow'),
        ('flow', 'airflow'),
    ]

    canonical_to_original: Dict[str, str] = {}
    canonical_order: List[str] = []

    for col in signals.columns:
        lower_col = col.lower()
        for pattern, canonical in search_channels:
            if pattern in lower_col and canonical not in canonical_to_original:
                canonical_to_original[canonical] = col
                canonical_order.append(canonical)
                break

    if not canonical_order:
        raise ValueError('No respiratory channels found in prepared data file.')

    data = pd.DataFrame({key: signals[canonical_to_original[key]].astype(float)
                         for key in canonical_order})
    channel_labels = [canonical_to_original[key] for key in canonical_order]

    return data, channel_labels

# Loading Functions
def multiple_channel_search(edf):
    # set respiratory channels
    edf_channels = edf.info['ch_names']
    search_channels = {
        'npt': 'ptaf',
        'therm': 'airflow',
        'flow': 'airflow',
        'abd': 'abd',
        'chest': 'chest',
        'thorax': 'chest',
        'sum effort': 'effort',
        'effort': 'effort',
        
    }
    
    # run over all search channels
    found_channels, columns = [], []
    for ch in search_channels.keys():
        # skip if already found channel
        if search_channels[ch] in columns: continue
        # run over all edf channels
        for c in edf_channels:
            if ch in c.lower():
                found_channels.append(c)
                columns.append(search_channels[ch])
                break

    # pick effort trace
    signals = edf.get_data(picks=found_channels) 
    data = pd.DataFrame(np.squeeze(signals).T, columns=columns)

    return data, found_channels

def original_effort_search(edf, edf_path: Path | str):
    # find respiratory effort channels
    edf_channels = edf.info['ch_names']
    respiratory_effort_channels = ['Abdomen', 'Abdominal', 'Chest', 'ABDOMEN', 'CHEST', 'ABDOMINAL', 'Effort THO']
    if not np.isin(respiratory_effort_channels, edf_channels).any():

        vprint(f'{edf_path}:')
        vprint('Expected Channel Name not found. Try flexible search for anything with "effort", '
               '"chest", "tho", or "abd" in it.')
        respiratory_effort_channels = [
            x for x in edf_channels if any(y in x.lower() for y in ['effort', 'chest', 'tho', 'abd'])
        ]
        if respiratory_effort_channels:
            vprint(f'Success. Channel used: {respiratory_effort_channels}')
    if not np.isin(respiratory_effort_channels, edf_channels).any():
        message = ('Code cannot be performed: No effort belt channel found in the EDF file. '
                   f'The file contains: {edf_channels}.')
        vprint(message)
        raise ValueError(message)

    # pick effort trace
    respiratory_effort_channels = [x for x in respiratory_effort_channels if x in edf_channels]
    respiratory_effort_channels.sort()
    channel_name = 'abd' if 'abd' in respiratory_effort_channels[0].lower() else 'chest'
    signals = edf.get_data(picks=respiratory_effort_channels[0])  # signals.shape=(#channel, T)
    data = pd.DataFrame(np.squeeze(signals), columns=[channel_name])

    return data

# Class Preprocessing 
def remove_nans(data, add_rem=50):
	# find all rows including NaN's
	ignores = ['patient_asleep', 'Pleth', 'EEG_events_anno']
	drops = []
	for ig in ignores:
		if ig in data.columns:
			drops.append(ig)
	nan_array = np.array(data.drop(columns=drops).isna().any(axis=1)).astype('int')

	nans = np.argwhere(nan_array>0)[:,0]  

	# define shift
	shift = add_rem

	# run over all nan rows
	n = 0
	while n < len(nans)-1:
		nan = nans[n]
		beg = nan-shift if nan-shift >= 0 else 0
		end = nan+shift if nan+shift <= data.shape[0]+1 else data.shape[0]+1
		s = 1
		# skip all consequetive NaN's
		while nan+s == nans[n+s]:
			end = nan+shift+s if nan+shift+s <= data.shape[0]+1 else data.shape[0]+1
			s += 1
			if nan+s-1 == nans[-1] or n+s == len(nans):
				break
		n += s
		data.loc[beg:end, :] = None

	return data  

def window_correction(array, window_size):
	half_window = int(window_size/2)
	events = find_events(array)
	corr_array = np.array(array)
	
	# run over all events in array
	for st, end in events:
		label = array[st]
		corr_array[st-half_window-1: end+half_window] = label

	return corr_array.astype(int) 

def find_events(signal):
	# ensure np array type
	signal = np.array(signal)
		
	# ini lists
	starts, ends = [], []
	# add start if array starts with an event
	if signal[0] > 0:
		starts.insert(0,0)

	# compute diff of channel
	diff_drops = pd.DataFrame(signal).diff()

	# find starts and ends of events
	starts, ends = define_events_start_ends(signal, diff_drops, starts, ends)

	# add last ind if event did not end
	if signal[-1] > 0:
		ends.append(diff_drops.shape[0])
		signal[-1] = 0

	# check basic conditions
	if len(ends) == len(starts) == 0:
		return []
		
	assert len(ends) == len(starts), 'ERROR in <method> find_events'

	# zip start with ends
	grouped_events = list(zip(starts,ends))
	
	return grouped_events

def define_events_start_ends(signal, diff_drops, starts, ends):
	for v in np.where(diff_drops[1:])[0]:
		loc = v + 1

		step = signal[loc]
		step_min_one = signal[loc-1]

		if step > step_min_one:
			starts.append(loc)
		elif step < step_min_one:
			ends.append(loc)

	return starts, ends

def label_correction(starts, ends, signal, Fs):
	# run over all found merged events
	merged_locs = search_for_merged_labels(signal)
	for p, loc, n in merged_locs:
		# split the two events
		event1 = signal[loc-1]
		event2 = signal[loc+1]
		# check wich event has priority
		priority_loc = np.argmin([event1, event2])
		if priority_loc == 0:
			# remove second start when priority = event1
			starts = [s for s in starts if p != s]
			# and convert second end into first end
			w = np.where(ends==n)[0]
			if len(w) > 0:
				ends[w[0]] = loc
		else:
			# remove first end when priority = event2
			ends = [e for e in ends if loc != e]
			# and convert first start into second start
			starts[np.where(starts==p)[0][0]] = loc

	return starts, ends

def events_to_array(events, len_array, labels=[]):
	array = np.zeros(len_array)
	if len(labels)==0: labels = [1]*len(events)
	for i, (st, end) in enumerate(events):
		array[st:end] = labels[i]
		
	return array


# More preprocessing 
def do_initial_preprocessing(signals, new_Fs, original_Fs):
    from mne.filter import filter_data, notch_filter
    from scipy.signal import resample_poly
    notch_freq_us = 60.                 # [Hz]
    notch_freq_eur = 50.                # [Hz]
    bandpass_freq_eeg = [0.1, 20]       # [Hz] [0.5, 40]
    bandpass_freq_airflow = [0., 10]    # [Hz]
    bandpass_freq_ecg = [0.3, None]     # [Hz]

    # setup new signal DF
    new_df = pd.DataFrame([], columns=signals.columns)

    for sig in signals.columns:
        # 1. Notch filter
        image = signals[sig].values
        if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2',
                        'abd', 'chest', 'effort', 'airflow', 'ptaf', 'cflow', 'breathing_trace', 'ecg']:
            image = notch_filter(image.astype(float), original_Fs, notch_freq_us, verbose=False)
            # image = notch_filter(image, 200, notch_freq_eur, verbose=False)

        # 2. Bandpass filter
        if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2']:
            image = filter_data(image, original_Fs, bandpass_freq_eeg[0], bandpass_freq_eeg[1], verbose=False)
        if sig in ['abd', 'chest', 'effort', 'airflow', 'ptaf', 'cflow', 'breathing_trace']:
            image = filter_data(image, original_Fs, bandpass_freq_airflow[0], bandpass_freq_airflow[1], verbose=False)
        if sig == 'ecg':
            image = filter_data(image, original_Fs, bandpass_freq_ecg[0], bandpass_freq_ecg[1], verbose=False)

        # 3. Resample data
        if new_Fs != original_Fs:
            if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2', 
                            'abd', 'chest', 'effort', 'airflow', 'ptaf', 'cflow', 'breathing_trace', 'ecg']:
                image = resample_poly(image, new_Fs, original_Fs)
            else:
                image = np.repeat(image, new_Fs)
                image = image[::original_Fs]                

        # 4. Insert in new DataFrame
        new_df.loc[:, sig] = image
    
    del signals
    return new_df

def clip_normalize_signals(signals, sample_rate, min_max_times_global_iqr=20):
    # run over all channels
    for chan in signals.columns:
        # skip labels
        if np.any([t in chan for t in ['stage', 'arousal', 'resp', 'cpap_pressure', 'cpap_on']]): continue
        if np.all(signals[chan] == 0): continue

        signal = signals.loc[:, chan].values
        # clips spo2 @60%
        if chan == 'spo2':
            signals.loc[:, chan] = np.clip(signal.round(), 60, 100)
            continue

        # for all EEG (&ECG) traces
        if chan in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2', 'ecg']:
            # Compute global IQR
            iqr = np.subtract(*np.percentile(signal, [75, 25]))
            threshold = iqr * min_max_times_global_iqr

            # clip outliers
            signal_clipped = np.clip(signal, -threshold, threshold)

            # normalize channel
            sig = np.atleast_2d(signal_clipped).T
            transformer = RobustScaler().fit(sig)
            signal_normalized = np.squeeze(transformer.transform(sig).T)        

        # for all breathing traces
        elif chan in ['abd', 'chest', 'effort', 'airflow', 'ptaf', 'cflow', 'breathing_trace']:
            # cut split-night recordings and do only local normalization
            region = np.arange(len(signal))

            # ski if all zeros
            if np.all(signal[region]==0): continue
            
            # normalize signal
            signal_clipped = np.clip(signal, np.nanpercentile(signal[region],5), np.nanpercentile(signal[region],95))
            signal_normalized = np.array((signal - np.nanmean(signal_clipped)) / np.nanstd(signal_clipped))
            
            # clip extreme values
            clp = 0.01 if chan in ['abd', 'chest', 'effort'] else 0.001
            factor = 10 if chan in ['abd', 'chest', 'effort'] else 20
            quan = 0.2 
            thresh = np.mean((np.abs(np.nanquantile(signal_normalized[region], quan)), np.abs(np.nanquantile(signal_normalized[region], 1-quan))))
            thresh = factor*thresh
            if region[0] == 0:
                signal_normalized[np.concatenate([signal_normalized[region] < -thresh, np.full(len(signal)-len(region), False)])] = -thresh
                signal_normalized[np.concatenate([signal_normalized[region] > thresh, np.full(len(signal)-len(region), False)])] = -thresh
            else:
                signal_normalized[np.concatenate([np.full(len(signal)-len(region), False), signal_normalized[region] < -thresh])] = -thresh
                signal_normalized[np.concatenate([np.full(len(signal)-len(region), False), signal_normalized[region] > thresh])] = thresh
            
        # replace original signal
        signals.loc[:, chan] = signal_normalized
        
    return signals


# Flow reduction analysis
def assess_ventilation(data, hdr, drop_hyp, drop_apnea, dur_apnea, dur_hyp, quant, extra_smooth=False, plot=False):
    # compute dynamic excursion threshold, both for apnea and hypopneas.
    Fs = hdr['newFs']
    excursion_duration = 60     # the larger the interval, the less dynamic a baseline gets computed.
    excursion_q = quant         # ventilation envelope quantile

    # use lagging moving windonw, events are found based on future eupnea / recovery breaths
    pos_excursion = data.Ventilation_pos_envelope.rolling(excursion_duration*Fs*2).quantile(excursion_q, interpolation='lower').values
    pos_excursion[:-excursion_duration*Fs*2] = pos_excursion[excursion_duration*Fs*2:]
    pos_excursion[-excursion_duration*Fs*2:] = np.nan
    neg_excursion = data.Ventilation_neg_envelope.rolling(excursion_duration*Fs*2).quantile(1-excursion_q, interpolation='lower').values
    neg_excursion[:-excursion_duration*Fs*2] = neg_excursion[excursion_duration*Fs*2:]
    neg_excursion[-excursion_duration*Fs*2:] = np.nan

    # add additional envelope smoothing
    if extra_smooth:
        minutes = 20
        win = int(Fs*60*minutes)
        pos = pd.DataFrame(data=pos_excursion).rolling(win, center=True, min_periods=1).quantile(0.4).values
        neg = pd.DataFrame(data=neg_excursion).rolling(win, center=True, min_periods=1).quantile(0.6).values
        pos_excursion = np.squeeze(pos)
        neg_excursion = np.squeeze(neg)  
        # compute smoothed uncalibrated apnea excursion
        middle = np.mean([pos_excursion, neg_excursion], 0)
        pos_distance_to_baseline = np.abs(pos_excursion-middle)
        neg_distance_to_baseline = np.abs(neg_excursion-middle)

    # Relative pos/neg excursion (Hypopnneas)
    pos_distance_to_baseline = np.abs(pos_excursion - data['Ventilation_baseline'])
    neg_distance_to_baseline = np.abs(neg_excursion - data['Ventilation_baseline'])
    data['pos_excursion_hyp'] = data['Ventilation_baseline'] + (pos_distance_to_baseline * (1-drop_hyp))
    data['neg_excursion_hyp'] = data['Ventilation_baseline'] - (neg_distance_to_baseline * (1-drop_hyp))
    ### add soft hypopneas ###
    data['pos_excursion_soft_hyp'] = data['Ventilation_default_baseline'] + (pos_distance_to_baseline * (1-drop_hyp/1.02))
    data['neg_excursion_soft_hyp'] = data['Ventilation_default_baseline'] - (neg_distance_to_baseline * (1-drop_hyp/1.02))

    # Average pos/neg excursion (Apneas)
    pos_distance_to_baseline = np.abs(pos_excursion - data['Ventilation_default_baseline'])
    neg_distance_to_baseline = np.abs(neg_excursion - data['Ventilation_default_baseline'])
    dist_to_baseline = np.mean([pos_distance_to_baseline, neg_distance_to_baseline], 0)
    data['pos_excursion_apnea'] = data['Ventilation_default_baseline'] + (dist_to_baseline * (1-drop_apnea))
    data['neg_excursion_apnea'] = data['Ventilation_default_baseline'] - (dist_to_baseline * (1-drop_apnea))

    # find drops in ventilation signal for apneas and hypopneas
    data = locate_ventilation_drops(data, hdr, dur_apnea, dur_hyp)

    # combine positive and negative excursion flow limitations
    data = pos_neg_excursion_combinations(data, hdr)

    return data

def remove_non_forward_drops(data, hdr, drop_hyp, drop_apnea, dur_apnea, dur_hyp, quant, extra_smooth=False):
    # compute dynamic excursion threshold, both for apnea and hypopneas.
    Fs = hdr['newFs']
    excursion_duration = 60     # the larger the interval, the less dynamic a baseline gets computed.
    excursion_q = quant         # ventilation envelope quantile

    # use lagging moving windonw, events are found based on future eupnea / recovery breaths
    pos_excursion = data.Ventilation_pos_envelope.rolling(excursion_duration*Fs*2).quantile(excursion_q, interpolation='lower').values
    neg_excursion = data.Ventilation_neg_envelope.rolling(excursion_duration*Fs*2).quantile(1-excursion_q, interpolation='lower').values

    # add additional envelope smoothing
    if extra_smooth:
        minutes = 5
        win = int(Fs*60*minutes)
        pos = pd.DataFrame(data=pos_excursion).rolling(win, center=True, min_periods=1).quantile(0.4).values
        neg = pd.DataFrame(data=neg_excursion).rolling(win, center=True, min_periods=1).quantile(0.6).values
        pos_excursion = np.squeeze(pos)
        neg_excursion = np.squeeze(neg)  

    # set excursion into DF
    pos_distance_to_baseline = np.abs(pos_excursion - data['Ventilation_baseline'])
    neg_distance_to_baseline = np.abs(neg_excursion - data['Ventilation_baseline'])
    data['pos_excursion_apnea'] = data['Ventilation_baseline'] + (pos_distance_to_baseline * (1-drop_apnea))
    data['pos_excursion_hyp'] = data['Ventilation_baseline'] + (pos_distance_to_baseline * (1-drop_hyp))
    data['neg_excursion_apnea'] = data['Ventilation_baseline'] - (neg_distance_to_baseline * (1-drop_apnea))
    data['neg_excursion_hyp'] = data['Ventilation_baseline'] - (neg_distance_to_baseline * (1-drop_hyp))
    ### add soft hypopneas ###
    data['pos_excursion_soft_hyp'] = data['Ventilation_default_baseline'] + (pos_distance_to_baseline * (1-drop_hyp/1.02))
    data['neg_excursion_soft_hyp'] = data['Ventilation_default_baseline'] - (neg_distance_to_baseline * (1-drop_hyp/1.02))
   
    # find drops in ventilation signal for apneas and hypopneas
    data = locate_ventilation_drops(data, hdr, dur_apnea, dur_hyp)

    apnea_cols = [c for c in data.columns if '_Ventilation_drop_apnea' in c]
    hyp_cols = [c for c in data.columns if '_Ventilation_drop_hypopnea' in c]
    soft_hyp_cols = [c for c in data.columns if '_Ventilation_drop_soft_hypopnea' in c]
    
    # run over all found apneas
    for st, end in find_events(data['Ventilation_drop_apnea']>0):
        # if no apnea is found in either trace, remove apnea
        if np.all(data.loc[st:end, apnea_cols]==0):
            data.loc[st:end, 'Ventilation_drop_apnea'] = 0
            # if hypopnea is found, replace apnea by hypopnea
            if np.any(data.loc[st:end, hyp_cols]>0):
                data.loc[st:end, 'Ventilation_drop_hypopnea'] = 1

    # run over all found hypopneas
    for st, end in find_events(data['Ventilation_drop_hypopnea']>0):
        # remove when no forward hypopnea is found
        if np.all(data.loc[st:end, hyp_cols]==0):
            data.loc[st:end, 'Ventilation_drop_hypopnea'] = 0

    # run over all found soft hypopneas
    for st, end in find_events(data['Ventilation_drop_soft_hypopnea']>0):
        # remove when no forward hypopnea is found
        if np.all(data.loc[st:end, soft_hyp_cols+hyp_cols]==0):
            data.loc[st:end, 'Ventilation_drop_soft_hypopnea'] = 0

    return data

def locate_ventilation_drops(data, hdr, dur_apnea, dur_hyp):
    # remove NaN values only for selected channels -->
    selected_columns = ['pos_excursion_apnea', 'pos_excursion_hyp', 'pos_excursion_soft_hyp',
                        'neg_excursion_apnea', 'neg_excursion_hyp', 'neg_excursion_soft_hyp',
                        'Ventilation_baseline']
    new_df = data[selected_columns + ['Ventilation_combined']].copy()

    # put selected columns in original dataframe
    for col in selected_columns:
        data[col] = new_df[col]

    # *add smoothed exursion thresholds*
    win = int(hdr['newFs']*60*10)
    data['pos_excursion_apnea_smooth'] = data['pos_excursion_apnea'].rolling(win, center=True, min_periods=1).median()
    data['neg_excursion_apnea_smooth'] = data['neg_excursion_apnea'].rolling(win, center=True, min_periods=1).median()
    data['pos_excursion_hyp_smooth'] = data['pos_excursion_hyp'].rolling(win, center=True, min_periods=1).median()
    data['neg_excursion_hyp_smooth'] = data['neg_excursion_hyp'].rolling(win, center=True, min_periods=1).median()
    data['pos_excursion_soft_hyp_smooth'] = data['pos_excursion_soft_hyp'].rolling(win, center=True, min_periods=1).median()
    data['neg_excursion_soft_hyp_smooth'] = data['neg_excursion_soft_hyp'].rolling(win, center=True, min_periods=1).median()

    # find areas with potential apnea / hypopnea flow limitations
    sig = data.Ventilation_combined
    data['pos_Ventilation_drop_apnea'] = np.logical_or(sig<data.pos_excursion_apnea, sig<data.pos_excursion_apnea_smooth)
    data['neg_Ventilation_drop_apnea'] = np.logical_or(sig>data.neg_excursion_apnea, sig>data.neg_excursion_apnea_smooth)
    data['pos_Ventilation_drop_hypopnea'] = np.logical_or(sig<data.pos_excursion_hyp, sig<data.pos_excursion_hyp_smooth)
    data['neg_Ventilation_drop_hypopnea'] = np.logical_or(sig>data.neg_excursion_hyp, sig>data.neg_excursion_hyp_smooth)
    data['pos_Ventilation_drop_soft_hypopnea'] = np.logical_or(sig<data.pos_excursion_soft_hyp, sig<data.pos_excursion_soft_hyp_smooth)
    data['neg_Ventilation_drop_soft_hypopnea'] = np.logical_or(sig>data.neg_excursion_soft_hyp, sig>data.neg_excursion_soft_hyp_smooth)

    # run over the various ventilation drop options, and find flow limitations
    data['either_hypes'] = 0  
    tag_window = [dur_apnea*0.8, dur_apnea, dur_hyp, dur_hyp]
    for ex in ['pos', 'neg']:
        tag_list = [f'{ex}_soft_ventilation_drop_apnea', 
                    f'{ex}_Ventilation_drop_apnea', 
                    f'{ex}_Ventilation_drop_hypopnea',
                    f'{ex}_Ventilation_drop_soft_hypopnea']
        data = find_flow_limitations(data, tag_list, tag_window, hdr['newFs'])

    return data

def find_flow_limitations(data, tag_list, tag_window, Fs):
    for t, tag in enumerate(tag_list):
        win = int(tag_window[t]*Fs)
        # find events with duration <win>
        if t==0:
            col = [c for c in tag_list if 'Ventilation_drop_apnea' in c]
            data[tag] = data[col].rolling(win, center=True).mean() > 0.75 # allow for small peaks exceeding threshold
        else:
            data[tag] = data[tag].rolling(win, center=True).mean() == 1 # <win> should stay below threshold

        # apply window correction
        data[tag] = np.array(data[tag].fillna(0))
        cut = Fs//2 if 'hypopnea' in tag else 0 # slightly shorten event for Hypopneas
        data[tag] = window_correction(data[tag], window_size=win-cut)

        # remove all apnea events with a duration > .. sec
        max_dur = 150 if 'hypopnea' in tag else 120
        data = remove_long_events(data, tag, Fs, max_duration=max_dur)

    return data

def pos_neg_excursion_combinations(data, hdr):
    # run over apnea options
    apnea_cols = ['Ventilation_drop_apnea', 'soft_ventilation_drop_apnea']
    hypopnea_cols = ['Ventilation_drop_hypopnea', 'Ventilation_drop_soft_hypopnea']
    for col in apnea_cols + hypopnea_cols:
        # for soft apneas, pos and neg flow limitation has to occur simultaniously
        if 'soft_ventilation' in col:
            data[col] = (data['pos_%s'%col] * data['neg_%s'%col]) > 0
        # for hypopneas, either positive or negative flow limitation is saved (prioiritize pos) 
        elif 'hypopnea' in col:
            data = hyp_flow_limitations(data, col)
        # for apneas, pos and neg flow limitation has to occur simultaniously
        else:
            data[col] = (data['pos_%s'%col] * data['neg_%s'%col]) > 0
            # connect apneas if pos or neg criteria continues
            data[col] = connect_apneas(data, col, 10, hdr['newFs'], max_dur=120)
            
        # connect events, within 5 sec (only if total event < 20sec)
        events = find_events(data[col].fillna(0)>0)
        if len(events) == 0: continue
        events, _ = connect_events(events, 3, hdr['newFs'], max_dur=20)
        data[col] = events_to_array(events, len(data))
        
        # remove events < 4 sec
        data[col] = remove_short_events(data[col], 4*hdr['newFs'])

    # remove soft ventilation drop apnea, if apnea found
    for st, end in find_events(data['soft_ventilation_drop_apnea']>0):
        if np.any(data.loc[st:end, 'Ventilation_drop_apnea']==1):
            data.loc[st:end, 'soft_ventilation_drop_apnea'] = 0
    # remove soft ventilation drop hypopnea, if apnea/hypopnea found
    for st, end in find_events(data['Ventilation_drop_soft_hypopnea']>0):
        if np.any(data.loc[st:end, ['Ventilation_drop_apnea', 'Ventilation_drop_hypopnea']]==1):
            data.loc[st:end, 'Ventilation_drop_soft_hypopnea'] = 0

    return data

def hyp_flow_limitations(data, col):
    data[col] = data[f'pos_{col}'].values
    neg = find_events(data[f'neg_{col}'].fillna(0)>0)

    # run over flow limitation regions
    for st, end in neg:
        region = list(range(st, end))
        
        # skip if already saved in pos array
        if any(data.loc[region, f'pos_{col}']): continue

        # save in array
        data.loc[region, col] = 1

    return data

def combine_flow_reductions(data, hdr):
    # set data arrays
    Fs = hdr['newFs']
    apneas = data.Ventilation_drop_apnea
    hypopneas = data.Ventilation_drop_hypopnea
    grouped_hypopneas = find_events(hypopneas>0)
    data['flow_reductions'] = apneas

    # add hypopneas to apnea array
    for st, end in grouped_hypopneas:
        region = list(range(st, end))
        # insert when no apnea is found in that region
        if np.all(apneas[region] == 0):
            data.loc[region, 'flow_reductions'] = 2
        else:
            reg = region[2*Fs:-2*Fs]
            if len(reg)<10*Fs: continue
            if np.all(apneas[reg] == 0):
                data.loc[reg, 'flow_reductions'] = 2
    
    return data

def connect_apneas(data, col, win, Fs, max_dur=False):
    # set events
    events = find_events(data[col].fillna(0)>0)
    if len(events) == 0: return data[col].values

    # connect apneas if pos or neg negative treshold remains
    new_events = []
    cnt = 0
    win = win*Fs
    while cnt < len(events)-1:
        st = events[cnt][0]
        end = events[cnt][1]
        dist1 = events[cnt+1][0] - end 
        condition1 = (dist1<win) if max_dur == False else (dist1<win) and ((events[cnt+1][1]-st) < max_dur*Fs)
        condition2 = any(np.all(data.loc[st:end+dist1, [f'pos_{col}', f'neg_{col}']] == 1, 0))
        if condition1 and condition2:      
            new_events.append((st, events[cnt+1][1]))
            cnt += 2
        else:
            new_events.append((st, end))
            cnt += 1  
    new_events.append((events[-1]))

    # convert back to array
    new_array = events_to_array(new_events, len(data))

    return new_array

##
def connect_events(events, win, Fs, max_dur=False, labels=[]):
    new_events, new_labels = [], []
    if len(events) > 0 :
        cnt = 0
        win = win*Fs
        if len(labels)==0: labels = [1]*len(events)
        while cnt < len(events)-1:
            st = events[cnt][0]
            end = events[cnt][1]
            dist1 = events[cnt+1][0] - end 
            condition1 = (dist1<win) if max_dur == False else (dist1<win) and ((events[cnt+1][1]-st) < max_dur*Fs)
            if condition1:            
                new_events.append((st, events[cnt+1][1]))
                lab = labels[cnt] if end-st > events[cnt+1][1]-events[cnt+1][0] else labels[cnt+1]
                new_labels.append(lab)
                cnt += 2
            else:
                new_events.append((st, end))
                new_labels.append(labels[cnt])
                cnt += 1  
        new_events.append((events[-1]))
        new_labels.append((labels[-1]))

    return new_events, new_labels

def merge_small_events(data, Fs):
    # define global hypopnea flow reduction threshold
    global_hyp_thresh = np.nanmedian(data['pos_excursion_hyp'])
    while True:
        # run over all flow reductions
        all_flow_reductions = find_events(data['flow_reductions']>0)
        if len(all_flow_reductions)<2: return data
        for i, (st, end) in enumerate(all_flow_reductions[:-1]):
            next_st = all_flow_reductions[i+1][0]
            next_end = all_flow_reductions[i+1][1]
            ss = int(np.median((st, end)))
            ee = int(np.median((next_st, next_end)))
            region = list(range(ss, ee))
            # skip events that would become >2min
            if (next_end-st) > 60*Fs: continue

            # if ventilation trace stays below local or global hyp threshold, merge events (by filling inbetween region)
            if len(data.loc[region, 'pos_excursion_hyp'].dropna()) == 0: continue
            local_thresh = np.nanmedian(data.loc[region, 'pos_excursion_hyp'])
            local_trace = data.loc[region, 'Ventilation_combined']
            check1 = np.sum(local_trace < local_thresh) > 0.9*len(region)
            check2 = np.sum(local_trace < global_hyp_thresh) > 0.9*len(region)
            if check1 or check2:
                # only if there is intermittant sleep
                if not np.any(data.loc[list(range(end, next_st)), 'Stage'] == 5):
                    vals, cnts = np.unique(data.loc[region, 'flow_reductions'], return_counts=True)
                    num = 1 if len(vals[1:]) == 1 and vals[1] == 1 else 4
                    data.loc[list(range(st, next_end)), 'flow_reductions'] = num
                    break
        if i == len(all_flow_reductions)-2: 
            break
        
    return data

def remove_wake_events(data):
    # run over all flow reductions
    all_flow_reductions = find_events(data['flow_reductions']>0)
    for i, (st, end) in enumerate(all_flow_reductions[:-1]):
        region = list(range(st, end))
        if np.sum(data.loc[region, 'patient_asleep']==0) > 0.75*len(region):
            data.loc[region, 'flow_reductions'] = 0

    return data


# Post-processing 
def remove_long_events(data, tag, Fs, max_duration=60):
    data['too_long_events'] = 0
    # find and remove events with duration > 'max_duration'
    events = [ev for ev in find_events(data[tag]>0) if ev[1]-ev[0] > max_duration*Fs]
    for st, end in events:
        region = list(range(st, end))
        data.loc[region, 'too_long_events'] = 1
        # fill only part associated with sat-drop or arousal
        if tag == 'algo_apneas' and data.loc[st, 'algo_apneas']==4:
            data.loc[region, 'algo_apneas'] = 0
            # fill based on single saturation drop found
            drops = [a for a in find_events(data.loc[region, 'saturation_drop']>0) if a[0] > 10*Fs]
            arousals = [a for a in find_events(data.loc[region, 'EEG_arousals']>0) if a[0] > 10*Fs]
            if len(drops) > 0:
                if len(drops) != 1: continue
                loc = int(st+np.mean(drops[0]))
                fill = list(range(loc-30*Fs, loc))
            # otherwise do based on arousals
            elif len(arousals) > 0:
                if len(arousals) != 1: continue
                loc = int(st+np.mean(arousals[0]))
                fill = list(range(loc-30*Fs, loc))
            # if none found, don't fill
            else: continue
            # fill based on location
            data.loc[fill, 'algo_apneas'] = 4
        else:
            # remove region
            data.loc[region, tag] = 0

    return data

def remove_short_events(array, duration, skip_RERA=False):
    # find and remove events with duration < 'duration'
    array = np.array(array)
    for st, end in find_events(array>0): 
        # skip RERAs
        if array[st] == 5 and skip_RERA: continue
        region = list(range(st, end))
        if len(region) < duration:
            array[region] = 0

    return array


# Envelope analysis
def compute_envelopes(data, Fs, channels='abd'):
    # set ABD trace to ventilation combined
    data['Ventilation_combined'] = data[channels] if type(channels) != list else data[channels].mean(axis=1)

    # compute envelope and baseline
    new_df = compute_envelope(data['Ventilation_combined'], int(Fs), env_smooth=5)
    data.loc[:, 'Ventilation_pos_envelope'] = new_df['pos_envelope'].values
    data.loc[:, 'Ventilation_neg_envelope'] = new_df['neg_envelope'].values
    data.loc[:, 'Ventilation_default_baseline'] = new_df['baseline'].values
    data.loc[:, 'Ventilation_baseline'] = new_df['correction_baseline'].values

    return data

def compute_envelope(signal, Fs, base_win=30, env_smooth=5):
    new_df = pd.DataFrame()
    new_df['x'] = signal

    # determine peaks of signal
    x = new_df['x']
    pos_peaks, _ = find_peaks(x, distance=int(Fs*1.5), width=int(0.4*Fs), rel_height=1)
    neg_peaks, _ = find_peaks(-x, distance=int(Fs*1.5), width=int(0.4*Fs), rel_height=1)

    new_df['pos_envelope'] = x[x.index[0] + pos_peaks]
    new_df['neg_envelope'] = x[x.index[0] + neg_peaks]

    # compute envelope of signal
    new_df['pos_envelope'] = new_df['pos_envelope'].interpolate(method='cubic', order=1, limit_area='inside')
    new_df['neg_envelope'] = new_df['neg_envelope'].interpolate(method='cubic', order=1, limit_area='inside')        
    new_df['pos_envelope'] = new_df['pos_envelope'].rolling(env_smooth*Fs, center=True).median()
    new_df['neg_envelope'] = new_df['neg_envelope'].rolling(env_smooth*Fs, center=True).median()
    check_invalids = new_df['pos_envelope'] < new_df['neg_envelope']
    new_df.loc[check_invalids, 'pos_envelope'] = 0
    new_df.loc[check_invalids, 'neg_envelope'] = 0
    
    new_df['baseline'], new_df['baseline2'], new_df['correction_baseline'] = compute_baseline(new_df, Fs, base_win)

    return new_df

def compute_baseline(new_df, Fs, base_win, correction_ratio=2):
    # compute baseline of signal
    pos = new_df['pos_envelope'].rolling(base_win*Fs, center=True).mean()
    neg = new_df['neg_envelope'].rolling(base_win*Fs, center=True).mean()
    base = (pos + neg) / 2

    base1 = new_df['x'].rolling(base_win*Fs, center=True).median().rolling(base_win*Fs, center=True).mean()
    base2 = base.rolling(base_win*Fs, center=True).mean()

    base_corr = (correction_ratio*base1 + base2) / (1+correction_ratio)
    base_corr = base_corr.rolling(base_win*Fs, center=True).mean()
    
    return base1, base2, base_corr

def compute_smooth_envelope(data, region):
    try:
        # analyze the two envelope traces
        for env_tag in ['Smooth_pos_envelope', 'Smooth_neg_envelope']:
            # create smoothed envelope
            original_env = env_tag.replace('Smooth', 'Ventilation')
            
            sig = data.loc[region, original_env].copy().astype(float)

            # If NaN occuring. interpolate small gaps, leave long ones masked
            nan_frac = sig.isna().mean()
            if nan_frac < 0.10:
                sig = sig.interpolate(limit_direction="both")  # fills short NaN runs
            else:
                # If >10% NaNs, safer fallback: median fill or skip
                sig = sig.fillna(sig.median())

            data.loc[region, env_tag] = savgol_filter(sig, 51, 1)

    except Exception as e:
        raise EnvelopeComputationFailed("compute smooth envelope failed") from e

    return data


# Self-Similarity 
def assess_potential_self_sim_spots(data, Fs):
    # binarize labels
    data['Smooth_pos_envelope'] = 0
    data['Smooth_neg_envelope'] = 0
    data['TAGGED'] = 0

    # run over each potential self similarity region
    for self_sim_st, self_sim_end in find_events(data.potential_self_sim.values>0):
        # get all flow reductions in self similarity region
        self_sim_region = list(range(self_sim_st, self_sim_end))
        flow_lims = find_events(data.loc[self_sim_region, 'flow_reductions']>0)

        for i, (st, end) in enumerate(flow_lims[:-1]):
            # define corrected event locations
            st, end = self_sim_st + st, self_sim_st + end
            next_start, next_end = self_sim_st + flow_lims[i+1][0], self_sim_st + flow_lims[i+1][1]

            # set search regions
            region1 = list(range(st, end))
            region2 = list(range(next_start, next_end))
            region_full = list(range(int(np.median((st, end))), int(np.median((next_start, next_end)))))

            # compute envelope + cycle
            data = compute_smooth_envelope(data, region_full)

            # determine cycle spots
            cycle = find_cycle_spots(data, [region1, region2])

            # compare 1st half to 2nd half of smooth envelope
            conv_scores = convolve_envelope(data, cycle, Fs)

            # apply self-sim tests
            data, tests = do_self_sim_tests(data, cycle, conv_scores, Fs)

    return data

def tag_potential_self_sim_spots(data, Fs):
    data['potential_self_sim'] = 0
    labels = np.array(data.flow_reductions.fillna(0) > 0).astype(int)

    epoch_size = int(180*Fs)
    epoch_inds = np.arange(0, len(labels)-epoch_size+1, 5*Fs)
    seg_ids = list(map(lambda x:np.arange(x, x+epoch_size), epoch_inds))

    for seg_id in seg_ids:
        try:
            if len(find_events(labels[seg_id])) >= 3:
                data.loc[seg_id, 'potential_self_sim'] = 1
        except:import pdb; pdb.set_trace()

    return data

def post_process_self_sim(data, Fs, SS_threshold):
    data['self similarity'] = 0
    data['consecutive complexes'] = 0
    data['ss_conv_score'] = np.nan

    # find 3 consecutive HLG-looking breathing oscillations    
    tagged = data.TAGGED
    window = 180*Fs # use a sliding window with length <window>
    rolling_sum = tagged.rolling(window, center=True).sum()
    data.loc[rolling_sum>=3, 'consecutive complexes'] = 2
    data['consecutive complexes'] = window_correction(data['consecutive complexes'], window_size=window)

    # assess self-similarity for each three complexs
    for st, end in find_events(data['consecutive complexes']):
        complexes = find_events(data.loc[list(range(st,end)), 'TAGGED']) + st
        for t in range(len(complexes)-2):
            tags = complexes[t:t+3]
            conv_score = assess_three_breathing_oscillations(data, tags, Fs)
            num = 2 if conv_score >= SS_threshold else 1
            # save 'red' for self-similarity, 'black' for chains w/o self-similarity
            if t == 0:
                ss = tags[0][0]
                ee = tags[1][0] + (tags[2][0] - tags[1][0])//2
                data.loc[list(range(ss, ee)), 'self similarity'] = num
                data.loc[tags[0][0], 'ss_conv_score'] = conv_score
                data.loc[tags[1][0], 'ss_conv_score'] = conv_score
            if t == len(complexes)-3 or len(complexes)==3:
                ss = tags[0][0] + (tags[1][0] - tags[0][0])//2
                ee = tags[2][1]
                data.loc[tags[1][0], 'ss_conv_score'] = conv_score
                data.loc[tags[2][0], 'ss_conv_score'] = conv_score
            else:   
                ss = tags[0][0] + (tags[1][0] - tags[0][0])//2
                ee = tags[1][0] + (tags[2][0] - tags[1][0])//2
                data.loc[tags[1][0], 'ss_conv_score'] = conv_score
            
            data.loc[list(range(ss, ee)), 'self similarity'] = num

    return data

def find_cycle_spots(data, regions, iq=.1):
    # find bottoms in events
    cycle = []
    for i in range(2):
        # event_data = data.loc[regions[i], :]
        # mins = []
        # for env_tag in ['Smooth_pos_envelope', 'Smooth_neg_envelope']:
        #     if 'pos' in env_tag:
        #         thresh = event_data.loc[:, env_tag].quantile(iq)
        #         locs = np.where(event_data.loc[:, env_tag] < thresh)[0]
        #     elif 'neg' in env_tag:
        #         thresh = event_data.loc[:, env_tag].quantile(1-iq)
        #         locs = np.where(event_data.loc[:, env_tag] > thresh)[0]
        #     mins.append(locs[len(locs)//2] + event_data.index[0])
        # cycle.append(int(np.mean(mins)))
        cycle.append(regions[i][len(regions[i])//2])

    # find top inbetween
    local_data = data.loc[list(range(cycle[0], cycle[1])), :]
    top = np.argmax(local_data['Smooth_pos_envelope']) + local_data.index[0]
    bot = np.argmin(local_data['Smooth_neg_envelope']) + local_data.index[0]
    cycle.append(int(np.mean([top, bot])))

    return cycle

def convolve_envelope(data, cycle, Fs):
    conv_scores = np.zeros(3)

    # specify region of interest
    baseline = data.loc[cycle[0]:cycle[1], 'Ventilation_baseline'].values
    pos = data.loc[cycle[0]:cycle[1], 'Smooth_pos_envelope'].values - baseline
    neg = baseline - data.loc[cycle[0]:cycle[1], 'Smooth_neg_envelope'].values

    # normalize envelopes
    if not all(np.isnan(pos)):
        pos = (pos - np.nanmean(pos)) / (np.nanstd(pos) + 0.000001)
        neg = (neg - np.nanmean(neg)) / (np.nanstd(neg) + 0.000001)

    # apply convolution
    pos[np.isnan(pos)] = 0
    neg[np.isnan(neg)] = 0
    val = np.nanmax(convolve(pos, neg, mode='same')) / len(pos)
    conv_scores[2] = val

    return conv_scores

def assess_three_breathing_oscillations(data, tags, Fs):
    # retrieve envelopes from the three segments
    pos_envelopes, neg_envelopes = [], []
    for t in range(3):
        loc = tags[t][0]
        win = 20*Fs
        pos_envelopes.append(data.loc[list(range(loc-win, loc+win)), 'Smooth_pos_envelope'].values)
        neg_envelopes.append(data.loc[list(range(loc-win, loc+win)), 'Smooth_neg_envelope'].values)

    # compute convulion scores twice, previous and next oscillation
    conv_scores = []
    for oscillation, reference in [(pos_envelopes[1], pos_envelopes[0]), (pos_envelopes[1], pos_envelopes[2])]:
        # normalize envelopes
        first = (oscillation - np.mean(oscillation)) / (np.std(oscillation) + 0.000001)
        second = (reference - np.mean(reference)) / (np.std(reference) + 0.000001)

        # apply convolution
        conv_scores.append(np.nanmax(convolve(first, second, mode='same')) / len(first))

    return np.max(conv_scores)

def do_self_sim_tests(data, cycle, conv_scores, Fs):
    tests = {}

    # single peak
    d_t = (cycle[1] - cycle[0]) / Fs
    pass_test = True if d_t < 120 and d_t > 10 else False
    tests['duration test'] = (pass_test, d_t)
    
    # relative height test
    sig = data.loc[list(range(cycle[0],cycle[1])), 'Ventilation_combined'].values
    thresh = np.nanmean(data.loc[:, 'pos_excursion_hyp'].values)
    pass_test = True if np.any(sig > thresh) else False
    tests['relative height'] = (pass_test, int(pass_test))

    # peak timing score
    p_t = (1 - np.abs((cycle[2]-cycle[0]) - (cycle[1]-cycle[2])) / (cycle[1]-cycle[0])) * 100
    pass_test = True if p_t > 50 else False
    tests['peak timing'] = (pass_test, p_t)

    # vertical mirror score
    h_s = conv_scores[2] * 100
    pass_test = True if h_s > 50 else False
    tests['horizontal symmetry'] = (pass_test, h_s)

    if np.all([tests[key][0] for key in tests.keys()]):
        data.loc[cycle[2], 'TAGGED'] = 1

    return data, tests


# Central event count
def compute_central_events(data, hdr):
    # create central apnea/hypopnea cols
    hdr['central apneas'], hdr['central hypopneas'] = 0, 0
    data['central apneas'], data['central hypopneas'] = 0, 0
    
    # run over all events
    i = 0
    events = find_events(data['flow_reductions']>0)
    for st, end in events[:-1]:
        # set start and ends
        mid1 = int(np.mean([events[i][0], events[i][1]]))
        mid2 = int(np.mean([events[i+1][0], events[i+1][1]]))
        # if (mid2-mid1) > hdr['newFs']*5*60: continue
        # if two events surround SS tag
        if any(data.loc[mid1:mid2, 'T_sim'] == 1):
            # save both events in their respective arrays
            for event in [events[i], events[i+1]]:
                val = data.loc[event[0], 'flow_reductions']
                tag = 'apneas' if val==1 else 'hypopneas'
                data.loc[event[0]:event[1], f'central {tag}'] = 1
                hdr[f'central {tag}'] += 1
            i += 2
        else: i += 1
        if i >= len(events)-2: break

    return data, hdr

def compute_main_channel(cols, channels):
    report_check = False
    for col, channel in zip(cols, channels):
        # perform main report/figure check
        check1 = 'effort' in cols and col=='effort'
        check2 = 'effort' not in cols and col=='abd'
        check3 = 'effort' not in cols and 'abd' not in cols and col =='chest'
        check4 = 'effort' not in cols and 'abd' not in cols and 'chest' not in cols and col=='ptaf'
        check5 = 'effort' not in cols and 'abd' not in cols and 'chest' not in cols and 'ptaf' not in cols and col=='airflow'
        if report_check==False and (check1 or check2 or check3 or check4 or check5):
            return (col, channel)
    
    return None

# Report
def create_report(output_data, hdr):
    # set sampling frequencies
    orinalFs = hdr['Fs']
    newFs = hdr['newFs']
    finalFs = 1

    # Init DF
    original_cols = ['flow_reductions', 'T_sim', 'TAGGED']
    data = pd.DataFrame([], columns=['start_idx', 'end_idx'] + original_cols)
    
    # Resample data to 1 Hz
    for sig in original_cols: 
        image = np.repeat(output_data[sig].values , finalFs)
        image = image[::newFs]    
        # 4. Insert in new DataFrame
        data[sig] = image
    
    # save columns of interest
    factor = orinalFs // finalFs
    ind0 = np.arange(0, data.shape[0]) * factor
    ind1 = np.concatenate([ind0[1:], [ind0[-1]+factor]])
    data['second'] = range(len(data))
    data['start_idx'] = ind0
    data['end_idx'] = ind1
    data['SS'] = data['T_sim'].values
    
    # create summary report
    summary_report = pd.DataFrame([])
    duration = len(data)/finalFs/3600
    summary_report['signal duration (h)'] = [np.round(duration, 2)]
    summary_report['detected central apneas'] = [hdr[f'central apneas']]
    summary_report['detected central hypopneas'] = [hdr[f'central hypopneas']]
    summary_report['cai'] = [np.round(hdr[f'central apneas'] / duration, 1)]
    summary_report['cahi'] = [np.round((hdr[f'central apneas']+hdr[f'central hypopneas']) / duration, 1)]
    summary_report['SS%'] = np.round((np.sum(data['T_sim']==1) / (len(data))) * 100, 1)  

    # remove original columns
    for col in original_cols: 
        if col in data.columns: 
            data = data.drop(columns=col)   
    
    # save data into .csv files
    full_report = pd.concat([data, summary_report], axis=1)
    
    return full_report, summary_report

def dict_SS_per_channel(reports):
    SS_dict = {}

    # run over all summary reports
    keys = [k for k in reports.keys() if 'summary' in k]
    for key in keys:
        col = key.split('_')[1]
        SS_dict[col] = reports[key]['SS%'].values[0]
    
    return SS_dict

def update_summary_report_with_any_SS(reports, summary_report):
    # run over all summary reports
    keys = [k for k in reports.keys() if 'full' in k]
    if not keys:
        summary_report['SS%'] = np.nan
        return summary_report, np.array([], dtype=int)

    stacked_values: List[np.ndarray] = []
    for key in keys:
        vals = np.asarray(reports[key]['SS'].values).astype(int)
        if vals.ndim == 0:
            vals = vals.reshape(1)
        stacked_values.append(vals)

    total = np.vstack(stacked_values)
    any_SS = np.any(total, axis=0).astype(int)
    denom = any_SS.size if any_SS.size else 1
    summary_report['SS%'] = np.round(any_SS.sum() / denom * 100, 1)
          
    return summary_report, any_SS

# Plotting
def self_sim_plot(data, hdr, summary_report, main_ch, plot_all_tagged=False):
    # take middle 5hr segment --> // 10 rows == 30 min per row
    fs = hdr['newFs']
    if isinstance(main_ch, tuple):
        final_plot = True
        main_col, main_ch = main_ch
        reports = summary_report
        summary_report = reports[f'summary_{main_col}']
        # create SS per channel list
        SS_per_channel = dict_SS_per_channel(reports)
        # compute and update summary
        summary_report, any_SS = update_summary_report_with_any_SS(reports, summary_report)
    else:
        final_plot = False
    main_ch_str = '' if main_ch is None else str(main_ch)
    
    # set signal variables
    signal = data.Ventilation_combined.values.astype(float)
    sleep_stages = data.Stage.values.astype(float)
    y_algo = data['central apneas'] + data['central hypopneas']*4
    tagged_breaths = data.tresh_TAGS.values.astype(int)
    ss_conv_score = data.ss_conv_score.values.astype(float)
    selfsim = data.T_sim.values.astype(int)
    if final_plot:
        any_SS = np.repeat(any_SS, fs)
        any_SS[selfsim==1] = 0


    # define the ids each row
    block = 60*60*fs
    row_ids = [np.arange(i*block, (i+1)*block) for i in range(len(signal)//block+1)]
    row_ids.reverse()
    row_ids[0] = np.arange(row_ids[0][0], len(data))
    nrow = len(row_ids)

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    row_height = 15

    # set sleep array
    sleep = np.array(signal)
    sleep[np.isnan(sleep_stages)] = np.nan
    sleep[sleep_stages==5] = np.nan
    # set wake array
    wake = np.zeros(signal.shape)
    wake[np.isnan(sleep_stages)] += signal[np.isnan(sleep_stages)]
    wake[sleep_stages==5] += signal[sleep_stages==5]
    wake[wake==0] = np.nan
    # set rem array
    rem = np.array(signal)
    rem[sleep_stages!=4] = np.nan
    # set envelope + baseline
    env_pos = data.Smooth_pos_envelope.values
    env_neg = data.Smooth_neg_envelope.values
    baseline = data.Ventilation_baseline.values

    # PLOT SIGNALS
    for ri in range(nrow):
        # set clip and autoscale factor
        pos_clip = (ri+1)*row_height - row_height/1.8
        neg_clip = (ri-1)*row_height + row_height/1.8
        factor = max(1.5 / np.quantile(signal[row_ids[ri]], 0.90), 1)        

        # plot signal
        for array, c in zip([sleep, wake, rem], ['k', 'r', 'b']):
            # fade large peaks
            arr = array[row_ids[ri]] * factor + ri*row_height
            fade = array[row_ids[ri]] * factor + ri*row_height
            arr[np.logical_or(arr>pos_clip, arr<neg_clip)] = np.nan
            # lower bound cut
            arr[arr < -row_height] = np.nan
            fade[fade < -row_height] = np.nan
            # upper bound cut
            arr[arr > nrow*row_height] = np.nan
            fade[fade > nrow*row_height] = np.nan
            # plot
            ax.plot(fade, c=c, lw=.3, alpha=0.0)
            ax.plot(arr, c=c, lw=.3)
            
        # set max_y
        if ri == nrow-1:
            max_y = np.nanmax([sleep[row_ids[ri]], wake[row_ids[ri]], rem[row_ids[ri]]]) + ri*row_height

    # PLOT LABELS
    ran = 4 if final_plot else 3
    for yi in range(ran):
        if yi==0:
            labels = y_algo                 # plot tech label
            label_color = [None, 'b', 'k', 'k', 'm', None, None]
        if yi == 1:
            labels = tagged_breaths         # '*' for HLG breathing oscillations
            label_color = [None, 'k', 'r']
        if yi == 2:
            labels = selfsim         # '*' for HLG breathing oscillations
            label_color = [None, 'b']
        if yi == 3:
            labels = any_SS         # '*' for HLG breathing oscillations
            label_color = [None, 'b']

        # run over each plot row
        for ri in range(nrow):            
            # group all labels and plot them
            loc = 0
            height = ri*row_height
            for i, j in groupby(labels[row_ids[ri]]):
                len_j = len(list(j))
                if not np.isnan(i) and label_color[int(i)] is not None:
                    # if yi == 0:
                    #     # add scored events
                    #     ax.plot([loc, loc+len_j], [height-2.5]*2, c=label_color[int(i)], lw=1)
                    if yi == 1:
                        # add tags
                        tag = '*' if i == 1 else '\''
                        c_score = np.round(ss_conv_score[row_ids[ri]][loc], 2)
                        c, sz = ('b', 12) if c_score >= hdr['SS_threshold'] else ('k', 8)
                        if c_score >= hdr['SS_threshold'] or plot_all_tagged:
                            offset = row_height/2
                            ax.text(loc, height + offset, tag, c=c, ha='center', va='top', fontsize=sz)
                            # ax.text(loc, height + offset+0.5, str(c_score), c='k', ha='center', va='bottom', fontsize=3)
                    if yi == 2:
                        # add SS bar
                        ymin = height - (row_height/2*0.9)
                        ymax = height + row_height/2
                        ax.fill_between([loc, loc+len_j], ymin, ymax, color='b', alpha=0.15, ec=None)
                    if yi == 3:
                        # add SS_any bar
                        ymin = height - (row_height/2*0.81)
                        ymax = height + (row_height/2*0.9)
                        ax.fill_between([loc, loc+len_j], ymin, ymax, color='k', alpha=0.15, ec=None)

                loc += len_j
                
    # plot layout setup
    ax.set_xlim([-0.01*block, 1.01*block])
    ax.axis('off')

    ### construct legend box ###
    len_x = len(row_ids[-1])

    # add <duration> min marking
    duration = 5
    offset = row_height*(nrow-1) + 17
    ax.plot([len_x-60*fs*duration, len_x], [offset]*2, color='k', lw=1)           # <duration>
    ax.plot([len_x-60*fs*duration]*2, [offset-0.5, offset+0.5], color='k', lw=1)  # left va
    ax.plot([len_x]*2, [offset-0.5, offset+0.5], color='k', lw=1)                 # right va
    ax.text(len_x-60*fs*(duration/2), offset+1, f'{duration} min', color='k', fontsize=8, ha='center', va='bottom')
    if main_ch_str:
        ax.text(len_x-60*fs*(duration/2), offset-1, f'({main_ch_str})', color='k', fontsize=8, ha='center', va='top')

    # add summary report
    dx = len_x//12
    for i, key in enumerate(summary_report.keys()):
        tag = key.replace('detected ', '').replace('central', 'c.').replace('signal ', '')
        if final_plot and key=='SS%':
            tag = '$SS_{{any}}$%'
        tag += '\n' + str(summary_report[key].values[0])
        ax.text((i)*dx, offset, tag, fontsize=7, ha='left', va='bottom')

    # add metrics per channel to final plot
    if final_plot:
        # add border
        ax.plot([(i+0.9)*dx]*2, [offset, offset+4.5], color='k', lw=1)

        shift = len(summary_report.keys())*dx
        dxx = len_x//16
        for i, key in enumerate(SS_per_channel.keys()):
            tag = f'$SS_{{{key}}}$'
            tag += f'%\n{SS_per_channel[key]}'
            ax.text(shift + i*dxx, offset, tag, fontsize=7, ha='left', va='bottom')
        

    plt.tight_layout()
    # plt.show()

# Apply algorithm
def run_SS_algorithm(data, hdr, col):
    # set hyperparamters
    hdr['SS_threshold'] = 0.80

    # compute envelope and baseline on ABD trace
    try:
        data = compute_envelopes(data, hdr['newFs'], channels=col)
    except Exception as e:
        raise ValueError("EnvelopeComputationFailed")

    # compute ventilation drops
    drop_h, drop_a, dur_apnea, dur_hyp, quant = 0.40, 0.85, 7, 7, 0.65
    data = assess_ventilation(data, hdr, drop_h, drop_a, dur_apnea, dur_hyp, quant, extra_smooth=True, plot=False)

    # remove found flow reductions, with no clear forward drop
    data = remove_non_forward_drops(data, hdr, drop_h, drop_a, dur_apnea, dur_hyp, quant, extra_smooth=True)

    # create flow reduction array
    data = combine_flow_reductions(data, hdr)

    # merge small separate event into long events
    data = merge_small_events(data, hdr['newFs'])

    # remove wake events
    data = remove_wake_events(data)

    # find potential self-similarity regions
    data = tag_potential_self_sim_spots(data, hdr['newFs'])

    # assess potential self-similarity regions
    data = assess_potential_self_sim_spots(data, hdr['newFs'])

    # apply AASM-rule post-processing
    data = post_process_self_sim(data, hdr['newFs'], hdr['SS_threshold'])

    # save SS score
    self_sim = data['self similarity'].values
    self_sim[self_sim!=2] = 0
    self_sim[self_sim==2] = 1
    data['T_sim'] = self_sim
    data['tresh_TAGS'] = np.array(data.ss_conv_score>hdr['SS_threshold']).astype(int)

    # compute number of central apneas hypopneas
    data, hdr = compute_central_events(data, hdr)

    return data, hdr

# Main helpers
def append_summary_entry(summary_path: Path, file_name: str, main_channel: str,
                         summary_main: pd.DataFrame, summary_any: pd.DataFrame) -> None:
    """Append a one-line summary for a processed file."""

    def _get_value(df: pd.DataFrame, column: str) -> float:
        if column not in df.columns:
            return np.nan
        return float(df[column].iloc[0])

    row = {
        'file': file_name,
        'fileid': Path(file_name).stem,
        'main_channel': str(main_channel) if main_channel is not None else '',
        'signal_duration_h': _get_value(summary_main, 'signal duration (h)'),
        'central_apneas': _get_value(summary_main, 'detected central apneas'),
        'central_hypopneas': _get_value(summary_main, 'detected central hypopneas'),
        'cai': _get_value(summary_main, 'cai'),
        'cahi': _get_value(summary_main, 'cahi'),
        'ss_percent_main': _get_value(summary_main, 'SS%'),
        'ss_percent_any': _get_value(summary_any, 'SS%'),
    }

    summary_df = pd.DataFrame([row])
    lock_path = Path(f"{summary_path}.lock")
    with FileLock(str(lock_path)):
        header = not summary_path.exists()
        summary_df.to_csv(summary_path, mode='a', header=header, index=False)


def process_file(file_path: Path,
                 summary_csv: Optional[Path],
                 plots_dir: Path,
                 reports_root: Path,
                 new_fs: int = 10,
                 plot_all_tagged: bool = False,
                 hypoxic_burden_map: Optional[dict[str, float]] = None) -> None:
    file_path = Path(file_path)
    file_id = file_path.stem
    file_name = file_path.name
    file_suffix = file_path.suffix.lower()
    # reports_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_report_path = reports_root / f'{file_id}_summary.csv'
    summary_plot_pdf = plots_dir / f'{file_id}_self_similarity.pdf'
    summary_plot_png = plots_dir / f'{file_id}_self_similarity.png'

    channel_reports_dir = reports_root / file_id if plot_all_tagged else None
    channel_plots_dir = (plots_dir / 'individual' / file_id) if plot_all_tagged else None
    if channel_reports_dir is not None:
        channel_reports_dir.mkdir(parents=True, exist_ok=True)
    if channel_plots_dir is not None:
        channel_plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        stage_array = None
        if file_suffix == '.edf':
            mne = _import_mne()
            edf = mne.io.read_raw_edf(str(file_path), stim_channel=None, preload=False, verbose=False)
            hdr = {'Fs': int(edf.info['sfreq']), 'newFs': new_fs}
            try:
                vprint(' searching for breathing signals ..')
                data, channels = multiple_channel_search(edf)
                vprint(f'   found {channels}\n')
            except Exception:
                vprint('\nunsuccessful --> do original effort search instead..\n')
                data = original_effort_search(edf, file_path)
                channels = list(data.columns)
        elif file_suffix == '.h5':
            signals, annotations, params = load_prepared_data(str(file_path))
            hdr = {'Fs': int(params['fs']), 'newFs': new_fs}
            data, channels = extract_breathing_channels(signals)
            if 'stage' in annotations.columns:
                stage_array = annotations['stage'].values.astype(float).flatten()
        else:
            raise ValueError(f'Unsupported file type: {file_suffix}')

        data = do_initial_preprocessing(data, hdr['newFs'], hdr['Fs'])

        if 'abd' in data.columns and 'chest' in data.columns and 'effort' not in data.columns:
            data['effort'] = data['abd'] + data['chest']
            channels = list(channels) + ['Sum Effort']
        else:
            channels = list(channels)

        data = clip_normalize_signals(data, hdr['newFs'])

        main_col_chan = compute_main_channel(data.columns, channels)
        if main_col_chan is None:
            raise RuntimeError('Could not determine main channel for reporting.')

        if stage_array is not None and len(stage_array) > 0:
            fs_ratio = hdr['Fs'] / hdr['newFs']
            if float(fs_ratio).is_integer():
                stage_resampled = stage_array[::int(fs_ratio)]
                if len(stage_resampled) == len(data):
                    data['Stage'] = stage_resampled
                else:
                    data['Stage'] = 1
            else:
                data['Stage'] = 1
        else:
            data['Stage'] = 1
        data['patient_asleep'] = np.logical_and(data.Stage>0, data.Stage<5)
        original_df = data.copy().astype(float)
        reports: Dict[str, pd.DataFrame] = {}
        hb_value = None
        if hypoxic_burden_map:
            hb_value = hypoxic_burden_map.get(file_name)
            if hb_value is None:
                hb_value = hypoxic_burden_map.get(file_id)

        for idx, (col, channel_label) in enumerate(zip(data.columns, channels)):
            channel_label_str = str(channel_label)
            if verbose:
                print(f"Assessing SS in '{channel_label_str}' [{idx+1}/{len(channels)}]    ", end='\r')
            if idx > 0:
                data = original_df.copy()

            try:
                data, hdr = run_SS_algorithm(data, hdr, col)
            except EnvelopeComputationFailed:
                continue
            except Exception as e:
                raise ValueError(f"SSComputationFailed for channel '{col}': {e}")
                
            full_report, summary_report = create_report(data, hdr)
            reports[f'full_{col}'] = full_report
            reports[f'summary_{col}'] = summary_report

            if plot_all_tagged and channel_reports_dir is not None and channel_plots_dir is not None:
                label_slug = channel_label_str.replace(' ', '_')
                report_path = channel_reports_dir / f'{label_slug}_summary.csv'
                full_report.to_csv(report_path, header=full_report.columns, index=None, mode='w+')

                self_sim_plot(data, hdr, summary_report, channel_label_str)
                plot_prefix = channel_plots_dir / f'{file_id}_{label_slug}'
                plt.savefig(fname=str(plot_prefix.with_suffix('.pdf')), format='pdf', dpi=1200)
                plt.savefig(fname=str(plot_prefix.with_suffix('.png')), dpi=300)
                plt.close()

        if verbose:
            print()

        if hb_value is not None:
            hb_value_rounded = float(np.round(hb_value, 1))
            for key, report in reports.items():
                if key.startswith('summary_'):
                    report['hypoxic burden'] = [hb_value_rounded]

        summary_main_copy = reports[f'summary_{main_col_chan[0]}'].copy()

        final_full_report = reports[f'full_{main_col_chan[0]}']
        # final_full_report.to_csv(summary_report_path, header=final_full_report.columns, index=None, mode='w+')

        self_sim_plot(data, hdr, reports, main_col_chan, plot_all_tagged=plot_all_tagged)
        plt.savefig(fname=str(summary_plot_pdf), format='pdf', dpi=1200)
        # plt.savefig(fname=str(summary_plot_png), dpi=300)
        plt.close()

        summary_any_copy = reports[f'summary_{main_col_chan[0]}'].copy()

        if summary_csv is not None:
            append_summary_entry(summary_csv, file_name, str(main_col_chan[1]), summary_main_copy, summary_any_copy)

    except Exception as exc:
        raise RuntimeError(f'Failed to process {file_path}: {exc}') from exc


def main():
    parser = argparse.ArgumentParser(
        description="Compute self-similarity metrics for EDF/H5 files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-i', '--input-path', '--i',
        type=Path,
        default=None,
        help="Directory with EDF/H5 files, a CSV with a 'file_path' column, or a single file",
    )
    parser.add_argument(
        '-d', '--results-dir',
        type=Path,
        default=None,
        help="Directory where results will be stored (defaults to ./results).",
    )
    parser.add_argument(
        '--new-fs',
        type=int,
        default=10,
        help="Sampling rate in Hz for resampling respiratory signals",
    )
    parser.add_argument(
        '-io', '--individual-output',
        action='store_true',
        help="Generate individual output files for each channel (default: False)",
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Enable informative console output."
    )
    parser.add_argument(
        '--errors-log',
        type=Path,
        default=None,
        help="Optional path to write a processing error log (defaults to ./self_similarity_errors.log when needed)."
    )

    args = parser.parse_args()

    global verbose
    verbose = args.verbose

    input_table = None
    if args.input_path is None:
        file_paths = _collect_data_files(Path.cwd())
    else:
        input_path = Path(args.input_path).expanduser()
        if input_path.is_dir():
            file_paths = _collect_data_files(input_path)
        elif input_path.is_file():
            if input_path.suffix.lower() == '.csv':
                input_table = pd.read_csv(input_path)
                if 'file_path' not in input_table.columns:
                    raise ValueError("Input table must contain a 'file_path' column.")
                file_paths = [Path(p).expanduser() for p in input_table['file_path'].tolist()]
            else:
                file_paths = [input_path]
        else:
            raise FileNotFoundError(f'Input path {input_path} does not exist.')

    errors: List[tuple[Path, str]] = []
    resolved_paths: List[Path] = []
    for path in file_paths:
        path_obj = Path(path).expanduser()
        if not path_obj.exists():
            try:
                resolved_missing = path_obj.resolve()
            except FileNotFoundError:
                resolved_missing = Path.cwd() / path_obj
            errors.append((resolved_missing, 'File not found'))
            continue
        resolved_paths.append(path_obj.resolve())

    resolved_paths = sorted({p for p in resolved_paths}, key=str)

    if not resolved_paths:
        print('No EDF or H5 files found for processing.')
        if errors:
            log_path = _resolve_error_log_path(args.errors_log)
            _write_error_log(log_path, errors)
            if verbose:
                print(f'Encountered {len(errors)} issue(s). Details saved to {log_path}.')
            else:
                print(f'Encountered {len(errors)} issue(s). See {log_path} for details.')
        return

    results_root = Path(args.results_dir).expanduser() if args.results_dir is not None else Path.cwd() / 'results'
    hypoxic_burden_map = _load_hypoxic_burden_map(results_root)
    summary_csv, plots_dir, reports_root = _prepare_self_similarity_paths(results_root)

    plot_all_tagged = args.individual_output

    processed_files: set[str] = set()
    if summary_csv.exists():
        try:
            summary_df = pd.read_csv(summary_csv)
            if 'file' in summary_df.columns:
                processed_files = set(summary_df['file'].astype(str))
        except Exception as exc:
            if verbose:
                print(f'Warning: could not read existing summary at {summary_csv}: {exc}')

    remaining_paths: List[Path] = []
    for file_path in resolved_paths:
        if file_path.name in processed_files:
            if verbose:
                print(f'Skipping {file_path.name}: already present in summary CSV.')
            continue
        remaining_paths.append(file_path)

    if not remaining_paths:
        if verbose:
            print('All files already present in summary CSV. Nothing to process.')
        if errors:
            log_path = _resolve_error_log_path(args.errors_log)
            _write_error_log(log_path, errors)
            if verbose:
                print(f'Encountered {len(errors)} issue(s). Details saved to {log_path}.')
            else:
                print(f'Encountered {len(errors)} issue(s). See {log_path} for details.')
        return

    if verbose:
        print(f'Processing {len(remaining_paths)} pending file(s)...')

    for file_path in tqdm(remaining_paths, desc='Self Similarity', unit='file'):
        try:
            process_file(
                file_path,
                summary_csv=summary_csv,
                plots_dir=plots_dir,
                reports_root=reports_root,
                new_fs=args.new_fs,
                plot_all_tagged=plot_all_tagged,
                hypoxic_burden_map=hypoxic_burden_map
            )
        except Exception as exc:
            message = f'{file_path}: {exc}'
            if verbose:
                print(f'[ERROR] {message}')
            errors.append((file_path, str(exc)))
        else:
            processed_files.add(file_path.name)

    if errors:
        log_path = _resolve_error_log_path(args.errors_log)
        _write_error_log(log_path, errors)
        if verbose:
            print(f'Encountered {len(errors)} issue(s). Details saved to {log_path}.')
        else:
            print(f'Encountered {len(errors)} issue(s). See {log_path} for details.')
    elif args.errors_log is not None:
        log_path = _resolve_error_log_path(args.errors_log)
        _write_error_log(log_path, [])
        if verbose:
            print(f'No errors encountered. Wrote empty log to {log_path}.')
    elif verbose:
        print('Processing complete without errors.')


if __name__ == '__main__':
    main()
