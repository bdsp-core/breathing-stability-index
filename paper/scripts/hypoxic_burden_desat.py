# %%
import os
import sys
import argparse
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath('./hypoxic_burden'))
from ODI import detect_oxygen_desaturation
from compute_hypoxic_burden import calc_hypoxic_burden
from breathing_stability import load_prepared_data
from filelock import FileLock
from tqdm import tqdm

# ---- Minimal PSG Reader for a Single Channel ----
class MinimalReader:
    def __init__(self, spo2, sfreq=1):
        self.spo2 = np.asarray(spo2)
        self.sfreq = sfreq

    def get_channel_sample_frequency(self, ch_name):
        return self.sfreq

    def get_single_channel_data(self, ch_name, start, end):
        start = max(0, int(start))
        end = min(len(self.spo2), int(end))
        return self.spo2[start:end]
    
def hypoxic_burden_main(file_path):

    # ---- Main Analysis Script ----

    # 1. Load your SpO2 signal 

    new_fs = 1  # Target sampling frequency in Hz

    if file_path.endswith('.h5'):
        # 'prepared data' format file:
        channel_names = ['spo2']
        signals, annotations, params = load_prepared_data(file_path, signals_to_load=channel_names)
        # signals shape: (n_samples, n_channels)
        signals = signals.T  # shape: (n_channels, n_samples)
        fs = params['fs']
        stage = annotations['stage'].values.flatten()
        fs_ratio = fs/new_fs
        assert fs_ratio.is_integer(), f"fs_ratio {fs_ratio} is not an integer. Cannot resample stage to {new_fs} Hz."
        stage = stage[::int(fs_ratio)]
        assert channel_names == ['spo2'], "The resampling code assumes simple Spo2 only data."
        signals = np.asarray(signals)
        signals = signals[:, ::int(fs_ratio)]  # Resample signals to new_fs
        spo2 = signals[0]  # Assuming 'spo2' is the first channel
        
    mask_sleep = np.isin(stage, [1, 2, 3, 4])  # Sleep stages: NREM (1, 2, 3), REM (4)
    spo2 = spo2[mask_sleep]
    stage = stage[mask_sleep]

    # 2. Detect desaturation events on all sleep
    od_events = detect_oxygen_desaturation(spo2, is_plot=False)
    event_times = od_events.Start.values + od_events.Duration.values

    reader = MinimalReader(spo2, sfreq=1)
    spo2_ch_name = 'SpO2'
    if len(event_times) < 2:
        hb_sleep = 0 if len(spo2) > 0 else np.nan
        hb_nrem  = 0 if np.any(np.isin(stage, [1, 2, 3])) else np.nan
        hb_rem   = 0 if np.any(stage == 4) else np.nan
        return hb_sleep, hb_nrem, hb_rem
    
    df_hb = calc_hypoxic_burden(event_times, reader, spo2_ch_name)

    # 3. Assign sleep stage to each event based on event end time
    # Each event's "end" falls in some epoch of stage_sleep; assign that stage to the event
    event_indices = np.round(event_times).astype(int)
    event_indices = np.clip(event_indices, 0, len(stage)-1)
    df_hb['Stage'] = stage[event_indices]
        
    # 4. Compute hypoxic burden for all sleep, NREM, REM
    total_hours_sleep = len(spo2) / 3600
    total_hours_nrem = np.sum(np.isin(stage, [1, 2, 3])) / 3600
    total_hours_rem  = np.sum(stage == 4) / 3600

    # 5. Compute hypoxic burden for all sleep, NREM, REM
    hb_sleep = df_hb['HB'].sum() / total_hours_sleep
    hb_nrem  = df_hb.loc[df_hb['Stage'].isin([1, 2, 3]), 'HB'].sum() / total_hours_nrem if total_hours_nrem > 0 else np.nan
    hb_rem   = df_hb.loc[df_hb['Stage'] == 4, 'HB'].sum() / total_hours_rem if total_hours_rem > 0 else np.nan

    if 0:
        print(f"Total Hypoxic Burden: {hb_sleep:.1f} units/hour")
        print(f"Total Hypoxic Burden NREM: {hb_nrem:.1f} units/hour")
        print(f"Total Hypoxic Burden REM: {hb_rem:.1f} units/hour")
        
    return hb_sleep, hb_nrem, hb_rem

def main(path_input_table):

    table = pd.read_csv(path_input_table)

    table_results = pd.DataFrame(columns=['fileid', 'file_path', 'hb_sleep', 'hb_nrem', 'hb_rem'])
    path_save_results = '/media/wolfgang/badgerhd/breathing_stability/hypoxic_burden_results.csv'

    for jloc, row in tqdm(table.iterrows()):
        
        try:
            fileid = row['fileid']
            file_path = row['file_path']
            
            hb_sleep, hb_nrem, hb_rem = hypoxic_burden_main(file_path)
            table_results = pd.concat([table_results, pd.DataFrame({
                'fileid': [fileid],
                'file_path': [file_path],
                'hb_sleep': [hb_sleep],
                'hb_nrem': [hb_nrem],
                'hb_rem': [hb_rem],
            })], ignore_index=True)
        except Exception as e:
            print(f"Error processing file {fileid}: {e}")
            continue

    # Save results to CSV
    if not os.path.exists(os.path.dirname(path_save_results)):
        os.makedirs(os.path.dirname(path_save_results))
    
    # append with FileLock, with 'append' mode:
    with FileLock(path_save_results + '.lock'):
        if os.path.exists(path_save_results):
            table_results.to_csv(path_save_results, mode='a', header=False, index=False)
        else:
            table_results.to_csv(path_save_results, mode='w', header=True, index=False)


### call this script from command line, e.g.: python hypoxic_burden_desat.py input_table.csv
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process hypoxic burden data.")
    parser.add_argument("input_table", type=str, help="Path to the input table CSV file.")
    args = parser.parse_args()
    
    main(args.input_table)