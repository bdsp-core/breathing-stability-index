#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   hypoxic_burden 
@Time        :   2023/4/17 14:22
@Author      :   Xuesong Chen
@Description :   
"""
import os, sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, find_peaks
import neurokit2 as nk
from Reader.Base.EDFReader import EDFReader, EDFHeader
from Reader.Base.AnnotationReader import AnnotationReader
from ODI import detect_oxygen_desaturation


B = [0.000109398212241, 0.000514594526374, 0.001350397179936, 0.002341700062534,
     0.002485940327008, 0.000207543145171, -0.005659450344228, -0.014258087808069,
     -0.021415481383353, -0.019969417749860, -0.002425120103463, 0.034794452821365,
     0.087695691366900, 0.144171828095816, 0.187717212244959, 0.204101948813338,
     0.187717212244959, 0.144171828095816, 0.087695691366900, 0.034794452821365,
     -0.002425120103463, -0.019969417749860, -0.021415481383353, -0.014258087808069,
     -0.005659450344228, 0.000207543145171, 0.002485940327008, 0.002341700062534,
     0.001350397179936, 0.000514594526374, 0.000109398212241]

BAD_SPO2_THRESHOLD = 80


def filter_spo2(spo2_arr, spo2_sfreq, verbose=False):
    # 将异常值，用均值替代
    spo2_mean = np.mean(spo2_arr[spo2_arr >= BAD_SPO2_THRESHOLD])
    spo2_arr[spo2_arr < BAD_SPO2_THRESHOLD] = spo2_mean

    if spo2_sfreq != 1:
        spo2_arr = nk.signal_resample(spo2_arr, sampling_rate=spo2_sfreq, desired_sampling_rate=1)
        spo2_sfreq = 1

    # 减少血氧抖动，并将血氧分辨率调整为0.5
    spo2_filtered = filtfilt(B, 1, spo2_arr, axis=0, padtype='odd')
    spo2_filtered *= 2
    spo2_filtered = np.round(spo2_filtered) / 2

    if verbose:
        abd = reader.get_single_channel_data('Abdominal', int(end - time_span), int(end + time_span))
        af = reader.get_single_channel_data('Airflow', int(end - time_span), int(end + time_span))
        tt = np.arange(len(spo2_arr))/spo2_sfreq
        tt_abd = np.arange(len(abd))/reader.get_channel_sample_frequency('Abdominal')
        tt_af = np.arange(len(af))/reader.get_channel_sample_frequency('Airflow')

        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(311); ax0 = ax
        ax.plot(tt, spo2_arr)
        ax.plot(tt, spo2_filtered)
        ax.axvline(x=time_span/spo2_sfreq)
        ax = fig.add_subplot(312, sharex=ax0)
        ax.plot(tt_abd, abd)
        ax = fig.add_subplot(313, sharex=ax0)
        ax.plot(tt_af, af)

        plt.show()
        #plt.savefig(f'./img/b/{idx}.png')
    return spo2_filtered


def calc_hypoxic_burden(event_times, reader, spo2_name, verbose=False, time_span=120):
    # 假设呼吸事件的时长为10~120s，由呼吸事件引起的低氧事件的时长最大delay为120s
    all_ah_related_spo2 = []
    good_event_ids = []
    spo2_sfreq = reader.get_channel_sample_frequency(spo2_name)
    #assert spo2_sfreq == 1
    for ei, et in enumerate(event_times):
        nearby_spo2 = reader.get_single_channel_data(spo2_name, int(et - time_span), int(et + time_span))
        if len(nearby_spo2) < 2*time_span*spo2_sfreq \
                or np.mean(nearby_spo2 < BAD_SPO2_THRESHOLD)>0.3:
            continue
        filtered_spo2 = filter_spo2(nearby_spo2, spo2_sfreq, verbose=False)
        # now sfreq=1
        all_ah_related_spo2.append(filtered_spo2)
        good_event_ids.append(ei)
    all_spo2_dest = np.array(all_ah_related_spo2)
    avg_spo2 = all_spo2_dest.mean(axis=0)
    avg_spo2 = filtfilt(B, 1, avg_spo2, axis=0, padtype='odd')

    peaks, _ = find_peaks(avg_spo2)
    start_secs = peaks[np.where(peaks < time_span)[0][-1]]
    end_secs = peaks[np.where(peaks > time_span)[0][0]]
    if verbose:
        x = np.arange(len(avg_spo2))
        plt.close()
        plt.plot(x, avg_spo2)
        plt.plot(x[start_secs], avg_spo2[start_secs], "o")
        plt.plot(x[end_secs], avg_spo2[end_secs], "*", markersize=10)
        plt.axvline(x=time_span)
        plt.title(f"{name}")
        plt.show()

    burdens = []
    for spo2_dest_curve in all_spo2_dest:
        baseline_spo2 = np.max(spo2_dest_curve[time_span - 100:time_span])
        interest_spo2 = spo2_dest_curve[start_secs: end_secs]
        burdens.append( sum(baseline_spo2 - interest_spo2)/60 )
    #per_ah_event_burden = total_burden / len(all_spo2_dest)
    #total_burden += per_ah_event_burden * (len(event_times) - len(all_spo2_dest))
    #total_sleep_time_in_hours = anno.get_total_sleep_time() / 60 / 60
    #return total_burden / total_sleep_time_in_hours
    res = pd.DataFrame(data={'EventTime':event_times})
    res.loc[good_event_ids, 'HB'] = burdens
    return res


if __name__ == '__main__':
    df = pd.read_csv('/data/haoqisun/AD_PD_prediction_from_sleep/mastersheet_MrOS_SOF.csv')
    dataset = 'SOF'
    types = ['desat', 'apnea']
    df = df[df.Dataset==dataset].reset_index(drop=True)
    
    cols = ['SID', 'Site', 'Dataset', 'HB_desat', 'HB_NREM_desat', 'HB_REM_desat', 'HB_apnea', 'HB_NREM_apnea', 'HB_REM_apnea']
    for type_ in types:
        df[f'HB_{type_}'] = np.nan
        df[f'HB_NREM_{type_}'] = np.nan
        df[f'HB_REM_{type_}'] = np.nan
    for i in tqdm(range(len(df))):
        sid = df.SID.iloc[i]
        edf_path = df.EDFPath.iloc[i]
        ann_path = df.AnnotPath.iloc[i]
        #header = EDFHeader(edf_path)
        spo2_name = 'SAO2'#header.get_spo2_channel()
        reader = EDFReader(edf_path, [spo2_name])
        anno = AnnotationReader(ann_path)
        sleep_ids = np.where(np.in1d(anno.stages, [1,2,3,4,5]))[0]
        if len(sleep_ids)==0:
            print(f'{sid}: len(sleep_ids) = 0 ')
            continue
        for type_ in types:
            if type_=='apnea':
                ah_events = anno.get_apnea_hypopnea()
                event_times = ah_events.Start.values+ah_events.Duration.values
            elif type_=='desat':
                od_events = anno.get_spo2_desaturation()
                if len(od_events)==0:
                    spo2 = reader.get_single_channel_data(spo2_name, tmax=None)
                    od_events = detect_oxygen_desaturation(spo2, is_plot=False)
                event_times = od_events.Start.values+od_events.Duration.values/2
            if len(event_times) < 2:
                print(f'{sid}: len(event_times) = {len(event_times)} < 2')
                continue
            df_res = calc_hypoxic_burden(event_times, reader, anno)
            ids_ = (df_res.EventTime//30).astype(int)
            df_res['Stage'] = anno.stages[np.clip(ids_, 0, len(anno.stages)-1)]
            
            start =  sleep_ids[0]*30
            end =  (sleep_ids[-1]+1)*30
            df.loc[i, f'HB_{type_}'] = df_res.HB[(df_res.EventTime>=start)&(df_res.EventTime<end)].sum()/((end-start)/3600)
            mask = np.in1d(df_res.Stage, [1,2,3,4])
            if mask.sum()>0:
                df.loc[i, f'HB_NREM_{type_}'] = df_res.HB[mask].sum()/(np.in1d(anno.stages, [1,2,3,4]).sum()*30/3600)
            mask = np.in1d(df_res.Stage, [5])
            if mask.sum()>0:
                df.loc[i, f'HB_REM_{type_}'] = df_res.HB[mask].sum()/(np.in1d(anno.stages, [5]).sum()*30/3600)
            
        if i%10==1:
            df[cols].to_csv(f'HB_results_{dataset}.csv', index=False)

    df = df[cols]
    print(df)
    df.to_csv(f'HB_results_{dataset}.csv', index=False)

