#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   EDFReader 
@Time        :   2023/4/18 12:21
@Author      :   Xuesong Chen
@Description :   
"""
from pyedflib import highlevel
from pyedflib.highlevel import read_edf_header
import re
import mne


class EDFHeader:
    def __init__(self, path):
        self.path = path
        self.header = read_edf_header(path)

    def get_all_channels(self):
        return self.header['channels']

    def get_spo2_channel(self):
        pattern = re.compile(r'S[spa]O2', re.I)
        for ch in self.header['channels']:
            res = re.search(pattern, ch)
            if res:
                return ch
        return None


class EDFReader:
    def __init__(self, path, ch_names=None):
        self.path = path
        try:
            self.signals, self.signal_headers, self.header = highlevel.read_edf(path, ch_names=ch_names)
        except Exception as ee:
            edf = mne.io.read_raw_edf(path, verbose=False, preload=False)
            if ch_names is not None:
                edf = mne.io.read_raw_edf(path, verbose=False, preload=False, exclude=[x for x in edf.ch_names if x not in ch_names])
            else:
                ch_names = edf.ch_names
            self.signals = edf.get_data(picks=ch_names)
            self.signal_headers = [{'label':x, 'sample_rate':edf.info['sfreq']} for x in ch_names]

        self.get_chan_index()

    def get_single_channel_data(self, ch_name, tmin=0, tmax=30):
        ch_idx = self.ch2index_dic[ch_name]
        sfreq = int(self.signal_headers[ch_idx]['sample_rate'])
        if not tmax:
            return self.signals[ch_idx]
        start_samp_idx = sfreq * tmin
        end_samp_idx = sfreq * tmax
        return self.signals[ch_idx][start_samp_idx:end_samp_idx]

    def get_channel_sample_frequency(self, ch_name):
        ch_idx = self.ch2index_dic[ch_name]
        sfreq = int(self.signal_headers[ch_idx]['sample_rate'])
        return sfreq

    def get_duration(self):
        ch_idx = 0
        sfreq = int(self.signal_headers[ch_idx]['sample_rate'])
        return len(self.signals[ch_idx]) / sfreq

    def get_chan_index(self):
        self.ch2index_dic = {}
        for idx, sig_header in enumerate(self.signal_headers):
            ch_name = sig_header['label']
            if ch_name in self.ch2index_dic.keys():
                ch_name = ch_name + ' 1'
            self.ch2index_dic[ch_name] = idx


if __name__ == '__main__':
    edf_reader = EDFReader(
        '/openSource/cpc_hb_code_example/example_mros_data/edfs/mros-visit1-aa0001.edf')
    print(edf_reader.get_all_channels())
