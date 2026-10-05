#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   AnnotationReader 
@Time        :   2023/4/18 12:21
@Author      :   Xuesong Chen
@Description :   
"""
import xmltodict
import numpy as np
import pandas as pd


class AnnotationReader:
    def __init__(self, path):
        with open(path, encoding='utf-8') as f:
            info_dict = xmltodict.parse(f.read())
            self.scored_events = pd.DataFrame(info_dict['CMPStudyConfig']['ScoredEvents']['ScoredEvent'])
            self.duration = self.scored_events.loc[0, 'Duration']
            self.scored_events[['Start', 'Duration']] = self.scored_events[['Start', 'Duration']].astype(float)
            #self.scored_events = self.scored_events.iloc[1:]
            self.stages = np.array(info_dict['CMPStudyConfig']['SleepStages']['SleepStage']).astype(int)

    def __len__(self):
        return self.duration

    def get_apnea_hypopnea(self):
        res = self.scored_events[
                self.scored_events['Name'].astype(str).str.lower().str.contains('pnea') |
                (self.scored_events['Name'] == 'Unsure')]
        return res#[(res['Duration'] > 10) & (res['Duration'] < 120)]

    def get_spo2_desaturation(self):
        res = self.scored_events[
            self.scored_events['Name'].astype(str).str.lower().str.contains('desat')&
            (~self.scored_events['Name'].astype(str).str.lower().str.contains('artifac'))
            ].reset_index(drop=True)
        return res

    def get_sleep_stages(self):
        return self.stages#scored_events[self.scored_events['EventType'] == 'Stages|Stages']

    def get_total_sleep_time(self):
        stages = self.get_sleep_stages()
        #return (stages!=0).sum()*30#stages[stages['EventConcept'] != 'Wake|0']['Duration'].sum()
        return np.in1d(stages, [1,2,3,4,5]).sum()*30#stages[stages['EventConcept'] != 'Wake|0']['Duration'].sum()

