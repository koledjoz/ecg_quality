import numpy as np

import tf_model
import utils
import neurokit2 as nk

model_list = ['base_model']


class ECG_Quality_Checker():

    def __init__(self, model:str = 'base_model', stride:float = 0.0, return_mode:str = 'score', thresholds:list = None, return_type:str = 'full', sampling_rate:int = 250, clean_data:bool=True):
        # urobime checks vsetkych modelov

        if model not in model_list:
            raise ValueError(model + ' is not a known model that can be used')

        if return_mode not in ['score', 'two_value', 'three_value']:
            raise ValueError('Return mdoe needs to be one of: score, two_value, three_value. Currently is ' + str(return_mode))

        if thresholds is None:
            thresholds = utils.get_default_thresholds(return_mode)


        if return_mode == 'score':
            if thresholds is not None:
                raise ValueError('Threshold count does not correspond to return mode')
        elif return_mode == 'two_value':
            if len(thresholds) != 1:
                raise ValueError('Threshold count does not correspond to return mode')
            elif not 0 <= thresholds[0] <= 1.0:
                raise ValueError('Threshold value not in 0 to 1 interval')
        elif return_mode == 'three_value':
            if len(thresholds) != 2:
                raise ValueError('Threshold count does not correspond to return mode')
            elif not 0 <= thresholds[0] <= 1.0 or not 0 <= thresholds[1] <= 1.0:
                raise ValueError('Threshold value not in 0 to 1 interval')
            thresholds = np.sort(thresholds)

        if return_type not in ['intervals', 'full']:
            raise ValueError('Return type needs to be one of: intervals, full. Currently is ' + str(return_type))

        if sampling_rate != 250:
            raise NotImplementedError('This class currently only support ECG with frequency of 250 Hz. Consider modyfing your data to comply with this prerequisite.')

        if stride < 0:
            raise ValueError('Stride for a model can not be negative')

        self.model = tf_model.tf_model(model)

        self.input_length = self.model.get_input_length()

        self.stride = utils.get_stride_length(self.model.get_input_length(), stride, sampling_rate)
        self.return_mode = return_mode
        self.return_type = return_type
        self.thresholds = thresholds
        self.sampling_rate = sampling_rate
        self.clean_data = clean_data

    def process_signal(self, signal):

        if self.return_type == 'full':
            return self._process_signal_full(signal)
        elif self.return_type == 'interval':
            return self._process_signal_interval(signal)

    def _process_signal_full(self, signal):
        if self.clean_data:
            signal = nk.ecg_clean(signal, sampling_rate=self.sampling_rate)

        output = np.zeros_like(signal)
        win_count = np.zeros_like(signal)

        for win_start in range(0, len(signal)-self.input_length, self.stride):
            score = self.model.process_ecg(signal[win_start:win_start + self.input_length])
            output[win_start:win_start + self.input_length] = output[win_start:win_start + self.input_length] + score
            win_count[win_start:win_start + self.input_length] = win_count[win_start:win_start + self.input_length] + 1
        return self._calc_precise_scores(output, win_count)

    def _process_signal_interval(self, signal):

        if self.clean_data:
            signal = nk.ecg_clean(signal, sampling_rate=self.sampling_rate)

        output = nk.zeros(len(signal) // self.stride)
        win_count = nk.zeros(len(signal) // self.stride)

        for win_start in range(0, len(signal) - self.input_length, self.stride):
            a = win_start // self.stride
            b = (win_start + self.input_length) // self.stride

            score = self.model.process_ecg(signal[win_start:win_start + self.input_length])
            output[a:b] = output[a:b] + score
            win_count[a:b] = win_count[a:b] + 1
        return self._calc_precise_scores(output, win_count)

    def _get_two_value(self, scores):
        return [0.0 if x < self.thresholds[0] else 1.0 for x in scores]

    def _get_three_value(self, scores):

        def mapper(val):
            if val < self.thresholds[0]:
                return 0.0
            elif val < self.thresholds[1]:
                return 0.5
            else:
                return 1.0
        return [mapper(x) for x in scores]

    def _calc_precise_scores(self, scores, win_counts):
        scores = np.divide(scores, win_counts)
        scores = scores[~np.isnan(scores)]

        if self.return_mode == 'score':
            return scores
        elif self.return_mode == 'two_value':
            return self._get_two_value(scores)
        elif self.return_mode == 'three_value':
            return self._get_three_value(scores)