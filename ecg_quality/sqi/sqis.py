from ecgdetectors import Detectors
import scipy
import numpy as np
import math
import neurokit2 as nk
from scipy import signal
import matplotlib.pyplot as plt
import EntropyHub as eh


def average_cross_correlation(signal, x):
    signal_array = np.array(signal)
    N = len(signal_array)

    # Create a lagged version of the signal
    if x != 0:
        lagged_signal = signal_array[:-x]
    else:
        lagged_signal = signal_array
    # Perform element-wise multiplication and calculate the mean
    avg_corr = np.mean(signal_array[:N - x] * lagged_signal)

    return avg_corr


class SQI_calculator():

    def __init__(self, sampling_rate: int = 250):
        self.detectors = Detectors(sampling_rate)
        self.sampling_rate = sampling_rate

    def get_peaks(self, signal, detector_name):
        if detector_name == 'wqrs':
            return [x for x in self.detectors.wqrs_detector(signal)
                    if 0.3 * self.sampling_rate <= x <= len(signal) - 0.1 * self.sampling_rate]
        if detector_name == 'engzee':
            return [x for x in self.detectors.engzee_detector(signal)
                    if 0.3 * self.sampling_rate <= x <= len(signal) - 0.1 * self.sampling_rate]

    def get_SQIs(self, signal: list, bSQI_first_detector = 'wqrs', BSQI_second_detector = 'engzee', bsSQI_detector = 'engzee', eSQI_detector = 'engzee',
                 hfSQI_dtector = 'engzee', rsdSQI_detector = 'engzee'):
        cleaned_signal = nk.ecg_clean(signal, sampling_rate=self.sampling_rate, method='biosppy')

        # entSQI je neefektivne a zle funguje
        # takze zatial bez toho, zajtra to orapvis a mozes riesit dalej, generovat data a tak podobne
        return self.hfSQI(signal), self.kSQI(signal), self.PiCASQI(signal), self.sSQI(signal), self.basSQI(signal), self.bsSQI(signal), self.entSQI(signal), self.pSQI(signal), self.purSQI(signal), self.hfMSQI(signal)

    
    def bSQI(self, signal: list, first_detector: str = 'wqrs', second_detector: str = 'engzee'):
        r_peaks1 = self.get_peaks(signal, first_detector)

        r_peaks2 = self.get_peaks(signal, second_detector)

        # this part is very inefficient for now, will need to be rewritten
        max_diff = 10
        count = 0
        same = 0
        for val in r_peaks1:
            closest_val = min(r_peaks2, key=lambda x: abs(x - val))
            if abs(closest_val - val) <= max_diff:
                count = count + 1
                same = same + 1

        for val in r_peaks1:
            closest_val = min(r_peaks2, key=lambda x: abs(x - val))

            if abs(closest_val - val) > max_diff:
                count = count + 1

        for val in r_peaks2:
            closest_val = min(r_peaks1, key=lambda x: abs(x - val))

            if abs(closest_val - val) > max_diff:
                count = count + 1

        score = same / count
        return score

    def sSQI(self, signal: list):
        return scipy.stats.skew(signal)

    def kSQI(self, signal: list):
        return scipy.stats.kurtosis(signal)

    def pSQI(self, signal: list):
        n = len(signal)
        t = 1 / self.sampling_rate

        yf = np.fft.fft(signal)
        xf = np.linspace(0.0, 1.0 / (2.0 * t), n // 2)

        pds_num = [np.abs(yf[idx]) for idx in range(len(xf)) if
                   xf[idx] >= 5 and xf[idx] <= 15]
        pds_denom = [np.abs(yf[idx]) for idx in range(len(xf)) if
                     xf[idx] >= 5 and xf[idx] <= 40]

        return np.sum(pds_num) / np.sum(pds_denom)

    def basSQI(self, signal):
        n = len(signal)
        t = 1 / self.sampling_rate

        yf = np.fft.fft(signal)
        xf = np.linspace(0.0, 1.0 / (2.0 * t), n // 2)

        pds_num = [np.abs(yf[idx]) for idx in range(len(xf)) if
                   xf[idx] >= 0 and xf[idx] <= 1]
        pds_denom = [np.abs(yf[idx]) for idx in range(len(xf)) if
                     xf[idx] >= 0 and xf[idx] <= 40]

        return 1 - (sum(pds_num) / sum(pds_denom))

    def bsSQI(self, signal: list, detector_name='engzee'):
        peaks = self.get_peaks(signal, detector_name)
        return self.bsSQI_inner(signal, peaks)

    def bsSQI_inner(self, signal, peaks):

        
        out = 0.0
        
        b, a = scipy.signal.butter(5, 1.0, fs = self.sampling_rate, btype='low', analog=False, output='ba')

        for r in peaks:
            rai = max(signal[int(max(0, r - 0.07 * self.sampling_rate)):int(min(len(signal), r + 0.08 * self.sampling_rate))])\
                - min(signal[int(max(0, r - 0.07 * self.sampling_rate)):int(min(len(signal), r + 0.08 * self.sampling_rate))])
            
            bai = max(scipy.signal.lfilter(b, a,
                                           signal[int(max(0, int(r - 1 * self.sampling_rate))):int(min(len(signal),
                                                                                              int(r + 1 * self.sampling_rate)))])) \
                - min(scipy.signal.lfilter(b, a,
                                           signal[int(max(0, int(r - 1 * self.sampling_rate))):int(min(len(signal),
                                                                                              int(r + 1 * self.sampling_rate)))]))

            out = out + (rai / bai)

        return 1 / len(peaks) * out

    def eSQI(self, signal: list, detector_name='engzee'):
        peaks = self.get_peaks(signal, detector_name)
        return self.eSQI_inner(signal, peaks)

    def eSQI_inner(self, signal, peaks):
        energy = 0.0

        for r in peaks:
            energy = energy + scipy.sum(
                [x ** 2 for x in signal[int(r - 0.07 * self.sampling_rate):int(r + 0.08 * self.sampling_rate)]])

        return energy / scipy.sum(x ** 2 for x in signal)

    def hfSQI(self, signal, detector_name='engzee'):
        peaks = self.get_peaks(signal, detector_name)
        return self.hfSQI_inner(signal, peaks)

    def hfSQI_inner(self, signal, peaks):

        if len(peaks) == 0:
            return -1000
        
        b = [1, -2, 1]
        a = [1, 1, 1]


        y = scipy.signal.convolve(signal, [1, -2, 1], mode='same')

        s = np.convolve(y, [1, 1, 1, 1, 1, 1], mode='same')


        out = 0.0
        for r in peaks:
            
            rai = max(
                signal[int(max(0, r - 0.07 * self.sampling_rate)):int(min(len(signal), r + 0.08 * self.sampling_rate))]) \
                - min(
                signal[int(max(0, r - 0.07 * self.sampling_rate)):int(min(len(signal), r + 0.08 * self.sampling_rate))])

            hi = np.mean(
                s[int(max(0, r - 0.28 * self.sampling_rate)):int(min(len(s), r - 0.05 * self.sampling_rate))])
            out = out + rai / hi

        return (1 / len(peaks)) * out

    def spectral_moment_inner(self, power_spectrum, frequencies, n):
        numerator = np.sum((np.array(frequencies) ** n) * power_spectrum)
        denominator = np.sum(power_spectrum)

        nth_moment = numerator / denominator

        return nth_moment

    def calculate_power_spectrum(self, signal, sampling_rate):
        fft_result = np.fft.fft(signal)

        power_spectrum = np.abs(fft_result) ** 2

        frequencies = np.fft.fftfreq(len(fft_result), 1.0 / sampling_rate)

        return frequencies, power_spectrum

    def spectral_moment(self, signal, order, sampling_freq):

        freq, power = self.calculate_power_spectrum(signal, sampling_freq)

        return self.spectral_moment_inner(power, freq, order)

    def purSQI(self, signal):
        return (self.spectral_moment(signal, 2, self.sampling_rate) ** 2) / \
               (self.spectral_moment(signal, 0, self.sampling_rate)
                * self.spectral_moment(signal, 4, self.sampling_rate))

    def rsdSQI(self, signal, detector_name='engzee'):
        peaks = self.get_peaks(signal, detector_name)
        return self.rsdSQI_inner(signal, peaks)

    def rsdSQI_inner(self, signal, peaks):
        out = 0.0

        for r in peaks:
            ri = np.std(signal[int(r - 0.07 * self.sampling_rate): int(r + 0.08 * self.sampling_rate)])
            ai = np.std(signal[int(r - 0.2 * self.sampling_rate): int(r + 0.2 * self.sampling_rate)])

            out = out + ri / (2 * ai)

        return (1 / len(signal)) * out
        
    def entSQI(self, signal, length = 2, tolerance = 0.2):
        x, _, _ = eh.SampEn(signal, m=length, r=tolerance)
        return x[-1]



    def hfMSQI(self, signal, detector_name='engzee'):
        peaks = self.get_peaks(signal, detector_name)
        return self.hfMSQI_inner(signal, peaks)

    def hfMSQI_inner(self, signal, peaks):        
        b, a = scipy.signal.butter(5, 40, fs = self.sampling_rate, btype='high', analog=False, output='ba')

        filtered = [x**2 for x  in scipy.signal.lfilter(b, a, signal)]
        
        cutoff_frequency = 0.05
        window = 'hamming'
        normalized_cutoff = cutoff_frequency / (0.5 * self.sampling_rate)
        filter_order = 101

        fir_coeff = scipy.signal.firwin(filter_order, normalized_cutoff, window=window, fs=self.sampling_rate)

        filtered = scipy.signal.lfilter(fir_coeff, 1.0, filtered)
        sum = 0
        for r in peaks:
            sum = sum + np.sum(filtered[int(max(0, r - 0.07 * self.sampling_rate)):int(min(len(signal), r + 0.08 * self.sampling_rate))])
        return sum / np.sum(filtered)

    def PiCASQI(self, signal, detector_name='engzee'):
        peaks = self.get_peaks(signal, detector_name)
        return self.PiCASQI_inner(signal, peaks)

    def PiCASQI_inner(self, signal, peaks: list):
        if len(peaks) <= 1:
            return -5
        t = int(np.mean(np.diff(np.array(peaks))))

        return np.abs(average_cross_correlation(signal, t) / average_cross_correlation(signal, 0))
