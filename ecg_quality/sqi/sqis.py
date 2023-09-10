from ecgdetectors import Detectors
import scipy
import numpy as np
import math
import neurokit2 as nk


def average_cross_correlation(signal, x):
    signal_array = np.array(signal)
    N = len(signal_array)

    # Create a lagged version of the signal
    lagged_signal = signal_array[:-x]

    # Perform element-wise multiplication and calculate the mean
    avg_corr = np.mean(signal_array[:N - x] * lagged_signal)

    return avg_corr


class SQI_calculator():

    def __int__(self, sampling_rate: int = 250):
        self.detectors = Detectors(sampling_rate)
        self.sampling_rate = sampling_rate

    def get_peaks(self, signal, detector_name):
        if detector_name == 'wqrs':
            return [x for x in self.detectors.wqrs_detector(signal)
                    if 0.3 * self.sampling_rate <= x <= len(signal) - 0.1 * self.sampling_rate]
        if detector_name == 'engzee':
            return [x for x in self.detectors.engzee_detector(signal)
                    if 0.3 * self.sampling_rate <= x <= len(signal) - 0.1 * self.sampling_rate]

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
        return scipy.stats.skewness(signal)

    def kSQI(self, signal: list):
        return scipy.stats.kurtosis(signal)

    def pSQI(self, signal):
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

        b, a = scipy.signal.butter(5, 1.0, self.sampling_rate, btype='low', analog=False, output='ba')

        for r in peaks:
            rai = max(
                signal[int(max(0, r - 0.07 * self.sampling_rate)):min(len(signal), r + 0.08 * self.sampling_rate)])
            bai = max(scipy.signal.lfilter(b, a,
                                           signal[max(0, int(r - 1 * self.sampling_rate)):min(len(signal),
                                                                                              int(r + 1 * self.sampling_rate))]))

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
        b = [1, -2, 1]
        a = []

        y = scipy.signal.lfilter(b, a, signal)

        s = [0] * 5 + np.convolve(y, np.ones(6, dtype='float'), 'valid')

        out = 0.0
        for r in peaks:
            rai = max(
                signal[int(max(0, r - 0.07 * self.sampling_rate)):min(len(signal), r + 0.08 * self.sampling_rate)])

            hi = np.mean(
                signal[int(max(0, r - 0.28 * self.sampling_rate)):min(len(signal), r - 0.05 * self.sampling_rate)])

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

    def spectral_moment(self, signal, order, sampling_frequency):

        freq, power = self.calculate_power_spectrum(signal, sampling_frequency)

        return self.spectral_moment_inner(power, freq, order)

    def purSQI(self, signal):
        return (self.spectral_moment(signal, 2, self.sampling_frequency) ** 2) / \
               (self.spectral_moment(signal, 0, self.sampling_frequency)
                * self.spectral_moment(signal, 4, self.sampling_frequency))

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

    def entSQI(self, signal, length, tolerance):
        b = np.lib.stride_tricks.sliding_window_view(signal, length)
        a = np.lib.stride_tricks.sliding_window_view(signal, length + 1)
        amr = 0.0
        bmr = 0.0
        for data in a:
            amr = amr + len(list(filter(lambda x: 0 < np.linalg.norm(data - x) <= tolerance, a))) * len(
                signal - length - 1) ** (-1)

        amr = amr * len(signal - length) ** (-1)

        for data in b:
            bmr = bmr + len(list(filter(lambda x: 0 < np.linalg.norm(data - x) <= tolerance, b))) * len(
                signal - length - 1) ** (-1)

        bmr = bmr * len(signal - length) ** (-1)
        return -math.log(amr / bmr)

    def PiCASQI(self, signal, detector_name='engzee'):
        peaks = self.get_peaks(signal, detector_name)
        return self.PiCASQI_inner(signal, peaks)

    def PiCASQI(self, signal, peaks: list):
        rate = np.mean(nk.ecg_rate(peaks, sampling_rate=self.sampling_rate))

        # toto su udery za minutu - z tohto ziskaj dsitance medzi nimi

        t = (self.sampling_rate * 60) / rate

        return np.abs(average_cross_correlation(signal, t) / average_cross_correlation(signal, 0))
