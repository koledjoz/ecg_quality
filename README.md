Library description
================

This library is meant to be used to check quality of ECG signal. Recordings of any length can be processed, memory is the only limit. The library work with three basic levels of quality:

 - Quality 1 : All important segment of ECG can be detected in the signal
 - Quality 2 : Noise is present, but the QRS complex can still be detected
 - Quality 3 : Noise is too large, not even QRS complex can be detected

Each model works with a sliding window. At each step, the window is evaluated using one of 4 models and one score from range 0 to 1 is returned. The length of sliding windows for individual models can be found in their names. The 4 supported models are:

- cnn2s
- cnn5s
- lstm2s
- oscnn2s

The stride of the sliding window can be specified. The class limits so this value is at least 1 second and is a divisor of the length of the sliding window. 

There are currently 4 return modes supported:
 

 - score : The model returns the score between 0 and 1. 0 is meant to represent quality 1, 0.5 to represent quality 2 and 1 to represent quality 3. This is however not given and the score distributions of different qualities can differ between models.
 - three_value : Three qualities are given using thresholds.
 - bin_clean : Signal is either marked as quality 1 or as quality 2/3
 - bin_qrs: Signal is either marked as quality 1/2 or as quality 3

The thresholds used in each of these modes, except of score, can be specified by the user. However, there are default thresholds, that aim to prioritize precision with the better classes. When using three_value, the returned values are one of 1, 2 and 3, depending on the quality of the signal. When using either of the binary modes, the returned values are either 1 or 2. 1 is for the better quality signal and 2 is for the worse quality.

In terms of length of output, there are two return_types supported:

 -  intervals : value is returned for each window of the size stride, since the value of the quality ill not change in these
 - full : value is returned for each value in the input

Models were trained on cleaned data. These data were cleaned using default NeuroKit2 ecg_clean method. The class can clean signal it is meant to process using ecg_clean. It can be specified not to do this. 

Full explanation of all the modes can be found inside the source code documentation.

Installation
-------------

The library can be easily installed through the pip utility.

		pip install ecg-quality

Usage
-------
The library offers a high-level interface. The simplest example using all the default settings:

	from ecg_quality.ECGQualityChecker import ECGQualityChecker  
  
	checker = ECGQualityChecker()  
	 
	signal = ...	 
  
	output = checker.process_signal(signal)  
  

