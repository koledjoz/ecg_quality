import os.path

MODEL_PATH_DICT = {'lstm2s' : 'LSTM2s.h5', 'cnn2s' : 'CNN2s.h5', 'cnn5s' : 'CNN5s.h5', 'oscnn2s' : 'OSCNN2s.h5'}

MODEL_WIN_LEN = {'lstm2s' : 2, 'cnn2s' : 2, 'cnn5s' : 5, 'oscnn2s' : 2}

MODEL_THRESHOLDS = {'lstm2s' : [0.125, 0.5], 'cnn2s' : [0.1, 0.625], 'cnn5s' : [0.2, 0.525], 'oscnn2s' : [0.1, 0.7]}

file_path = os.path.dirname(__file__)
lib_path = os.path.dirname(file_path)



def get_stride_length(input_len, stride_val, sampling_rate):
    divisors = []
    for i in range(1, input_len+1):
        if input_len % i == 0:
            divisors.append(i)
    min_val = min(divisors, key=lambda x:abs(x - sampling_rate))
    stride_num = input_len*stride_val
    closest_stride = min(divisors, key= lambda x: abs(x - stride_num))
    closest_stride = max(closest_stride, min_val)
    return closest_stride

def get_default_thresholds(model:str, mode:str):
    if mode == 'binary_clean':
        return MODEL_THRESHOLDS[model][0]
    if mode == 'binary_qrs':
        return MODEL_THRESHOLDS[model][1]
    if mode == 'three_value':
        return MODEL_THRESHOLDS[model]
    return None

