def get_stride_length(input_len, stride_val, sampling_rate):
    divisors = []
    for i in range(0, input_len+1):
        if input_len % i == 0:
            divisors.append(i)
    min_val = min(divisors, key=lambda x:abs(x - sampling_rate))
    stride_num = input_len*stride_val
    closest_stride = min(divisors, key= lambda x: abs(x - stride_num))
    closest_stride = min(closest_stride, min_val)
    return closest_stride


def get_default_thresholds(mode:str):
    if mode == 'two_value':
        return [0.5]
    if mode == 'three_value':
        return [1/3, 2/3]

    return None