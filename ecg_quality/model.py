class Model:

    def __int__(self, model:str):
        raise NotImplementedError('This is method of an interface, call class inheriting from this class')

    def process_ecg(self, signal:list):
        raise NotImplementedError('This is method of an interface, call class inheriting from this class')


    def process_ecg_batch(self, batch:np.array):
        if batch.ndim != 2:
            raise ValueError('Batch needs to have at least dimensions, with the first one being used to store other inputs')
        output = []
        for signal in batch:
            output.append(self.process_ecg(signal))
        return output

    def get_input_length(self):
        raise NotImplementedError('This is method of an interface, call class inheriting from this class')