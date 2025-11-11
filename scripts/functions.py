import wfdb
import numpy as np


def target_encoder(x):
    """
    Falls eine der Arten von Herzinfarkt in SCP - Code von einer Person aufgelistet wurde, wird das als 1 klassifiziert, sonst 0
    """
    mi = ["IMI", "ASMI", "ILMI", "AMI", "ALMI", "INJAS", "LMI", "INJAL", "IPLMI", "IPMI", "INJIN", "INJLA", "PMI", "INJIL"]
    for element in mi:
        if element in x:
            return 1
    return 0

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data