import numpy as np

vocab = np.load('vocabulary.npz', allow_pickle=True)
int_to_events = vocab['int_to_events'].item()
events_to_int = vocab['events_to_int'].item()
timeshift_event = 'timeshift_10'
TIMESHIFT_RES = vocab['timeshift_res'].item()  # 10ms
SEQ_LEN = 255