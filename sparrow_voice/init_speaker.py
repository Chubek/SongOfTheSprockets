import os

import numpy as np
from scipy.io import wavfile

from sprocket.speech import FeatureExtractor



class InitSpeaker:

    def __init__(self):
        self.progress_init = 0


    def main(self, list_file, work_dir):
        f0s = []
        npows = []        
        bit_depth = None
        fs_ = None
        for f in list_file:
            # open waveform
            f = f.rstrip()
            wavf = os.path.join(work_dir, f + '.wav')
            fs, x = wavfile.read(wavf)
            
            try:
                x = x[:, 0]
            except:
                x = x
                
            bit_depth = x.dtype.itemsize
            fs_ = fs
            x = np.array(x, dtype=np.float)
            print("Extract: " + wavf)

            # constract FeatureExtractor class
            feat = FeatureExtractor(analyzer='world', fs=fs)

            # f0 and npow extraction
            f0, _, _ = feat.analyze(x)
            npow = feat.npow()
            f0s.append(f0)
            npows.append(npow)

            self.progress_init += 1

        f0s = np.hstack(f0s).flatten()
        npows = np.hstack(npows).flatten()
    
        return f0s.tolist(), npows.tolist(), (int(f0s[f0s != 0].min()), int(f0s.max())), int(npows[npows != 0].min()), bit_depth, fs_
