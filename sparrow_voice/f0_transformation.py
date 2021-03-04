import numpy as np
from scipy.io import wavfile
import os
from sprocket.model import F0statistics
from sprocket.speech import FeatureExtractor, Shifter
from .misc import low_cut_filter
from .yml import SpeakerYML


class F0Transform:

    def __init__(self):
        self.progress = 0
        self.finalf0 = 0
        self.wavfs_wrong = False
        self.error_file = None


    def get_f0s_from_list(self, conf, list_file, work_dir):
    
        f0s = []
        for i, f in enumerate(list_file):
            current_file = f + '.wav'
            f = f.rstrip()
            wavf = os.path.join(work_dir, f + '.wav')
            fs, x = wavfile.read(wavf)

            try:
                x = x[:, 0]
            except:
                x = x

            x = np.array(x, dtype=np.float)
            x = low_cut_filter(x, fs, cutoff=70)
            
            if fs != conf.wav_fs:
                self.wavfs_wrong = True
                self.error_file = f

            assert fs == conf.wav_fs

            print("Extract F0: " + wavf)

            feat = FeatureExtractor(analyzer=conf.analyzer, fs=conf.wav_fs,
                                    fftl=conf.wav_fftl, shiftms=conf.wav_shiftms,
                                    minf0=conf.f0_minf0, maxf0=conf.f0_maxf0)
            f0 = feat.analyze_f0(x)
            f0s.append(f0)

            self.progress += 1

        return f0s


    def transform_f0_from_list(self, speaker, f0rate, wav_fs, list_file, work_dir):
        # Construct Shifter class
        shifter = Shifter(wav_fs, f0rate=f0rate)

        # check output directory
        transformed_wavdir = os.path.join(work_dir, speaker + '_' + str(f0rate))
        if not os.path.exists(transformed_wavdir):
            os.makedirs(transformed_wavdir)

        for f in list_file:
            self.progress += 1
            # open wave file
            f = f.rstrip()
            wavf = os.path.join(work_dir, f + '.wav')

            # output file path
            transformed_wavpath = os.path.join(
                transformed_wavdir, os.path.basename(wavf))

            # flag for completion of high frequency range
            if f0rate < 1.0:
                completion = True
            else:
                completion = False
            if not os.path.exists(transformed_wavpath):
                # transform F0 of waveform
                fs, x = wavfile.read(wavf)

                try:
                    x = x[:, 0]
                except:
                    x = x

                x = np.array(x, dtype=np.float)
                x = low_cut_filter(x, fs, cutoff=70)

                if fs != wav_fs:
                    self.wavfs_wrong = True
                    self.error_file = f

                assert fs == wav_fs
                transformed_x = shifter.f0transform(x, completion=completion)

                wavfile.write(transformed_wavpath, fs,
                            transformed_x.astype(np.int16))
                print('F0 transformed wav file: ' + transformed_wavpath)
            else:
                print('F0 transformed wav file already exists: ' + transformed_wavpath)




        
    def f0t_main(self, f0rate, speaker, org_yml, tar_yml, org_train_list, tar_train_list, org_eval_list, work_dir):
        # read parameters from speaker yml
        org_conf = SpeakerYML(org_yml)
        tar_conf = SpeakerYML(tar_yml)

        if f0rate == -1:
            # get f0 list to calculate F0 transformation ratio
            org_f0s = self.get_f0s_from_list(
                org_conf, org_train_list, work_dir)
            tar_f0s = self.get_f0s_from_list(
                tar_conf, tar_train_list, work_dir)

            # calculate F0 statistics of original and target speaker
            f0stats = F0statistics()
            orgf0stats = f0stats.estimate(org_f0s)
            tarf0stats = f0stats.estimate(tar_f0s)

            # calculate F0 transformation ratio between original and target speakers
            f0rate = np.round(np.exp(tarf0stats[0] - orgf0stats[0]), decimals=2)
            self.finalf0 = f0rate
        else:
            f0rate = f0rate
        print('F0 transformation ratio: ' + str(f0rate))

        self.transform_f0_from_list(speaker, f0rate, org_conf.wav_fs,
                                org_train_list, work_dir)
        self.transform_f0_from_list(speaker, f0rate, org_conf.wav_fs,
                                org_eval_list, work_dir)

        return f0rate
    