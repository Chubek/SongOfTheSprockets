import numpy as np
from scipy.io import wavfile

from sprocket.speech import FeatureExtractor, Synthesizer
from sprocket.util import HDF5
import os
from .misc import low_cut_filter
from .yml import SpeakerYML


class ExtractFeatures:

    def __init__(self):
        self.ef_progress = 0

    def ef_main(self, overwrite, speaker, ymlf, list_file, pair_dir, work_path):
        # read parameters from speaker yml
        sconf = SpeakerYML(ymlf)
        h5_dir = os.path.join(pair_dir, 'h5')
        anasyn_dir = os.path.join(pair_dir, 'anasyn')
        if not os.path.exists(os.path.join(h5_dir, speaker)):
            os.makedirs(os.path.join(h5_dir, speaker))
        if not os.path.exists(os.path.join(anasyn_dir, speaker)):
            os.makedirs(os.path.join(anasyn_dir, speaker))

        # constract FeatureExtractor class
        feat = FeatureExtractor(analyzer=sconf.analyzer,
                                fs=sconf.wav_fs,
                                fftl=sconf.wav_fftl,
                                shiftms=sconf.wav_shiftms,
                                minf0=sconf.f0_minf0,
                                maxf0=sconf.f0_maxf0)

        # constract Synthesizer class
        synthesizer = Synthesizer(fs=sconf.wav_fs,
                                fftl=sconf.wav_fftl,
                                shiftms=sconf.wav_shiftms)

        # open list file
        with open(list_file, 'r') as fp:
            for line in fp:
                f = line.rstrip()
                self.ef_progress += 1
                h5f = os.path.join(h5_dir, f + '.h5')

                if (not os.path.exists(h5f)) or overwrite:
                    wavf = os.path.join(work_path, f + '.wav')
                    fs, x = wavfile.read(wavf)
                    x = np.array(x, dtype=np.float)
                    x = low_cut_filter(x, fs, cutoff=70)
                    assert fs == sconf.wav_fs

                    print("Extract acoustic features: " + wavf)

                    # analyze F0, spc, and ap
                    f0, spc, ap = feat.analyze(x)
                    mcep = feat.mcep(dim=sconf.mcep_dim, alpha=sconf.mcep_alpha)
                    npow = feat.npow()
                    codeap = feat.codeap()

                    # save features into a hdf5 file
                    h5 = HDF5(h5f, mode='a')
                    h5.save(f0, ext='f0')
                    # h5.save(spc, ext='spc')
                    # h5.save(ap, ext='ap')
                    h5.save(mcep, ext='mcep')
                    h5.save(npow, ext='npow')
                    h5.save(codeap, ext='codeap')
                    h5.close()

                    # analysis/synthesis using F0, mcep, and ap
                    wav = synthesizer.synthesis(f0,
                                                mcep,
                                                ap,
                                                alpha=sconf.mcep_alpha,
                                                )
                    wav = np.clip(wav, -32768, 32767)
                    anasynf = os.path.join(anasyn_dir, f + '.wav')
                    wavfile.write(anasynf, fs, np.array(wav, dtype=np.int16))
                else:
                    print("Acoustic features already exist: " + h5f)