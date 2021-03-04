import yaml
import os

def create_default_pair(n_iter_jnt, n_mix_mcep, n_iter_mcep, n_mix_codeap, n_iter_codeap, morph_coeff, work_path, source_speaker, target_speaker):
    pair_default = {
        'jnt': {
            'n_iter': n_iter_jnt
        },
        'GMM': {
            'mcep': {
                'n_mix': n_mix_mcep,
                'n_iter': n_iter_mcep,
                'covtype': 'full',
                'cvtype': 'mlpg'
            },
            'codeap': {
                'n_mix': n_mix_codeap,
                'n_iter': n_iter_codeap,
                'covtype': 'full',
                'cvtype': 'mlpg'
            }
        },
        'GV': {
            'morph_coeff': morph_coeff
        }
    }

    if not os.path.exists(os.path.join(work_path, "conf", "pair")):
        os.makedirs(os.path.join(work_path, "conf", "pair"))

    yaml.dump(pair_default, open(os.path.join(work_path, "conf", "pair", f"{source_speaker}-{target_speaker}.yml"), "w"))



def create_speaker(work_path, speaker, fs, bit, minf0, maxf0, threshold):
    speaker_conf = {
         'wav': {
            'fs': fs,
            'bit': bit,
            'fftl': 1024,
            'shiftms': 5
            },

        'f0': {
            'minf0': minf0,
            'maxf0': maxf0
        },
        'mcep':  {
            'dim': 24,
            'alpha': 0.410
        },
        'power': {
            'threshold': threshold
            },
        'analyzer': 'world'
    }

    if not os.path.exists(os.path.join(work_path, "conf", "speaker")):
        os.makedirs(os.path.join(work_path, "conf", "speaker"))
    
    yaml.dump(speaker_conf, open(os.path.join(work_path, "conf", "speaker", f"{speaker}.yml"), "w"))


