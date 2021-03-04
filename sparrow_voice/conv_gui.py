from subprocess import call
from dearpygui.core import *
from dearpygui.simple import *
from random import randint
import json
import os
import glob
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
import time
from shutil import copy2
import sklearn
from .create_conf import create_default_pair, create_speaker
from .estimate_feature_statistics import efs_main
from .estimate_twf_and_jnt import ETJ
from .f0_transformation import F0Transform
from .init_speaker import InitSpeaker
from .extract_features import ExtractFeatures
from .train_GMM import main as train_main
from .convert import Convert

import threading


main_data = {
    "src_files": [],
    "tgt_files": [],
    "conv_files": [],
    "source_path": "",
    "target_path": "",
    "conv_path": "",
    "work_path": "",
    "src_full": [],
    "tgt_full": [],
    "conv_full": [],
    "source_speaker": "",
    "target_speaker": "",
    "pair_dir": "",
    "f0_ratios": [],
    "conf_dir": "",
    "files_copied": False,
    "conf_pair_created": False,
    "conf_speaker_created": False,
    "speaker_f0s": [],
    "target_f0s": [],
    "speaker_pow": [],
    "target_pow": [],
    "allow_plot": False,
    "min_max_source": (0, 0),
    "min_max_target": (0, 0),
    "min_pow_source": 0,
    "min_pow_target": 0,
    "n_iter_jnt": 3,
    "n_mix_mcep": 32,
    "n_mix_codeap": 16,
    "n_iter_codeap": 100,
    "morph_coeff": 1.0,
    "source_fs": 0,
    "target_fs": 0,
    "source_bitd": 0,
    "target_bitd": 0,
    "workpath_selected": False,
    "convpath_selected": False,
    "srcpath_selected": False,
    "tgtpath_selected": False,
    "f0trans_done": False,
    "efs_done": False,
    "ef_done": False,
    "etj_done": False,
    "train_done": False,
    "conversion_done": False,
    "enable_main_conv": True,
    "enable_conv_1": True,
    "enable_conv_2": False,
    "enable_conv_3": False,
    "enable_conv_4": False,
    "enable_conv_5": False
}
init_speaker = InitSpeaker()
f0trans = F0Transform()
ef = ExtractFeatures()
etj = ETJ()
conv = Convert()

def main():

    def split_silence():
        file_loc = get_value("File Location")
        save_loc = get_value("Save Location")
        ms = get_value("Silence Length Threshold")
        pow = get_value("Silence Power Threshold")

        audio_segment = AudioSegment.from_wav(file_loc)  
        chunks = split_on_silence(audio_segment, min_silence_len=ms, silence_thresh=pow, keep_silence=200)

        for i, chunk in enumerate(chunks):
            chunk.export(save_loc, f"{i}.wav", format="wav")
            set_value('Split Progress', i / len(chunks))
    
    def set_split_save(sender, data):
        directory = data[0]
        set_value("Save Location", directory)

    def get_split_file(sender, data):
        file = data[1]
        directory = data[0]

        set_value("File Location", os.path.join(directory, file))

    def reset_vals():
        set_value("File Location", "")
        set_value("Save Location", "")

    def fe_progress_bar():
        configure_item("Wrong FS", show=False)
        configure_item("Feature Extraction Progress", show=True)
        max_value = len(main_data["src_files"]) + len(main_data["tgt_files"])
        while True:
            if init_speaker.progress_init >= max_value:
                break
            set_value("Feature Extraction Progress", (init_speaker.progress_init / max_value))
        configure_item("Feature Extraction Progress", show=False)

    def f0_progress_bar():
        configure_item("F0 Transform Progress", show=True)
        configure_item("Wrong FS", show=False)
        set_value("Wrong FS", "")
        max_value = (len(main_data["src_files"]) * 2) + len(main_data["tgt_files"]) + len(main_data["conv_files"])
        while True:
            if f0trans.progress >= max_value:
                set_value("F0 Transform Progress", 1)
                break
            if f0trans.wavfs_wrong == True:
                configure_item("Wrong FS", show=True)
                set_value("Wrong FS", f"The file {f0trans.error_file} has the wrong Sample Rate!")
                break

            set_value("F0 Transform Progress", (f0trans.progress / max_value))
        configure_item("F0 Transform Progress", show=False)

    def ef_progress():
        configure_item("EF Progress", show=True)

        max_value = len(main_data["src_files"]) + len(main_data["tgt_files"])

        while True or not main_data["ef_done"]:
            if ef.ef_progress >= max_value:
                set_value("EF Progress", 1)
                break

            set_value("EF Progress", (ef.ef_progress / max_value))

    def efs_progress():
        configure_item("EFS Progress", show=True)

        t_1 = time.time()

        divisor = 0.01

        while not main_data["efs_done"]:
            t_2 = time.time()
            time.sleep(10)
            difference = t_2 - t_1
            if difference % divisor >= 1:
                divisor /= 10
  

                divisor += 0.0001

            set_value("EFS Progress", difference % divisor)

        set_value("EFS Progress", 1.0) 


    def etj_progress():
        configure_item("ETJ Progress", show=True)

        max_value = len(main_data["src_files"])

        while True or not main_data["etj_done"]:
            if etj.progress >= max_value:
                set_value("ETJ Progress", 1)
                break

            set_value("ETJ Progress", (etj.progress / max_value))

    def train_progress():
        configure_item("Training Progress", show=True)

        t_1 = time.time()

        divisor = 0.01

        while not main_data["efs_done"]:
            t_2 = time.time()
            time.sleep(10)
            difference = t_2 - t_1
            if difference % divisor >= 1:
                divisor /= 10
  

                divisor += 0.0001

            set_value("Training Progress", difference % divisor)

        set_value("Training Progress", 1.0) 

    def conv_progress():
       
        configure_item("Conversion Progress", show=True)
        
        max_value = len(main_data["conv_files"])

        if get_value('Do Non-Diff GMM as Well?'):
            max_value *= 2

        while True or not main_data["conversion_done"]:
            if conv.progress >= max_value:
                set_value("Conversion Progress", 1)
                break

            set_value("Conversion Progress", (conv.progress / max_value))

    def f0_transform():
        configure_item("Transformed", show=False)
        f0trans.wavfs_wrong = False

        if get_value("Calculate Automatically (Recommended)"):
            f0rate = -1
        
        f0rate = get_value("F0 Ratio")
        speaker = main_data["source_speaker"]
        org_yml = os.path.join(main_data["work_path"], "conf", "speaker", main_data["source_speaker"] + ".yml")
        tar_yml = os.path.join(main_data["work_path"], "conf", "speaker", main_data["target_speaker"] + ".yml")
        org_train_list = main_data["src_files"]
        tar_train_list = main_data["tgt_files"]
        org_eval_list = main_data["conv_files"]
        work_dir = main_data["work_path"]

        f0trans.f0t_main(f0rate, speaker, org_yml, tar_yml, org_train_list, tar_train_list, org_eval_list, work_dir)

        f0rate = f0trans.finalf0

        main_data["f0_ratios"].append(f0rate)
        configure_item("Transformed", show=True)
        set_value("Transformed", f"Transformed with F0 value {f0rate}.")
        configure_item("Select F0 Ratio", items=main_data["f0_ratios"])
        main_data["f0trans_done"] = True
        json.dump(main_data, open(os.path.join(main_data["work_path"], "state.conv"), "w"))


    def conv_step_1():
        configure_item('Step 1: Feature Extraction', enabled=False)

        f0 = main_data["f0_ratios"][get_value("Select From F0 Ratios")]

        src_files_split = (d.split("\\") for d in main_data['src_files'])
        src_files = [f"{f}_{f0}\\{d}" for f, d in src_files_split]

        overwrite = get_value("Overwrite Saved File?")
        
        org_yml = os.path.join(main_data["work_path"], "conf", "speaker", main_data["source_speaker"] + ".yml")
        tar_yml = os.path.join(main_data["work_path"], "conf", "speaker", main_data["target_speaker"] + ".yml")

        ef.ef_main(overwrite, main_data["source_speaker"], org_yml, src_files, os.path.join(main_data["work_path"], "pair"), main_data["work_path"])
        ef.ef_main(overwrite, main_data["target_speaker"], tar_yml, main_data["tgt_files"], os.path.join(main_data["work_path"], "pair"), main_data["work_path"])

        main_data["ef_done"] = True
        configure_item("EF Done", show=True)
        main_data["enable_conv_2"] = True
        configure_item('Step 2: Statistical Feature Extraction', enabled=True)
        configure_item('Step 1: Feature Extraction', enabled=True)
        json.dump(main_data, open(os.path.join(main_data["work_path"], "state.conv"), "w"))


    def conv_step_2():
        configure_item('Step 2: Statistical Feature Extraction', enabled=False)
        f0 = main_data["f0_ratios"][get_value("Select From F0 Ratios")]

        src_files_split = (d.split("\\") for d in main_data['src_files'])
        src_files = [f"{f}_{f0}\\{d}" for f, d in src_files_split]

        efs_main(main_data["source_speaker"], src_files, os.path.join(main_data["work_path"], "pair"))
        efs_main(main_data["target_speaker"], main_data["tgt_files"], os.path.join(main_data["work_path"], "pair"))

        main_data["efs_done"] = True
        configure_item("EFS Done", show=True)
        main_data["enable_conv_3"] = True
        configure_item('Step 3: ETJ Extraction', enabled=True)
        configure_item('Step 2: Statistical Feature Extraction', enabled=True)
        json.dump(main_data, open(os.path.join(main_data["work_path"], "state.conv"), "w"))

    def conv_step_3():
        configure_item('Step 3: ETJ Extraction', enabled=False)
        f0 = main_data["f0_ratios"][get_value("Select From F0 Ratios")]

        src_files_split = (d.split("\\") for d in main_data['src_files'])
        src_files = [f"{f}_{f0}\\{d}" for f, d in src_files_split]
        org_yml = os.path.join(main_data["work_path"], "conf", "speaker", main_data["source_speaker"] + ".yml")
        tar_yml = os.path.join(main_data["work_path"], "conf", "speaker", main_data["target_speaker"] + ".yml")
        pair_yml = os.path.join(main_data["work_path"], "conf", "pair", main_data["source_speaker"] + "-" + main_data["target_speaker"] + ".yml")

        configure_item('Step 3: ETJ Extraction', enabled=True)
        configure_item('Step 4: Training', enabled=True)
        main_data["enable_conv_4"] = True
        etj.etj_main(org_yml, tar_yml, pair_yml, src_files, os.path.join(main_data["work_path"], "pair"))

        
        main_data["etj_done"] = True
        configure_item("ETJ Done", show=True)
        json.dump(main_data, open(os.path.join(main_data["work_path"], "state.conv"), "w"))

    def pop_list(list, list2, index):
        list.pop(index)
        list2.pop(index)

        return list

    def conv_step_4():
        configure_item('Step 4: Training', enabled=False)

        f0 = main_data["f0_ratios"][get_value("Select From F0 Ratios")]

        src_files_split = (d.split("\\") for d in main_data['src_files'])
        src_files = [f"{f}_{f0}\\{d}" for f, d in src_files_split]
        pair_yml = os.path.join(main_data["work_path"], "conf", "pair", main_data["source_speaker"] + "-" + main_data["target_speaker"] + ".yml")

        train_main(src_files, pair_yml, os.path.join(main_data["work_path"], "pair"))
        
        configure_item('Step 4: Training', enabled=True)
        configure_item('Step 5: Conversion', enabled=True)
        main_data["train_done"] = True
        configure_item("Training Done", show=True)
        main_data["enable_conv_5"] = True
        json.dump(main_data, open(os.path.join(main_data["work_path"], "state.conv"), "w"))

    
    def conv_step_5():
        configure_item('Step 5: Conversion', enabled=False)
        f0 = main_data["f0_ratios"][get_value("Select From F0 Ratios")]
        org = f"{main_data['source_speaker']}_{f0}"
        conv_files_split = (d.split("\\") for d in main_data['conv_files'])
        conv_files = [f"{f}_{f0}\\{d}" for f, d in conv_files_split]
        pair_yml = os.path.join(main_data["work_path"], "conf", "pair", main_data["source_speaker"] + "-" + main_data["target_speaker"] + ".yml")
        org_yml = os.path.join(main_data["work_path"], "conf", "speaker", main_data["source_speaker"] + ".yml")

        if get_value('Do Non-Diff GMM as Well?'):
            conv.main(None, org, main_data["target_speaker"], org_yml, pair_yml, conv_files, main_data["work_path"],\
                 os.path.join(main_data["work_path"], "pair"), get_value("Save Path"))

        conv.main('diff', org, main_data["target_speaker"], org_yml, pair_yml, conv_files, main_data["work_path"],\
                 os.path.join(main_data["work_path"], "pair"), get_value("Save Path"))

        main_data["conversion_done"] = True
        configure_item("Conversion Done", show=True)

        configure_item('Step 5: Conversion', enabled=True)

        json.dump(main_data, open(os.path.join(main_data["work_path"], "state.conv"), "w"))

        

    def copy_files():
        configure_item("Copy Progress", show=True)

        src_full, tgt_full, conv_full = main_data["src_full"].copy(), main_data["tgt_full"].copy(), main_data["conv_full"].copy()

        if not os.path.exists(os.path.join(main_data["work_path"], main_data["source_speaker"])):
            os.makedirs(os.path.join(main_data["work_path"], main_data["source_speaker"]))
            print(os.path.join(main_data["work_path"], main_data["source_speaker"]) + " created.")
        if not os.path.exists(os.path.join(main_data["work_path"], main_data["target_speaker"])):
            os.makedirs(os.path.join(main_data["work_path"], main_data["target_speaker"]))
            print(os.path.join(main_data["work_path"], main_data["target_speaker"]) + " created.")
        if not os.path.exists(os.path.join(main_data["work_path"], "conf", "speaker")):
            os.makedirs(os.path.join(main_data["work_path"], "conf", "speaker"))
            print(os.path.join(main_data["work_path"], "conf", "speaker") + " created.")
        if not os.path.exists(os.path.join(main_data["work_path"], "conf", "pair")):
            os.makedirs(os.path.join(main_data["work_path"], "conf", "pair"))
            print(os.path.join(main_data["work_path"], "conf", "pair") + " created.")
        if not os.path.exists(os.path.join(main_data["work_path"], "pair")):
            os.makedirs(os.path.join(main_data["work_path"], "pair"))
            print(os.path.join(main_data["work_path"], "pair") + " created.")

        prog_full = len(src_full) + len(tgt_full) + len(conv_full)
        
        for i, src in enumerate(main_data["src_full"]):
            copy2(src, os.path.join(main_data["work_path"], main_data["src_files"][i] + ".wav"))
            del src_full[-1]
            prog_num = 2 - (((len(src_full) + len(tgt_full) + len(conv_full)) / prog_full) * 2)
            set_value("Copy Progress", prog_num)

        for i, src in enumerate(main_data["tgt_full"]):
            copy2(src, os.path.join(main_data["work_path"], main_data["tgt_files"][i] + ".wav"))
            del tgt_full[-1]
            prog_num = 2 - (((len(src_full) + len(tgt_full) + len(conv_full)) / prog_full) * 2)
            set_value("Copy Progress", prog_num)

        
        for i, src in enumerate(main_data["conv_full"]):
            copy2(src, os.path.join(main_data["work_path"], main_data["conv_files"][i] + ".wav"))
            del conv_full[-1]
            prog_num = 2 - (((len(src_full) + len(tgt_full) + len(conv_full)) / prog_full) * 2)
            set_value("Copy Progress", prog_num)

        
        configure_item("Copy Progress", show=False)
        main_data["files_copied"] = True
        configure_item("Create Configuration Files ##button", enabled=main_data["files_copied"])

        json.dump(main_data, open(os.path.join(main_data["work_path"], "state.conv"), "w"))


    def create_conf_pair():    
    
        main_data["n_iter_jnt"] = get_value("n_iter_jnt")
        main_data["n_mix_mcep"] = get_value("n_mix_mcep")
        main_data["n_mix_codeap"] =  get_value("n_iter_mcep")
        main_data["n_iter_codeap"] = get_value("n_iter_codeap")
        main_data["morph_coeff"] = get_value("morph_coeff")

        create_default_pair(get_value("n_iter_jnt"), \
            get_value("n_mix_mcep"),\
                get_value("n_iter_mcep"), \
                    get_value("n_mix_codeap"), \
                        get_value("n_iter_codeap"), \
                            get_value("morph_coeff"), main_data["work_path"], main_data["source_speaker"], main_data["target_speaker"])

        create_default_pair(get_value("n_iter_jnt"), \
            get_value("n_mix_mcep"),\
                get_value("n_iter_mcep"), \
                    get_value("n_mix_codeap"), \
                        get_value("n_iter_codeap"), \
                            get_value("morph_coeff"), main_data["work_path"], main_data["target_speaker"],  main_data["source_speaker"])

        main_data["conf_pair_created"] = True
        configure_item("Pair Files [Already] Ceated.", show=main_data["conf_pair_created"])
        configure_item("F0 Transformation ##button", enabled=main_data["conf_pair_created"] & main_data['conf_speaker_created'])

        json.dump(main_data, open(os.path.join(main_data["work_path"], "state.conv"), "w"))


    def create_speaker_conf():    

        f0s_source, pows_source, min_max_source, pow_min_source, \
                                                        bit_depth_source, fs_source = init_speaker.main(main_data["src_files"], main_data["work_path"])
        f0s_target, pows_target, min_max_target, pow_min_target,\
                                                        bit_depth_target, fs_target = init_speaker.main(main_data["tgt_files"], main_data["work_path"])

        main_data["speaker_f0s"], main_data["target_f0s"] = f0s_source, f0s_target
        main_data["speaker_pow"], main_data["target_pow"] = pows_source, pows_target
        main_data["min_max_source"], main_data["min_max_target"] = min_max_source, min_max_target
        main_data["min_pow_source"], main_data["min_pow_target"] = pow_min_source, pow_min_target
        main_data["source_fs"], main_data["target_fs"] = fs_source, fs_target
        main_data["source_bitd"], main_data["target_bitd"] = bit_depth_source, bit_depth_target
            
        create_speaker(main_data["work_path"], main_data["source_speaker"], fs_source, bit_depth_source,\
            min_max_source[0], min_max_source[1], pow_min_source)
        create_speaker(main_data["work_path"], main_data["target_speaker"], fs_target, bit_depth_target,\
            min_max_target[0], min_max_target[1], pow_min_target)


        main_data["allow_plot"] = True
        main_data["conf_speaker_created"] = True
        configure_item("Display Plots", enabled=main_data["allow_plot"])
        configure_item("Speaker Files [Already] Ceated.", show=main_data["conf_speaker_created"])

        set_value("Min/Max F0 SRC", main_data["min_max_source"])
        set_value("Min/Max F0 TGT", main_data["min_max_target"])
        set_value("Threshold SRC", main_data["min_pow_source"])
        set_value("Threshold TGT", main_data["min_pow_target"])
        set_value("Frame Rate SRC", main_data["source_fs"])
        set_value("Frame Rate TGT", main_data["target_fs"])
        set_value("Bitdepth SRC", main_data["source_bitd"])
        set_value("Bitdepth TGT",  main_data["target_bitd"])

        set_value("SRC F0", main_data["speaker_f0s"])
        set_value("SRC POW", main_data["speaker_pow"])
        set_value("TGT F0", main_data["target_f0s"])
        set_value("TGT POW", main_data["target_pow"])

        configure_item("F0 Transformation ##button", enabled=main_data["conf_pair_created"] & main_data['conf_speaker_created'])

        json.dump(main_data, open(os.path.join(main_data["work_path"], "state.conv"), "w"))


    def create_speaker_progress():
        jobs = []

        T1 = threading.Thread(target=create_speaker_conf)

        T2 = threading.Thread(target=fe_progress_bar)

        T1.start()
        print("Thread1 started")
        T2.start()
        print("Thread2 started")
        T1.join()
        print("Thread1 joined")
        T2.join()
        print("Thread2 joined")

    def create_f0_progress():
        jobs = []

        T1 = threading.Thread(target=f0_transform)

        T2 = threading.Thread(target=f0_progress_bar)

        T1.start()
        print("Thread1 started")
        T2.start()
        print("Thread2 started")
        T1.join()
        print("Thread1 joined")
        T2.join()
        print("Thread2 joined")

    def create_conv_1():
        jobs = []

        T1 = threading.Thread(target=conv_step_1)

        T2 = threading.Thread(target=ef_progress)

        T1.start()
        print("Thread1 started")
        T2.start()
        print("Thread2 started")
        T1.join()
        print("Thread1 joined")
        T2.join()
        print("Thread2 joined")

    def create_conv_2():
        jobs = []

        T1 = threading.Thread(target=conv_step_2)

        T2 = threading.Thread(target=efs_progress)

        T1.start()
        print("Thread1 started")
        T2.start()
        print("Thread2 started")
        T1.join()
        print("Thread1 joined")
        T2.join()
        print("Thread2 joined")

    def create_conv_3():
        jobs = []

        T1 = threading.Thread(target=conv_step_3)

        T2 = threading.Thread(target=etj_progress)

        T1.start()
        print("Thread1 started")
        T2.start()
        print("Thread2 started")
        T1.join()
        print("Thread1 joined")
        T2.join()
        print("Thread2 joined")

    def create_conv_4():
        jobs = []

        T1 = threading.Thread(target=conv_step_4)

        T2 = threading.Thread(target=train_progress)

        T1.start()
        print("Thread1 started")
        T2.start()
        print("Thread2 started")
        T1.join()
        print("Thread1 joined")
        T2.join()
        print("Thread2 joined")

    def create_conv_5():
        jobs = []

        T1 = threading.Thread(target=conv_step_5)

        T2 = threading.Thread(target=conv_progress)

        T1.start()
        print("Thread1 started")
        T2.start()
        print("Thread2 started")
        T1.join()
        print("Thread1 joined")
        T2.join()
        print("Thread2 joined")

    


    def recreate_speaker_conf():
            
        main_data["min_max_source"], main_data["min_max_target"] = get_value("Min/Max F0 SRC"), get_value("Min/Max F0 TGT")
        main_data["min_pow_source"], main_data["min_pow_target"] = get_value("Threshold SRC"), get_value("Threshold TGT")
        main_data["source_fs"], main_data["target_fs"] = get_value("Frame Rate SRC"), get_value("Frame Rate TGT")
        main_data["source_bitd"], main_data["target_bitd"] = get_value("Bitdepth SRC"), get_value("Bitdepth TGT")

        create_speaker(main_data["work_path"], main_data["source_speaker"], main_data["source_fs"],main_data["source_bitd"],\
            main_data["min_max_source"][0],  main_data["min_max_source"][1], main_data["min_pow_source"])
        create_speaker(main_data["work_path"], main_data["target_speaker"], main_data["target_fs"],main_data["target_bitd"],\
            main_data["min_max_target"][0],  main_data["min_max_target"][1], main_data["min_pow_target"])

        json.dump(main_data, open(os.path.join(main_data["work_path"], "state.conv"), "w"))



    def apply_selected_train_source(sender, data):
        directory = data[0]
        main_data["source_path"] = directory
        main_data["src_full"] =   [str(p) for p in Path(data[0]).iterdir() if p.suffix == ".wav"]

        main_data["source_speaker"] = get_value("Source Speaker Name")

        if len(main_data["source_speaker"]) < 1:
            configure_item("Please enter the source and/or target speaker name", show=True)
            return
        else:
            configure_item("Please enter the source and/or target speaker name", show=False)


        main_data["src_files"] = [main_data["source_speaker"] + "\\" + p.stem for p in Path(data[0]).iterdir() if p.suffix == ".wav"]

        if not main_data["src_files"]:
            configure_item("Source list empty (no wav files).", show=True)
        else:
            configure_item("Source list empty (no wav files).", show=False)

        configure_item("Train Source List", items=main_data["src_files"])
        set_value("train_path_source", directory)

        main_data["src_files"].sort()
        main_data["src_full"].sort()    
        
        main_data["srcpath_selected"] = True
        configure_item("Copy and Create", enabled=main_data["workpath_selected"] &\
                                                                main_data["convpath_selected"] &\
                                                                        main_data["srcpath_selected"] & \
                                                                                main_data["tgtpath_selected"])


    def apply_selected_train_target(sender, data):
        directory = data[0]
        main_data["target_path"] = directory

        main_data["tgt_full"] = [str(p) for p in Path(data[0]).iterdir() if p.suffix == ".wav"]

        main_data["target_speaker"] = get_value("Target Speaker Name")

        if len(main_data["target_speaker"]) < 1:
            configure_item("Please enter the source and/or target speaker name", show=True)
            return
        else:
            configure_item("Please enter the source and/or target speaker name", show=False)

        main_data["tgt_files"] = [main_data["target_speaker"] + "\\" + p.stem for p in Path(data[0]).iterdir() if p.suffix == ".wav"] 

        if not main_data["tgt_files"]:
            configure_item("Target list empty (no wav files).", show=True)
        else:
            configure_item("Target list empty (no wav files).", show=False)


        configure_item("Train Target List", items=main_data["tgt_files"])
        set_value("train_path_target", directory)
        
        main_data["tgt_files"].sort()
        main_data["tgt_full"].sort()

        
        main_data["tgtpath_selected"] = True
        configure_item("Copy and Create", enabled=main_data["workpath_selected"] &\
                                                                main_data["convpath_selected"] &\
                                                                        main_data["srcpath_selected"] & \
                                                                                main_data["tgtpath_selected"])

    def apply_selected_conv(sender, data):
        directory = data[0]
        main_data["conv_path"] = directory

        main_data["conv_full"] = [str(p) for p in Path(data[0]).iterdir() if p.suffix == ".wav"]
        
        main_data["conv_files"] = [main_data["source_speaker"] + "\\" + p.stem for p in Path(data[0]).iterdir() if p.suffix == ".wav"]

        if not main_data["conv_files"]:
            configure_item("Conversion list empty (no wav files).", show=True)
        else:
            configure_item("Conversion list empty (no wav files).", show=False)

        configure_item("Conversion List", items=main_data["conv_files"])
        set_value("conv_path", directory)
        
        main_data["conv_files"].sort()
        main_data["conv_full"].sort()

        main_data["convpath_selected"] = True
        configure_item("Copy and Create", enabled=main_data["workpath_selected"] &\
                                                                main_data["convpath_selected"] &\
                                                                        main_data["srcpath_selected"] & \
                                                                                main_data["tgtpath_selected"])

    def apply_selected_main_dir(sender, data):
        directory = data[0]
        main_data["work_path"] = directory

        set_value("main_path", directory)

        main_data["workpath_selected"] = True
        configure_item("Copy and Create", enabled=main_data["workpath_selected"] &\
                                                                main_data["convpath_selected"] &\
                                                                        main_data["srcpath_selected"] & \
                                                                                main_data["tgtpath_selected"])
            

    def load_data(data):
        
        global main_data

        main_data = data


        if not data["src_files"]:
            configure_item("Source list empty (no wav files).", show=True)
        else:
            configure_item("Source list empty (no wav files).", show=False)
        configure_item("Train Source List", items=data["src_files"])
        set_value("train_path_source", data["source_path"])

        if not data["tgt_files"]:
            configure_item("Target list empty (no wav files).", show=True)
        else:
            configure_item("Target list empty (no wav files).", show=False)

        configure_item("Train Target List", items=data["tgt_files"])
        set_value("train_path_target", data["target_path"])

        if not data["conv_files"]:
            configure_item("Conversion list empty (no wav files).", show=True)
        else:
            configure_item("Conversion list empty (no wav files).", show=False)
        configure_item("Conversion List", items=data["conv_files"])
        set_value("conv_path", data["conv_path"])

        set_value("main_path", data["work_path"])
        
        set_value("Source Speaker Name", main_data["source_speaker"])
        set_value("Target Speaker Name", main_data["target_speaker"])

        configure_item("Files Already Copied", show=main_data["files_copied"])
        configure_item("Display Plots", enabled=main_data["allow_plot"])
        configure_item("Create Configuration Files ##button", enabled=main_data["files_copied"])
        configure_item("Pair Files [Already] Ceated.", show=main_data["conf_pair_created"])
        configure_item("Speaker Files [Already] Ceated.", show=main_data["conf_speaker_created"])

        configure_item("Copy and Create", enabled=main_data["workpath_selected"] &\
                                                                main_data["convpath_selected"] &\
                                                                        main_data["srcpath_selected"] & \
                                                                                main_data["tgtpath_selected"])
        set_value("SRC F0", main_data["speaker_f0s"])
        set_value("SRC POW", main_data["speaker_pow"])
        set_value("TGT F0", main_data["target_f0s"])
        set_value("TGT POW", main_data["target_pow"])

        configure_item("F0 Transformation ##button", enabled=main_data["conf_pair_created"] & main_data['conf_speaker_created'])

        configure_item('Step 5: Conversion', enabled=main_data["enable_conv_5"])
        configure_item('Step 4: Training', enabled=main_data["enable_conv_4"])
        configure_item('Step 3: ETJ Extraction', enabled=main_data["enable_conv_3"])
        configure_item('Step 2: Statistical Feature Extraction', enabled=main_data["enable_conv_2"])
        configure_item('Step 1: Feature Extraction', enabled=main_data["enable_conv_1"])

        configure_item("Select From F0 Ratios", item=main_data["f0_ratios"])





    def open_json(sender, data):
        main_data = json.load(open(os.path.join(data[0], data[1])))
        print("wp", main_data["work_path"], "sp", main_data["source_speaker"])
        load_data(main_data)
        set_value("Opened...", f"Opened file {data[1]}")
        configure_item("Opened...", show=True)


    def save_json(sender, data):
        file = data[1] + ".conv"
        main_data["source_speaker"] = get_value("Source Speaker Name")
        main_data["target_speaker"] = get_value("Target Speaker Name")
        
        
        json.dump(main_data, open(os.path.join(data[0], file), "w"))

        saved_loc = file

        set_value("Saved as...", f"Saved as {saved_loc}")
        configure_item("Saved as...", show=True)



    with window("Create Conversion Settings", y_pos=randint(200, 300), width=780, height=320):
        with managed_columns("SettingsButton", 2):        
            with group("file_dialogs"):
                add_button("Load Settings", tip="Must be a .conv file", callback=lambda: open_file_dialog(callback=open_json, extensions=".conv"))
                add_button("Save Settings", tip="No need to enter extension!", callback=lambda: open_file_dialog(callback=save_json, extensions=".*"))

            with group("txt_grp"):
                add_text("Please fill in filename!", show=False)
                add_text(f"Saved as...", show=False, default_value="")
                add_text(f"Opened...", show=False, default_value="")
                add_text(f"Files Already Copied", show=main_data['files_copied'])




        add_separator()
        
        with managed_columns("speakers", 1):
            add_input_text("Source Speaker Name")
            add_input_text("Target Speaker Name")

        add_separator()

        with managed_columns("Source", 2):
            add_button("Browse... ##ID1", callback=lambda: select_directory_dialog(callback=apply_selected_train_source))
            add_input_text("Train Path Source", source="train_path_source")

        with managed_columns("Target", 2):
            add_button("Browse... ##ID2", callback=lambda: select_directory_dialog(callback=apply_selected_train_target))
            add_input_text("Train Path Target", source="train_path_target")

        with managed_columns("Conversion", 2):
            add_button("Browse... ##ID3", callback=lambda: select_directory_dialog(callback=apply_selected_conv))
            add_input_text("Conversion Path", source="conv_path")

        with managed_columns("Workspace", 2):
            add_button("Browse... ##ID4", callback=lambda: select_directory_dialog(callback=apply_selected_main_dir))    
            add_input_text("Work Directory", source="main_path")

        add_separator()

        with managed_columns("CopyButton", 3):
            add_next_column()
            add_button("Copy and Create", tip="There must be enough space on the workspace drive!", enabled=main_data["workpath_selected"] &\
                                                                                        main_data["convpath_selected"] &\
                                                                                            main_data["srcpath_selected"] & \
                                                                                                main_data["tgtpath_selected"], callback=lambda: copy_files())
            unindent()

        with child("InfoTexts", autosize_x=True, autosize_y=True):
            add_indent(offset=10)
            add_text("Source list empty (no wav files).", show=False)
            add_text("Target list empty (no wav files).", show=False)
            add_text("Conversion list empty (no wav files).", show=False)
            add_text("Please enter the source and/or target speaker name", color=(255, 0, 0), show=False)
            add_progress_bar("Copy Progress", default_value=0, show=False)

            unindent()


        add_separator()


    with window("Create Train/Conversion List ##window", x_pos=randint(400, 500), y_pos=randint(200, 450), width=760, height=320):
        with managed_columns("TSL", 2):
            add_listbox("Train Source List")
            add_button("Pop ##1", callback=lambda: configure_item("Train Source List", items=pop_list(main_data["src_files"], get_value("Train Source List"))))

        add_separator()

        with managed_columns("TTL", 2):
            add_listbox("Train Target List")
            add_button("Pop ##2", callback=lambda: configure_item("Train Target List", items=pop_list(main_data["tgt_files"], get_value("Train Target List"))))

        add_separator()

        with managed_columns("TCL", 2):
            add_listbox("Conversion List")
            add_button("Pop ##3", callback=lambda: configure_item("Conversion List", items=pop_list(main_data["conv_files"], get_value("Conversion List"))))

        add_separator()

    with window("Create Configuration Files", x_pos=randint(400, 400), y_pos=randint(0, 450), width=700, height=480):
        with group("pair"):
            add_text("Pair Configuration:", tip="It's REALLY recommended to leave these values be! ")
            add_input_int("n_iter_jnt", source="n_iter_jnt", tip="Number of JNT Iterations", default_value=3)
            add_separator()
            add_input_int("n_mix_mcep", source="n_mix_mcep", tip="Number of MCEP GMM Mixtures", default_value=32)
            add_input_int("n_iter_mcep", source="n_iter_mcep", tip="Number of MCEP GMM Iterations", default_value=100)
            add_separator()
            add_input_int("n_mix_codeap", source="n_mix_codeap", tip="Number of CODEAP GMM Mixtures", default_value=16)
            add_input_int("n_iter_codeap", source="n_iter_codeap", tip="Number of CODEAP GMM Mixtures", default_value=100)
            add_separator()
            add_input_float("morph_coeff", source="morph_coeff", tip="GV Morph Coefficient", default_value=1.0)
            add_separator()
            add_button("Create Pair Files", callback=lambda: create_conf_pair())
            add_text("Pair Files [Already] Ceated.", parent="pair", show=main_data["conf_pair_created"])
        
        add_separator()

        with group("get_minmax"):
            add_text("Speakers Configuration")
            add_button("Calculate Values and Create Files", callback=lambda: create_speaker_progress())
            add_button("Display Plots", enabled=main_data["allow_plot"], callback=lambda: show_item("plots"))
            add_progress_bar("Feature Extraction Progress", show=False)
            add_text("Speaker Files [Already] Ceated.", parent="get_minmax", show=main_data["conf_pair_created"])

        add_separator()
        with managed_columns("speakers_confs", 2):
            with group("src_spkr"):
                add_text("Source Speaker")
                add_input_int2("Min/Max F0 SRC", default_value=main_data["min_max_source"])
                add_input_int("Threshold SRC", default_value=main_data["min_pow_source"])
                add_input_int("Frame Rate SRC", default_value=main_data["source_fs"])
                add_input_int("Bitdepth SRC", default_value=main_data["source_bitd"])
            with group('tgt_spkr'):
                add_text("Target Speaker")
                add_input_int2("Min/Max F0 TGT", default_value=main_data["min_max_target"])
                add_input_int("Threshold TGT", default_value=main_data["min_pow_target"])
                add_input_int("Frame Rate TGT", default_value=main_data["target_fs"])
                add_input_int("Bitdepth TGT", default_value=main_data["target_bitd"])

        add_separator()
        add_button("Recreate Speaker Configs", callback=lambda: recreate_speaker_conf())

    with window("plots", width=1200, height=680):
        with managed_columns("plt", 2):
            with group("src_plt"):
                add_text("Source Speaker Plots")
                add_simple_plot("SRC F0", minscale=-1.0, maxscale=1.0, height=300, value=main_data["speaker_f0s"])
                add_simple_plot("SRC POW", minscale=-1.0, maxscale=1.0, height=300, value=main_data["speaker_pow"])

            with group("tgt_plot"):
                add_text("Target Speaker Plots")
                add_simple_plot("TGT F0", minscale=-1.0, maxscale=1.0, height=300, value=main_data["target_f0s"])
                add_simple_plot("TGT POW", minscale=-1.0, maxscale=1.0, height=300, value=main_data["target_pow"])

    with window("F0 Transformation", width=800, height=400):
        with managed_columns("f0", 2):
            add_input_float("F0 Ratio", default_value=-1)
            add_checkbox("Calculate Automatically (Recommended)", default_value=True)
        add_separator()
        add_button("Transofrm F0", callback=lambda: create_f0_progress())
        add_separator()
        add_text("Transformed", default_value="", show=False)
        add_separator()
        add_progress_bar("F0 Transform Progress", show=False)
        add_separator()
        add_text("Wrong FS", color=(255, 0, 0), show=False, default_value="")

    def all_at_once():
        create_conv_1()
        create_conv_2()
        create_conv_3()
        create_conv_4()
        create_conv_5()

    with window("Training and Conversion", width=400, height=800):
        with managed_columns("main_conv", 2):
            add_listbox("Select F0 Ratio", items=main_data["f0_ratios"])
            add_button("All at Once!")

        add_separator()
        add_button('Step 1: Feature Extraction', enabled=main_data["enable_conv_1"], callback=lambda: create_conv_1())
        add_progress_bar("EF Progress")
        add_separator()
        add_button('Step 2: Statistical Feature Extraction', enabled=main_data["enable_conv_2"], callback=lambda: create_conv_2())
        add_progress_bar("EFS Progress")
        add_separator()
        add_button('Step 3: ETJ Extraction', enabled=main_data["enable_conv_3"], callback=lambda: create_conv_3())
        add_progress_bar("ETJ Progress")
        add_separator()
        add_button('Step 4: Training', enabled=main_data["enable_conv_4"], callback=lambda: create_conv_4())
        add_progress_bar("Training Progress")
        add_separator()
        add_button('Step 5: Conversion', enabled=main_data["enable_conv_5"], callback=lambda: create_conv_5())
        add_progress_bar("Conversion Progress")
        add_separator()
        add_text("EF Done", show=False)
        add_text("EFS Done", show=False)
        add_text("ETJ Done", show=False)
        add_text("Training Done", show=False)
        add_text("Conversion Done", show=False)

    with window("Split File on Silence", width=400, height=200):
        with managed_columns("getfile", 2):
            add_input_text("File Location")
            add_button("Browse... ##232", callback=lambda: get_split_file())

        add_separator()

        with managed_columns("savefile", 2):
            add_input_text("Save Location")
            add_button("Browse... ##2322", callback=lambda: set_split_save())
        
        add_separator()

        add_button("Split", callback=lambda: split_on_silence())

        add_progress_bar("Split Progress")

        add_button("Reset", callback=lambda: reset_vals())
        

    with window("Main Window", no_close=True, width=800, height=500, x_pos=randint(0, 500), y_pos=randint(0, 450)):
        with managed_columns("Main Buttons", 3):
            add_button("Create/Load New Settings", callback=lambda: show_item("Create Conversion Settings"), height=150, width=250)
            add_button("Create Train/Conversion List ##button", callback=lambda: show_item("Create Train/Conversion List ##window"), height=150, width=250)
            add_button("Create Configuration Files ##button", callback=lambda: show_item("Create Configuration Files"), enabled=main_data["files_copied"], height=150, width=250)
            add_button("F0 Transformation ##button", callback=lambda: show_item("F0 Transformation"), enabled=main_data["conf_pair_created"] & main_data["conf_speaker_created"], height=150, width=250)
            add_button("Training and Conversion ##button", callback=lambda: show_item("Training and Conversion"), enabled=main_data["conf_pair_created"] & main_data["conf_speaker_created"] & main_data["f0strans_done"], height=150, width=250)
            add_button("Split File on Silence ##button", callback=lambda: show_item("Split File on Silence"), height=150, width=250)

        with menu_bar("Main Menu Bar"):
            with menu("File"):
                add_menu_item("Credits", callback=lambda: show_item("Credits"))

    
    with window("Credits", width=200, height=150):
        add_text("2021 (C)")
        add_text("OctoShrew")
        add_text("Chubak Bidpaa")
        add_text("www.octoshrew.com")
        add_text("chubak.bidpaa@octoshrew.com")

    hide_item("Create Train/Conversion List ##window")
    hide_item("Create Conversion Settings")
    hide_item("plots")
    hide_item("F0 Transformation")
    hide_item("Create Configuration Files")
    hide_item("Training and Conversion")
    hide_item("Split File on Silence")
    hide_item("Credits")

    start_dearpygui()




