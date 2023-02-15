# %%

import numpy as np
import pandas as pd
import pickle

# %%

# Define dictionary of datasets and respective types

datasets_and_types = {
    "SelfRegulationSCP2": "EEG",
    "StandWalkJump": "ECG",
    "SpokenArabicDigits": "Speech",
    "DuckDuckGeese": "Audio",
    "ArticularyWordRecognition": "Motion",
    "CharacterTrajectories": "Motion",
    "EigenWorms": "Motion",
    "PenDigits": "Motion",
    "Handwriting": "HAR",
    "NATOPS": "HAR",
    "RacketSports": "HAR",
    "UWaveGestureLibrary": "HAR"
}

# %%

# Summarize accuracies into dataframe

accuracies = [[], [], [], []]

run_names = ["default_ar_1", "default_ar_2", "default_ar_3", "UEA_replicated"]
model_names = ["AR (s=0)", "AR (s=0.1)", "AR (s=0.2)", "Replicated"]

for dataset_and_type in datasets_and_types.items():

    for i, name in enumerate(run_names):

        path = "../training/" + "UEA_" + dataset_and_type[0] + "__" + name + "/"

        with open(path + "eval_res.pkl", 'rb') as f:
            eval_res = pickle.load(f)

        accuracies[i].append(eval_res["acc"])

dict_for_df = {"Dataset": list(datasets_and_types.keys())}

for i, name in enumerate(model_names):
    dict_for_df[name] = accuracies[i]

dict_for_df["Type"] = list(datasets_and_types.values())

results = pd.DataFrame(dict_for_df)

display(results)

# %%

# Add mean over all datasets

results.loc[len(results)] = ["Mean (all datasets)"] + list(results.mean(axis=0)) + [""]

display(results)

# %%

# Add mean over HAR datasets

results.loc[len(results)] = ["Mean (HAR datasets)"] + list(results[results["Type"] == "HAR"].mean(axis=0)) + [""]

display(results)

# %%
