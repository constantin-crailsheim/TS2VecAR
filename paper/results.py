# %%

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

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

run_names = ["default_ar_1", "default_ar_2", "default_ar_3", "replicated"]
model_names = ["AR (s=0)", "AR (s=0.1)", "AR (s=0.2)", "Replicated"]

for dataset_and_type in datasets_and_types.items():

    for i, name in enumerate(run_names):

        path = "../training/" + dataset_and_type[0] + "__" + name + "/"

        with open(path + "eval_res.pkl", 'rb') as f:
            eval_res = pickle.load(f)

        accuracies[i].append(eval_res["acc"])

dict_for_df = {"Dataset": list(datasets_and_types.keys())}

for i, name in enumerate(model_names):
    dict_for_df[name] = accuracies[i]

dict_for_df["Type"] = list(datasets_and_types.values())

results = pd.DataFrame(dict_for_df)

pd.options.display.float_format = '{:,.3f}'.format
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

# Store as markdown

# https://gist.github.com/jplsightm/c7df5cd2bc62dc84c5158a80cf0af6df

import tabulate

def pandas_df_to_markdown_table(df):
    # Dependent upon ipython
    # shamelessly stolen from https://stackoverflow.com/questions/33181846/programmatically-convert-pandas-dataframe-to-markdown-table
    from IPython.display import Markdown, display
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    # display(Markdown(df_formatted.to_csv(sep="|", index=False)))
    return df_formatted.to_csv(sep="|", index=False)
#     return df_formatted

def df_to_markdown(df, y_index=False):
    blob = tabulate(df, headers='keys', tablefmt='pipe')
    if not y_index:
        # Remove the index with some creative splicing and iteration
        return '\n'.join(['| {}'.format(row.split('|', 2)[-1]) for row in blob.split('\n')])
    return blob

# %%

fmt = ['---' for i in range(len(results.columns))]
df_fmt = pd.DataFrame([fmt], columns=results.columns)
df_formatted = pd.concat([df_fmt, results])
print(df_formatted.to_csv(sep="|", index=False, float_format='%.3f'))


# %%

# Plot loss log 

# Set datasets and s
dataset = "StandWalkJump"
s = "0.1"

# Plot loss convergence
s_to_number = {"0": "1", "0.1": "2", "0.2": "3"}
number = s_to_number[s]
name = "default_ar_" + number

path = "../training/" + dataset + "__" + name + "/"

with open(path + "loss_log.pkl", 'rb') as f:
    loss_log = pickle.load(f)

fig, axs = plt.subplots(1,1)

plt.plot(np.arange(len(loss_log)), loss_log)
plt.xlabel("Epochs")
plt.ylabel("Loss")
# plt.title("Loss convergence of " + dataset + " for s = " + s)
plt.savefig("text/fig/" + dataset + "_" + name + ".png")


# %%

# Plot loss log 

s_to_number = {"0": "1", "0.1": "2", "0.2": "3"}
for dataset in list(datasets_and_types.keys()):
    for s in ["0", "0.1", "0.2"]:
        number = s_to_number[s]
        name = "default_ar_" + number

        path = "../training/" + dataset + "__" + name + "/"

        with open(path + "loss_log.pkl", 'rb') as f:
            loss_log = pickle.load(f)

        fig, axs = plt.subplots(1,1)

        plt.plot(np.arange(len(loss_log)), loss_log)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss convergence of " + dataset + " for s = " + s)
        plt.savefig("misc/" + dataset + "_" + name + ".png")

# %%
