import numpy as np
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
import os
os.chdir("../")
import soundfile as snd

CORPUS = "Androids-Corpus"
TASK_LIST = ["Interview-Task", "Reading-Task"]
problematic_signals = []
with_interviewer = False
def plot_histogram(data_healthy=None, data_unhealthy=None, save_path="./", xlabel="", ylabel="", title=""):
    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.hist(data_healthy, color="blue")
    plt.hist(data_unhealthy, color="red")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(save_path)
    plt.close()
def subplot_2figure_histogram(data_healthy=None,data_unhealthy= None,save_path="./",xlabel="",ylabel="",title=""):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)
    axs[0].hist(data_healthy, color="blue")
    axs[1].hist(data_unhealthy, color="red")

    axs[0].set_ylabel(ylabel)
    axs[0].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)
    axs[1].set_xlabel(xlabel)

    plt.savefig(save_path)
    plt.close()

def main():
    for task in TASK_LIST:
        SAVE = f"Corpora_analysis/{CORPUS}/{task}/"

        #Creating save folders if necessary
        if not os.path.exists(f"Corpora_analysis/{CORPUS}"):
            os.mkdir(f"Corpora_analysis/{CORPUS}")
            print(f"Creating result analysis folder")
        if not os.path.exists(f"Corpora_analysis/{CORPUS}/{task}/"):
            os.mkdir(f"Corpora_analysis/{CORPUS}/{task}/")
            print(f"Creating result analysis folder")

        label = np.genfromtxt(f"label/label_{CORPUS}_{task}.txt", dtype=str)

        #Global demographic analysis
        x = np.where(label[1:,3] == "x")[0] # Setting to 0 participants without their educational level informatin
        label[1:,3][x] = "0"

        pos_cond = label[1:,-1] == "1"
        neg_cond = label[1:,-1] == "0"

        # Global Health condition
        plot_histogram(data_healthy =label[1:,-1].astype(float)[neg_cond],
            data_unhealthy = label[1:,-1].astype(float)[pos_cond],
            save_path=f"{SAVE}health_condition_{task}.pdf",
            xlabel="Health condition",
            ylabel="Frequency",
            title="Health condition")

        #Graphics - Educational level
        subplot_2figure_histogram(data_healthy =label[1:,3].astype(float)[neg_cond],
                        data_unhealthy = label[1:,3].astype(float)[pos_cond],
                        save_path=f"{SAVE}educational-level_{task}.pdf",
                        xlabel="Educational level",
                        ylabel="Frequency",
                        title="Educational level distribution. Control vs Depressed patients")

        #Graphics - Age
        subplot_2figure_histogram(data_healthy =label[1:,2].astype(float)[neg_cond],
                        data_unhealthy = label[1:,2].astype(float)[pos_cond],
                        save_path=f"{SAVE}age_{task}.pdf",
                        xlabel="Age",
                        ylabel="Frequency",
                        title="Age. Control vs Depressed patients")

        #Graphics - Gender
        fig, axs = plt.subplots(1,2, figsize=(10, 4))

        female_cond = label[1:, 1] == "F"
        male_cond = label[1:, 1] == "M"

        pos_female = pos_cond & female_cond
        neg_female = neg_cond & female_cond

        pos_male = pos_cond & male_cond
        neg_male = neg_cond & male_cond

        bin_list = [0,0.5,1,1.5]
        fig.suptitle("Female vs Male")
        axs[0].hist(label[1:,-1][neg_female].astype(float), color = "blue", bins = bin_list)
        axs[0].hist(label[1:,-1][pos_female].astype(float), color = "red", bins = bin_list)

        axs[1].hist(label[1:,-1][neg_male].astype(float), color = "blue", bins = bin_list)
        axs[1].hist(label[1:,-1][pos_male].astype(float), color = "red", bins = bin_list)

        axs[0].set_xticks(bin_list)
        axs[0].set_xticklabels(["Healthy","","","Unhealthy"])
        axs[1].set_xticks(bin_list)
        axs[1].set_xticklabels(["Healthy","","","Unhealthy"])

        axs[0].set_title("Female")
        axs[0].set_ylabel("Frequency")
        axs[0].set_xlabel("Health Condition")
        axs[1].set_title("Male")
        axs[1].set_ylabel("Frequency")
        axs[1].set_xlabel("Health Condition")

        plt.savefig(f"{SAVE}gender_{task}.pdf")
        plt.close()

        # for task in ["Interview-Task", "Reading-Task"]:
        if task == "Interview-Task" and not with_interviewer:
            audio_path = f"/media/ecampbell/D/Data-io/Androids-Corpus/{task}/audio_clip_gathering/"
        elif task == "Interview-Task" and with_interviewer:
            audio_path = f"/media/ecampbell/D/Data-io/Androids-Corpus/{task}/audio/"
        elif task == "Reading-Task":
            audio_path = f"/media/ecampbell/D/Data-io/Androids-Corpus/{task}/audio_clip_gathering/"
        duration_pos, duration_neg = [], []
        for i, name in enumerate(label[1:,0]):
            print(f"Recording {i+1} - {len(label[1:,0])}  {task}")
            samples, fs = snd.read(f"{audio_path}{name}")

            if np.any(np.isnan(samples)):
                raise ValueError(f"Signal {name} has NAN values")

            duration = len(samples) / (fs*60)
            if duration < 5/60:
                problematic_signals.append([name,duration,task])

            if label[1:,-1][i] == "0":
                duration_neg.append(duration)
            elif label[1:,-1][i] == "1":
                duration_pos.append(duration)

        plt.figure(figsize=(5, 4))
        # plt.title("Recording length of healthy and unhealthy patients.")
        plt.boxplot([duration_neg,duration_pos], showfliers=False)
        plt.xticks([1, 2], labels=["Control", "Depression"])
        plt.ylim(0,6)
        plt.ylabel("Minutes")
        # plt.xlabel("Health condition")
        plt.savefig(f"{SAVE}recording_length-{task}.pdf")
        plt.close()



if __name__ == "__main__":
    main()
    print(f"Problematic signals {problematic_signals}")