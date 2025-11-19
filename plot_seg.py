import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable



def plot_results():
    Eval_all = np.load('Eval_all.npy', allow_pickle=True)
    # unet = np.load('Eval_unet.npy', allow_pickle=True)
    # deeplabv3 = np.load('Eval_DeepLabv3.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD', 'VARIANCE']
    Algorithm = ['TERMS', 'AVOA', 'LO', 'WHO', 'SFO', 'PROPOSED']
    Methods = ['TERMS', 'UNET', 'RESUNET', 'RESUNET++', 'PROPOSED']
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        # value_unet = unet[n]
        # value_deep = deeplabv3[n]

        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 2, 5))
        for i in range(4,value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 2):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])
                # elif j == value_all.shape[0]:
                #     stats[i, j, 0] = np.max(value_unet[:, i])
                #     stats[i, j, 1] = np.min(value_unet[:, i])
                #     stats[i, j, 2] = np.mean(value_unet[:, i])
                #     stats[i, j, 3] = np.median(value_unet[:, i])
                #     stats[i, j, 4] = np.std(value_unet[:, i])
                # elif j == value_all.shape[0] + 1:
                #     stats[i, j, 0] = np.max(value_deep[:, i])
                #     stats[i, j, 1] = np.min(value_deep[:, i])
                #     stats[i, j, 2] = np.mean(value_deep[:, i])
                #     stats[i, j, 3] = np.median(value_deep[:, i])
                #     stats[i, j, 4] = np.std(value_deep[:, i])

            X = np.arange(stats.shape[2])

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 0, :], color='r', width=0.10, label="FCM")
            ax.bar(X + 0.10, stats[i, 1, :], color='g', width=0.10, label="FullyCNN")
            ax.bar(X + 0.20, stats[i, 2, :], color='b', width=0.10, label="MaskRCNN")
            ax.bar(X + 0.30, stats[i, 3, :], color='m', width=0.10, label="RCNN")
            ax.bar(X + 0.40, stats[i, 4, :], color='k', width=0.10, label="EP-WHSO-ACAN-TResUnet")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i-4])
            plt.legend(loc=1)
            path1 = "./Results/Dataset_%s_%s_alg.png" % (str(n + 1), Terms[i-4])
            plt.savefig(path1)
            plt.show()

            # fig = plt.figure()
            # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            # ax.bar(X + 0.00, stats[i, 5, :], color='r', width=0.10, label="UNet [30]")
            # ax.bar(X + 0.10, stats[i, 6, :], color='g', width=0.10, label="DeepLabV3 [31]")
            # ax.bar(X + 0.20, stats[i, 4, :], color='k', width=0.10, label="G-RDA-DeepLabV3")
            # plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            # plt.xlabel('Statisticsal Analysis')
            # plt.ylabel(Terms[i])
            # plt.legend(loc=1)
            # path1 = "./Results/Dataset_%s_%s_met.png" % (str(n + 1), Terms[i - 4])
            # plt.savefig(path1)
            # plt.show()


if __name__ == '__main__':
    plot_results()
