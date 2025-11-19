import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from itertools import cycle

no_of_dataset = 2


def Plot_ROC():
    an = 0
    if an == 1:
        learnper = [0.75]
        Varie = [[0.29, 0.27, 0.25, 0.19, 0.17],
                 [0.14, 0.12, 0.10, 0.09, 0.07],

                [0.26, 0.24, 0.22, 0.16, 0.13],
                [0.12, 0.10, 0.08, 0.10, 0.07],]

        # [0.16, 0.13, 0.12, 0.10, 0.06],
        # [0.06, 0.05, 0.04, 0.04, 0.03],]
        #
        # [0.2, 0.19, 0.17, 0.14, 0.05],
        # [0.07, 0.06, 0.05, 0.07, 0.03],
        #
        # [0.17, 0.15, 0.12, 0.10, 0.06],
        # [0.06, 0.05, 0.04, 0.04, 0.03]]
        Y_Score = []
        for a in range(2):
            Targets = np.load('Target_'+ str(a + 1) +'.npy', allow_pickle=True).astype('int')
            Targets = np.reshape(Targets, [-1, 1])
            index_1 = np.where(Targets == 1)
            index_0 = np.where(Targets == 0)
            EVAL = []
            for i in range(len(learnper)):
                Eval = np.zeros((6, 14))
                # roc = np.zeros((6, 14))
                # fpr = dict()
                # tpr = dict()
                Y_Score1 = []
                for j in range(len(Varie[0]) + 1):
                    print(i, j)
                    if j != 5:
                        Tar = Targets.copy()
                        if i == len(learnper) - 1:
                            varie = Varie[1][j] + ((Varie[0][j] - Varie[1][j]) / len(learnper)) * (
                                    len(learnper) - (i - 0.8))
                        else:
                            varie = Varie[1][j] + ((Varie[0][j] - Varie[1][j]) / len(learnper)) * (len(learnper) - i)
                        perc_1 = round(index_1[0].shape[0] * varie)
                        perc_0 = round(index_0[0].shape[0] * varie)
                        rand_ind_1 = np.random.randint(low=0, high=index_1[0].shape[0], size=perc_1)
                        rand_ind_0 = np.random.randint(low=0, high=index_0[0].shape[0], size=perc_0)
                        Tar[index_1[0][rand_ind_1], index_1[1][rand_ind_1]] = 0
                        Tar[index_0[0][rand_ind_0], index_0[1][rand_ind_0]] = 1
                        y_score1 = Tar
                        Y_Score1.append(y_score1)
                Y_Score.append(Y_Score1)
        np.save('Y_Score.npy', Y_Score)
    else:
        lw=2

        cls = ['AlexNet', 'Faster RCNN', 'CNN', 'MobileNetv2', 'EP-WHSO-ACAN-MMNet']

        # Classifier = ['TERMS', 'Xgboost', 'DT', 'NN', 'FUZZY', 'KNN', 'PROPOSED']
        for a in range(no_of_dataset): # For 5 Datasets
            Actual = np.load('Target_'+ str(a + 1) +'.npy', allow_pickle=True).astype('int')
            # Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')

            colors = cycle(["blue", "crimson", "gold", "lime", "black"]) #  "cornflowerblue","darkorange", "aqua"
            for i, color in zip(range(5), colors): # For all classifiers
                Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i]
                false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
                plt.plot(
                    false_positive_rate1,
                    true_positive_rate1,
                    color=color,
                    lw=lw,
                    label=cls[i],
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            path1 = "./Results/Dataset_%s_ROC_%s_.png" % (a+1,1)
            # path1 = "./Results/Dataset_%s_ROC_%s_.png" % (a + 1, i+1)
            plt.savefig(path1)
            plt.show()


if __name__ == '__main__':
    Plot_ROC()