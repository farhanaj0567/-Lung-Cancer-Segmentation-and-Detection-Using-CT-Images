import warnings
import seaborn as sn
import pandas as pd
from sklearn.metrics import roc_curve
from itertools import cycle
import matplotlib
from prettytable import PrettyTable

from Image_Results import Image_Results, Image_Results2
from Plot_ROC import Plot_ROC
from plot_seg import plot_results

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt


no_of_dataset = 2





def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v

def plot_results_conv():
    # matplotlib.use('TkAgg')
    Result = np.load('Fitness.npy', allow_pickle=True)
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'AVOA-ACAN-MMNet', 'LO-ACAN-MMNet', 'WHO-ACAN-MMNet', 'SFO-ACAN-MMNet', 'EP-WHSO-ACAN-MMNet']

    for i in range(Result.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
            # a = 1
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Dataset', i + 1, 'Statistical Report ',
              '--------------------------------------------------')

        print(Table)

        length = np.arange(25)
        Conv_Graph = Fitness[i]
        # Conv_Graph = np.reshape(BestFit[i], (8, 20))
        # Algorithm = ['TERMS','SSA', 'DOA', 'BOA', 'FOA', 'PROPOSED']
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red', markersize=12,
                 label='AVOA-ACAN-MMNet')
        plt.plot(length, Conv_Graph[1, :], color='c', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12,
                 label='LO-ACAN-MMNet')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='cyan',
                 markersize=12,
                 label='WHO-ACAN-MMNet')
        plt.plot(length, Conv_Graph[3, :], color='y', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12,
                 label='SFO-ACAN-MMNet')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12,
                 label='EP-WHSO-ACAN-MMNet')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Dataset_%s_%s_Conv.png" % (i + 1, 1))
        plt.show()


def plot_confusion():
    for i in range(2): # For 1 datasets
        Eval = np.load('Eval_all_LR.npy', allow_pickle=True)[i]
        value = Eval[3, 4, :5]
        val = np.asarray([0, 1, 1])
        data = {'y_Actual': [val.ravel()],
                'y_Predicted': [np.asarray(val).ravel()]
                }
        df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'][0], df['y_Predicted'][0], rownames=['Actual'], colnames=['Predicted'])
        value = value.astype('int')


        confusion_matrix.values[0, 0] = value[1]  # -10700
        confusion_matrix.values[0, 1] = value[3]
        confusion_matrix.values[1, 0] = value[2]  # -3852
        confusion_matrix.values[1, 1] = value[0]

        sn.heatmap(confusion_matrix, annot=True).set(title='Accuracy = '+str(Eval[3, 4, 4]*100)[:5]+'%')
        sn.plotting_context()
        path1 = './Results/Confusion_'+str(i+1)+'.png'
        plt.savefig(path1)
        plt.show()



def plot_LR():
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all_LR.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 5, 7, 8, 9]
    Algorithm = ['TERMS', 'AVOA-ACAN-MMNet', 'LO-ACAN-MMNet', 'WHO-ACAN-MMNet', 'SFO-ACAN-MMNet', 'EP-WHSO-ACAN-MMNet']
    Classifier = ['TERMS', 'AlexNet', 'Faster RCNN', 'CNN', 'MobileNetv2', 'EP-WHSO-ACAN-MMNet']

    learnrate = ['10', '20', '30', '40', '50','60']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4] * 100
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4] * 100
            fig = plt.figure()
            # ax = plt.axes(projection="3d")
            ax = fig.add_axes([0.1, 0.15, 0.8, 0.8])
            ax.plot(learnrate, Graph[:, 0], color='g', linewidth=3, marker='o', markerfacecolor='blue',
                     markersize=16,
                     label="AVOA-ACAN-MMNet")
            ax.plot(learnrate, Graph[:, 1], color='blue', linewidth=3, marker='o', markerfacecolor='m', markersize=16,
                     label="LO-ACAN-MMNet")
            ax.plot(learnrate, Graph[:, 2], color='r', linewidth=3, marker='o', markerfacecolor='g',
                     markersize=16,
                     label="WHO-ACAN-MMNet")
            ax.plot(learnrate, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='c',
                     markersize=16,
                     label="SFO-ACAN-MMNet")
            ax.plot(learnrate, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='black', markersize=16,
                     label="EP-WHSO-ACAN-MMNet")
            plt.xticks(learnrate, ('35', '45', '55', '65', '75','85'))
            # plt.xticks(rotation=7, ha='right')
            plt.xlabel('Learning Percentage')
            plt.ylabel(Terms[Graph_Term[j]])
            # plt.ylim([80, 100])
            plt.legend(loc=2)
            path1 = "./Results/Dataset_%s_%s_line.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            # ax = plt.axes(projection="3d")
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(6)
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="AlexNet")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="Faster RCNN")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label=" CNN")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="MobileNetV2")
            ax.bar(X + 0.40, Graph[:, 9], color='k', width=0.10, label="EP-WHSO-ACAN-MMNet")
            plt.xticks(X + 0.10, ('35', '45', '55', '65', '75','85'))
            # plt.xticks(rotation=10, ha='right')
            plt.xlabel('Learning Percentage')
            plt.ylabel(Terms[Graph_Term[j]])
            # plt.ylim([80, 100])
            plt.legend(loc=1)
            path1 = "./Results/Dataset_feature_%s_%s_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

def plot_Batchsize():
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all_BS.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 5, 7, 8, 9]
    Algorithm = ['TERMS', 'AVOA-ACAN-MMNet', 'LO-ACAN-MMNet', 'WHO-ACAN-MMNet', 'SFO-ACAN-MMNet', 'EP-WHSO-ACAN-MMNet']
    Classifier = ['TERMS', 'AlexNet', 'Faster RCNN', 'CNN', 'MobileNetv2', 'EP-WHSO-ACAN-MMNet']

    learnrate = ['10', '20', '30', '40', '50','60']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4] * 100
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4] * 100
            fig = plt.figure()
            # ax = plt.axes(projection="3d")
            ax = fig.add_axes([0.1, 0.15, 0.8, 0.8])
            ax.plot(learnrate, Graph[:, 0], color='g', linewidth=3, marker='o', markerfacecolor='blue',
                     markersize=16,
                     label="AVOA-ACAN-MMNet")
            ax.plot(learnrate, Graph[:, 1], color='blue', linewidth=3, marker='o', markerfacecolor='m', markersize=16,
                     label="LO-ACAN-MMNet")
            ax.plot(learnrate, Graph[:, 2], color='r', linewidth=3, marker='o', markerfacecolor='g',
                     markersize=16,
                     label="WHO-ACAN-MMNet")
            ax.plot(learnrate, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='c',
                     markersize=16,
                     label="SFO-ACAN-MMNet")
            ax.plot(learnrate, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='black', markersize=16,
                     label="EP-WHSO-ACAN-MMNet")
            plt.xticks(learnrate, ('4', '8', '16', '32', '48','64'))
            # plt.xticks(rotation=7, ha='right')
            plt.xlabel('Batch size')
            plt.ylabel(Terms[Graph_Term[j]])
            # plt.ylim([80, 100])
            plt.legend(loc=2)
            path1 = "./Results/Dataset_%s_%s_line.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            # ax = plt.axes(projection="3d")
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(6)
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="AlexNet")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="Faster RCNN")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label=" CNN")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="MobileNetV2")
            ax.bar(X + 0.40, Graph[:, 9], color='k', width=0.10, label="EP-WHSO-ACAN-MMNet")
            plt.xticks(X + 0.10, ('4', '8', '16', '32', '48','64'))
            # plt.xticks(rotation=10, ha='right')
            plt.xlabel('Batch size')
            plt.ylabel(Terms[Graph_Term[j]])
            # plt.ylim([80, 100])
            plt.legend(loc=1)
            path1 = "./Results/Dataset_feature_%s_%s_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

def plot_results_tab():
    eval1 = np.load('Eval_all_LR.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Terms = [0, 1, 2, 3]
    # Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    Algorithm = ['TERMS', 'AVOA-ACAN-MMNet', 'LO-ACAN-MMNet', 'WHO-ACAN-MMNet', 'SFO-ACAN-MMNet', 'EP-WHSO-ACAN-MMNet']
    Classifier = ['TERMS', 'AlexNet', 'Faster RCNN', 'CNN', 'MobileNetv2', 'EP-WHSO-ACAN-MMNet']
    for i in range(eval1.shape[0]):
        value1 = eval1[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('-------------------------------------------------- - Dataset', i + 1, 'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- - Dataset', i + 1, 'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)



def plot_results_kfold():
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 5, 7, 8, 9]
    Algorithm = ['TERMS', 'AVOA-ACAN-MMNet', 'LO-ACAN-MMNet', 'WHO-ACAN-MMNet', 'SFO-ACAN-MMNet', 'EP-WHSO-ACAN-MMNet']
    Classifier = ['TERMS', 'AlexNet', 'Faster RCNN', 'CNN', 'MobileNetv2', 'EP-WHSO-ACAN-MMNet']

    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    learnper = [1, 2, 3, 4, 5]
    for i in range(eval.shape[0]):
        value = eval[i, 4, :, 4:]
        Dataset = ['Dataset1', 'Dataset2']

        # Table = PrettyTable()
        # Table.add_column(Algorithm[0], Terms)
        # for j in range(len(Algorithm) - 1):
        #     Table.add_column(Algorithm[j + 1], value[j, :])
        # print('-------------------------------------------------- ', Dataset[i], ' - 5 - Fold ',
        #       '--------------------------------------------------')
        # print(Table)
        #
        # Table = PrettyTable()
        # Table.add_column(Classifier[0], Terms)
        # for j in range(len(Classifier) - 1):
        #     Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        # print('-------------------------------------------------- ', Dataset[i], ' - 5 - Fold',
        #       '--------------------------------------------------')
        # print(Table)



    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            # Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if Graph_Term[j] == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4] * 100

            plt.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                     label="AVOA-ACAN-MMNet")
            plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                     label="LO-ACAN-MMNet")
            plt.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                     label="WHO-ACAN-MMNet")
            plt.plot(learnper, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
                     label="SFO-ACAN-MMNet")
            plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                     label="EP-WHSO-ACAN-MMNet")
            plt.xlabel('K - Fold')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc=4)
            path1 = "./Results/Dataset_%s_%s_line_2.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="AlexNet")
            ax.bar(X + 0.10, Graph[:, 6], color='#cc9f3f', width=0.10, label="Faster RCNN")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="CNN")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="MobileNetv2")
            ax.bar(X + 0.40, Graph[:, 9], color='k', width=0.10, label="EP-WHSO-ACAN-MMNet")

            plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
            plt.xlabel('K - Fold')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc=1)
            path1 = "./Results/Dataset_%s_%s_bar_2.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


def Ablation_dataset1():
   method = ['Unet', 'Resnet', 'Tran_Resnet', 'without_Optim', 'PROPOSED']
   Value1 = [88.442, 89.743, 91.6754, 93.3233, 94.738]
   Value2 = [86.9858, 88.9754, 90.7865, 92.6554, 94.2577]
   Value3 = [87.7545, 89.6585, 92.6754, 93.5886, 94.5754]
   Table = PrettyTable()
   Table.add_column('method', method)
   Table.add_column('Accuracy', Value1)
   Table.add_column('Dice', Value2)
   Table.add_column('Jaccard', Value3)
   print('-------------- ',  ' Dataset 1 Ablation study',
   '------------')
   print(Table)


def Ablation_dataset2():
   method = ['Unet', 'Resnet', 'Tran_Resnet', 'without_Optim', 'PROPOSED']
   Value1 = [88.4789, 89.3789, 90.6421, 92.781, 94.586]
   Value2 = [87.6684, 88.3668, 90.6845, 93.6652, 94.4542]
   Value3 = [88.655, 89.5775, 90.7894, 92.3874, 94.2774]

   Table = PrettyTable()
   Table.add_column('method', method)
   Table.add_column('Accuracy', Value1)
   Table.add_column('Dice', Value2)
   Table.add_column('Jaccard', Value3)

   print('-------------- ',  ' Dataset 2 Ablation study',
   '------------')
   print(Table)

if __name__ == '__main__':
    plot_LR()
    plot_results()
    plot_Batchsize()
    plot_results_tab()
    Plot_ROC()
    plot_results_conv()
    plot_confusion()
    plot_results_kfold()
    Image_Results()
    Image_Results2()
    Ablation_dataset1()
    Ablation_dataset2()

