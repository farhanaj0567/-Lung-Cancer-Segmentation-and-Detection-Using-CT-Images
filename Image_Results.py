import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

no_of_dataset = 2

def Image_Results():
    for n in range(1):
        Orig = np.load('Img_1.npy',allow_pickle=True)
        # Images = np.load('Prep_Images.npy', allow_pickle=True)
        segment = np.load('Segmented.npy', allow_pickle=True)
        ind = [11, 131, 176, 177, 178]
        for j in range(len(ind)):
            original = Orig[ind[j]]
            # image = Images[ind[j]].astype('uint8')
            seg = segment[ind[j]]
            # cv.imshow('im', Output)
            # cv.waitKey(0)

            fig, ax = plt.subplots(1, 2)
            plt.suptitle("Image %d" % (j + 1), fontsize=20)
            plt.subplot(1, 2, 1)
            plt.title('Orig')
            plt.imshow(original)
            # plt.subplot(1, 3, 2)
            # plt.title('Prep')
            # plt.imshow(image)
            plt.subplot(1, 2, 2)
            plt.title('Seg')
            plt.imshow(seg)
            cv.imwrite('./sample image/Dataset1/orig-' + str(j + 1) + '.png', original)
            path1 = "./Results/Image_Res/Dataset1_image_%s.png" % (j + 1)
            plt.savefig(path1)
            plt.show()
        # cv.imwrite('./sample/orig-' + str(j + 1) + '.png', original)
        # cv.imwrite('./Results/pre-proc-' + str(j + 1) + '.png', image)
        # cv.imwrite('./Results/segment-' + str(j + 1) + '.png', Output)
        # cv.imwrite('./Results/ground-' + str(j + 1) + '.png', gt)

def tes_images():
    segment = np.load('Segmented.npy', allow_pickle=True)
    for j in range(len(segment)):
        print(j)
        img = segment[j]
        cv.imwrite('./All_img/'+str(j+1)+'.jpg', img)


def Image_Results2():
    for n in range(1):
        Orig = np.load('Img_2.npy',allow_pickle=True)
        # Images = np.load('Prep_Images.npy', allow_pickle=True)
        segment = np.load('Segmented.npy', allow_pickle=True)
        ind = [206, 251, 262, 283, 286]
        for j in range(len(ind)):
            original = Orig[ind[j]]
            # image = Images[ind[j]].astype('uint8')
            seg = segment[ind[j]]
            # cv.imshow('im', Output)
            # cv.waitKey(0)

            fig, ax = plt.subplots(1, 2)
            plt.suptitle("Image %d" % (j + 1), fontsize=20)
            plt.subplot(1, 2, 1)
            plt.title('Orig')
            plt.imshow(original)
            # plt.subplot(1, 3, 2)
            # plt.title('Prep')
            # plt.imshow(image)
            plt.subplot(1, 2, 2)
            plt.title('Seg')
            plt.imshow(seg)
            cv.imwrite('./sample image/Dataset2/orig-' + str(j + 1) + '.png', original)
            path1 = "./Results/Image_Res/Dataset2_image_%s.png" % (j + 1)
            plt.savefig(path1)
            plt.show()


if __name__ =='__main__':
    Image_Results() #used
    Image_Results2()  #used
    # tes_images()