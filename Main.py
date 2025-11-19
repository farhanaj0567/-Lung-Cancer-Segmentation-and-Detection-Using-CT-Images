import numpy as np


an = 1
if an == 1:
    tar = np.random.randint(low=0, high=2, size=(1400))
    Target = np.asarray(tar)
    uniq = np.unique(Target)  # Get Unique Values(0 to 10)
    target = np.zeros((Target.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        print(uni)
        index = np.where(Target == uniq[uni])
        target[index[0], uni] = 1
    np.save('Target_2.npy', target)


an = 0
if an == 1:
    Target_1 = np.load('Target_1.npy', allow_pickle=True).reshape(-1, 1)
    k=1