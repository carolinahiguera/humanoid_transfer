import pickle
import numpy as np
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt 

path_task1 = f'./low_friction/'
path_task2 = f'./normal_friction/'


with open(f'{path_task1}/seq_states_100e.pkl', 'rb') as f:
    seq_states_t1 = pickle.load(f)

with open(f'{path_task2}/seq_states_100e.pkl', 'rb') as f:
    seq_states_t2 = pickle.load(f)

num_states = len(seq_states_t1[0][0])
seq_len = 3
dataset_task1 = np.empty((0,num_states*seq_len))
dataset_task2 = np.empty((0,num_states*seq_len))


def get_datasets():
    global dataset_task1, dataset_task2
    for i in range( len(seq_states_t1)):
        for j in range(seq_len-1,len(seq_states_t1[i])):
            seq = []
            for k in range(seq_len-1,-1,-1):
                for m in range(0,num_states):
                    seq.append(seq_states_t1[i][j-k][m])
            dataset_task1 = np.append(dataset_task1, np.array([seq]), axis=0)

    for i in range( len(seq_states_t2)):
        for j in range(seq_len-1,len(seq_states_t2[i])):
            seq = []
            for k in range(seq_len-1,-1,-1):
                for m in range(0,num_states):
                    seq.append(seq_states_t2[i][j-k][m])
            dataset_task2 = np.append(dataset_task2, np.array([seq]), axis=0)
    

def compute_confusion_matrix(clfs):
    conf_matrix = np.zeros((2,2))
    for t in range(2):
        for j in range(2):
            y_pred = clfs[t].predict(Xtest[j])
            if(j==t):
                n_error = y_pred[y_pred == -1].size / Xtest[j].shape[0]
            else:
                n_error = y_pred[y_pred == 1].size / Xtest[j].shape[0]
            conf_matrix[t][j] = n_error
    return conf_matrix


# get_datasets()
# np.save('./transfer/dataset_task1.npy', dataset_task1)
# np.save('./transfer/dataset_task2.npy', dataset_task2)
path = f'./transfer/one_svm_3states/'
dataset_task1 = np.load(f'{path}/dataset_task1.npy')
dataset_task2 = np.load(f'{path}/dataset_task2.npy')




# nu = [10e-4, 10e-3, 10e-2, 10e-1, 0.2, 0.5, 0.7, 0.9]
# kernel = ['linear', 'sigmoid']
# degree = [2,3,4,5]
# gamma = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e-0]

# params 2 estados
# nu = [0.001]
# kernel = ['rbf',]
# degree = [2,3,4,5]
# gamma = [100]


# nu = [10e-4, 10e-3, 10e-2, 10e-1, 0.2, 0.5, 0.7, 0.9]
# kernel = ['rbf']
# degree = [2,3,4,5]
# gamma = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e-0, 10e1, 10e2]

#params 3 estados
nu = [0.01]
kernel = ['rbf']
gamma = [0.1]

Xtrain = []
Xtest = []
clfs = []

Xtrain.append(dataset_task1[0:int(0.7*dataset_task1.shape[0])])
Xtest.append(dataset_task1[int(0.7*dataset_task1.shape[0])+1:dataset_task1.shape[0]])
Xtrain.append(dataset_task2[0:int(0.7*dataset_task2.shape[0])])
Xtest.append(dataset_task2[int(0.7*dataset_task2.shape[0])+1:dataset_task2.shape[0]])




for param_kernel in kernel:
    save_path = f'{path}/{param_kernel}/'
    for param_nu in nu:
        for param_gamma in gamma:            
            if(param_kernel!='poly'):
                print(f'kernel={param_kernel} - gamma={param_gamma} - nu={param_nu}')
                clfs = []
                for i in range(2):
                    clfs.append(svm.OneClassSVM(nu=param_nu, kernel=param_kernel, gamma=param_gamma))
                    clfs[i].fit(Xtrain[i])
                    pkl_filename = f'{save_path}/svm_model_3seq_T{i}.pkl'
                    with open(pkl_filename, 'wb') as file:
                        pickle.dump(clfs[i], file)
                conf_matrix = compute_confusion_matrix(clfs)
                fig = plt.gcf()
                ax = plt.subplot()
                sns.heatmap(conf_matrix, annot=True, ax = ax, cmap="YlGnBu");
                ax.xaxis.set_ticklabels(['T1', 'T2']); ax.yaxis.set_ticklabels(['T1', 'T2']);
                plt.savefig(f'{save_path}/FINAL_n{param_nu}_r{param_gamma}.png')
                plt.clf()                
            else:
                for param_degree in degree:
                    print(f'kernel={param_kernel} - gamma={param_gamma} - nu={param_nu} - d={param_degree}')
                    clfs = []
                    for i in range(2):
                        clfs.append(svm.OneClassSVM(nu=param_nu, kernel=param_kernel, gamma=param_gamma, degree=param_degree))
                        clfs[i].fit(Xtrain[i])
                    conf_matrix = compute_confusion_matrix(clfs)
                    fig = plt.gcf()
                    ax = plt.subplot()
                    sns.heatmap(conf_matrix, annot=True, ax = ax, cmap="YlGnBu");
                    ax.xaxis.set_ticklabels(['T1', 'T2']); ax.yaxis.set_ticklabels(['T1', 'T2']);
                    plt.savefig(f'{save_path}/n{param_nu}_r{param_gamma}_d{param_degree}.png')
                    plt.clf()




