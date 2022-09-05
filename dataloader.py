import sys
sys.path.append('/')
from readdata import DataReader
import numpy as np
from tqdm import tqdm


def getDataLoader(train_data_path, valid_data_path, test_data_path, params):
    handle = DataReader(train_data_path, valid_data_path, test_data_path, params.max_step, params.n_knowledge_concept)

    kc_data_train, respose_data_train, exercise_data_train = handle.getTrainData()
    kc_data_valid, respose_data_valid, exercise_data_valid = handle.getValidData()
    kc_data_test, respose_data_test, exercise_data_test = handle.getTestData()

    # build_map(kc_data_train, exercise_data_train, kc_data_valid,
    #           exercise_data_valid, kc_data_test, exercise_data_test, params)

    return kc_data_train, respose_data_train, exercise_data_train,kc_data_valid, respose_data_valid, exercise_data_valid, kc_data_test, respose_data_test, exercise_data_test


# 把所有数据加起来，做映射关系，并生成文件
def build_map(kc_data_train, exercise_data_train,kc_data_valid, exercise_data_valid, kc_data_test, exercise_data_test, params):

    n_kc =params.n_knowledge_concept
    n_exercise = params.n_exercise

    kc_data = np.vstack((kc_data_train, kc_data_valid))
    kc_data = np.vstack((kc_data, kc_data_test)).astype(int)

    exercise_data = np.vstack((exercise_data_train, exercise_data_valid))
    exercise_data = np.vstack((exercise_data, exercise_data_test)).astype(int)

    map_matrix = np.zeros((n_exercise,n_kc), dtype=np.int64)

    kc_data_idx = kc_data -1
    exercise_data_idx = exercise_data -1


    for i in tqdm( range(0, exercise_data_idx.shape[0]) ):
        for j in range(0, exercise_data_idx.shape[1]):
            if exercise_data_idx[i][j] != -1:
                map_matrix[exercise_data_idx[i][j]][kc_data_idx[i][j]] = map_matrix[exercise_data_idx[i][j]][kc_data_idx[i][j]] + 1
        map_matrix = np.int64(map_matrix > 0) # 将数组中所有大于0的值变为1

    np.savetxt(params.data_dir + "/exercise_kc_map.txt", map_matrix)

    # np.loadtxt("exercise_kc_map.txt") load matrix
