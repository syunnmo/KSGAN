import numpy as np
import math
import torch
from torch import nn
import utils as utils
from sklearn import metrics
from tqdm import tqdm

def train(epoch_num, model, params, optimizer, adj_exercise_kc, train_kc_data, train_exercise_data, train_exercise_respond_data):
    N = int(math.floor(len(train_exercise_data) / params.batch_size))
    shuffle_index = np.random.permutation(train_exercise_data.shape[0])
    train_kc_data = train_kc_data[shuffle_index]
    train_exercise_data = train_exercise_data[shuffle_index]
    train_exercise_respond_data = train_exercise_respond_data[shuffle_index]

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.train()
    adj_exercise_kc = utils.varible(torch.from_numpy(adj_exercise_kc), params.gpu)

    for idx in tqdm (range(N), desc='Training'):
        data_length = train_kc_data.shape[0]
        begin, end = idx * params.batch_size, min((idx + 1) * params.batch_size, data_length - 1)

        kc_one_seq = train_kc_data[begin : end, :]
        exercise_one_seq = train_exercise_data[begin : end, :]
        exercise_respond_batch_seq = train_exercise_respond_data[begin : end, :]
        target = train_exercise_respond_data[begin : end, :]

        target = (target - 1) / params.n_exercise
        target = np.floor(target)

        input_kc = utils.varible(torch.LongTensor(kc_one_seq), params.gpu)
        input_exercise = utils.varible(torch.LongTensor(exercise_one_seq), params.gpu)
        input_exercise_respond = utils.varible(torch.LongTensor(exercise_respond_batch_seq), params.gpu)
        target = utils.varible(torch.FloatTensor(target), params.gpu)
        target = target[ : , 1 : ]
        target_to_1d = torch.chunk(target, target.shape[0], 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(target.shape[0])], 1)
        target_1d = target_1d.permute(1, 0)

        model.zero_grad()
        loss, filtered_pred, filtered_target = model.forward(adj_exercise_kc, input_kc, input_exercise, input_exercise_respond, target_1d)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), params.maxgradnorm)
        optimizer.step()
        epoch_loss += utils.to_scalar(loss)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        # print(right_pred)
        # print(right_target)
        # right_index = np.flatnonzero(right_target != -1.).tolist()
        pred_list.append(right_pred)
        target_list.append(right_target)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    # if (epoch_num + 1) % params.decay_epoch == 0:
    #     utils.adjust_learning_rate(optimizer, params.init_lr * params.lr_decay)
    # print('lr: ', params.init_lr / (1 + 0.75))
    # utils.adjust_learning_rate(optimizer, params.init_lr / (1 + 0.75))
    # print("all_target", all_target)
    # print("all_pred", all_pred)
    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)
    # f1 = metrics.f1_score(all_target, all_pred)

    return epoch_loss/N, accuracy, auc




def test(model, params, optimizer, adj_exercise_kc, kc_data, exercise_data, exercise_respond_data):
    N = int(math.floor(len(exercise_data) / params.batch_size))

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.eval()
    adj_exercise_kc = utils.varible(torch.from_numpy(adj_exercise_kc), params.gpu)

    # init_memory_value = np.random.normal(0.0, params.init_std, ())
    for idx in tqdm(range(N), desc='Validing/testing'):
        data_length = kc_data.shape[0]
        begin, end = idx * params.batch_size, min((idx + 1) * params.batch_size, data_length - 1)

        kc_one_seq = kc_data[begin : end, :]
        exercise_one_seq = exercise_data[begin : end, :]
        exercise_respond_batch_seq = exercise_respond_data[begin : end, :]
        target = exercise_respond_data[begin : end, :]

        target = (target - 1) / params.n_exercise
        target = np.floor(target)

        input_kc = utils.varible(torch.LongTensor(kc_one_seq), params.gpu)
        input_exercise = utils.varible(torch.LongTensor(exercise_one_seq), params.gpu)
        input_exercise_respond = utils.varible(torch.LongTensor(exercise_respond_batch_seq), params.gpu)
        target = utils.varible(torch.FloatTensor(target), params.gpu)
        target = target[:, 1:]
        target_to_1d = torch.chunk(target, target.shape[0], 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(target.shape[0])], 1)
        target_1d = target_1d.permute(1, 0)

        loss, filtered_pred, filtered_target = model.forward(adj_exercise_kc, input_kc, input_exercise, input_exercise_respond, target_1d)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)
        epoch_loss += utils.to_scalar(loss)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    # print("all_target", all_target)
    # print("all_pred", all_pred)
    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)
    # f1 = metrics.f1_score(all_target, all_pred)

    return epoch_loss/N, accuracy, auc









