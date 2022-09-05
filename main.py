import torch
import argparse
from model import MODEL
from run import train, test
import numpy as np
import torch.optim as optim
from dataloader import getDataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, help='the gpu will be used, e.g "0,1,2,3"')
    parser.add_argument('--max_iter', type=int, default=1000, help='number of iterations')
    parser.add_argument('--init_lr', type=float, default= 0.001, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.75, help='learning rate decay')
    parser.add_argument('--final_lr', type=float, default=1E-5,
                        help='learning rate will not decrease after hitting this threshold')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--exercise_embed_dim', type=int, default=128, help='exercise embedding dimensions')
    parser.add_argument('--hidden_dim', type=float, default=128, help='hidden state dim for LSTM')
    parser.add_argument('--layer_num', type=float, default=1, help='layer number for LSTM')
    parser.add_argument('--max_step', type=int, default=100, help='the allowed maximum length of a sequence')
    parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
    parser.add_argument('--Lamda', type=float, default=0, help='hyper-parameter Lamda')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of head attentions.')
    parser.add_argument('--mode', type=int, default=3, help='mode of Integration Function. '
                                                            '1:ca'
                                                            '2:mul'
                                                            '3:ca mul'
                                                            '4:rasch')
    parser.add_argument('--fold', type=str, default='1', help='number of fold')
    dataset = 'assist2012'

    if dataset == 'assist2009_B':
        parser.add_argument('--batch_size', type=int, default=256, help='the batch size')
        parser.add_argument('--n_knowledge_concept', type=int, default=110, help='the number of unique questions in the dataset')
        parser.add_argument('--n_exercise', type=int, default=16891, help='the number of unique questions in the dataset')
        parser.add_argument('--data_dir', type=str, default='./data/assist2009_B', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2009_B', help='data set name')

    if dataset == 'assist2012':
        parser.add_argument('--batch_size', type=int, default=512, help='the batch size')
        parser.add_argument('--n_knowledge_concept', type=int, default=245, help='the number of unique questions in the dataset')
        parser.add_argument('--n_exercise', type=int, default=50918, help='the number of unique questions in the dataset')
        parser.add_argument('--data_dir', type=str, default='./data/assist2012', help='data directory')
        parser.add_argument('--data_name', type=str, default='ASSISTments12', help='data set name')

    if dataset == 'slepemapy':
        parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
        parser.add_argument('--n_knowledge_concept', type=int, default=1473, help='the number of unique questions in the dataset')
        parser.add_argument('--n_exercise', type=int, default=2946, help='the number of unique questions in the dataset')
        parser.add_argument('--data_dir', type=str, default='./data/slepemapy', help='data directory')
        parser.add_argument('--data_name', type=str, default='slepemapy', help='data set name')

    params = parser.parse_args()
    params.memory_size = params.n_knowledge_concept
    params.lr = params.init_lr
    params.memory_key_state_dim = params.exercise_embed_dim
    params.memory_value_state_dim = params.exercise_embed_dim * 2

    train_data_path = params.data_dir + "/" + params.data_name + "_train"+ params.fold + ".csv"
    valid_data_path = params.data_dir + "/" + params.data_name + "_valid"+ params.fold + ".csv"
    test_data_path = params. data_dir + "/" + params.data_name + "_test"+ params.fold + ".csv"

    train_kc_data, train_respond_data, train_exercise_data, \
    valid_kc_data, valid_respose_data, valid_exercise_data, \
    test_kc_data, test_respose_data, test_exercise_data \
        = getDataLoader(train_data_path, valid_data_path, test_data_path, params)

    train_exercise_respond_data = train_respond_data * params.n_exercise + train_exercise_data
    valid_exercise_respose_data = valid_respose_data * params.n_exercise + valid_exercise_data
    test_exercise_respose_data = test_respose_data * params.n_exercise + test_exercise_data
    adj_exercise_kc = np.loadtxt(params.data_dir + "/exercise_kc_map.txt")


    model = MODEL(n_exercise=params.n_exercise,
                  batch_size=params.batch_size,
                  exercise_embed_dim=params.exercise_embed_dim,
                  hidden_dim = params.hidden_dim,
                  layer_num = params.layer_num,
                  Lamda = params.Lamda,
                  params = params )

    optimizer = optim.Adam(params=model.parameters(), lr=params.lr, betas=(0.9, 0.9), weight_decay=1e-5)

    if params.gpu >= 0:
        print('device: ' + str(params.gpu))
        torch.cuda.set_device(params.gpu)
        model.cuda()

    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0
    best_test_auc = 0
    best_test_acc= 0


    for idx in range(params.max_iter):
        train_loss, train_accuracy, train_auc = train(idx, model, params, optimizer, adj_exercise_kc, train_kc_data, train_exercise_data, train_exercise_respond_data)
        print('Epoch %d/%d, loss : %3.5f, auc : %3.5f, accuracy : %3.5f' % (idx + 1, params.max_iter, train_loss, train_auc, train_accuracy))
        with torch.no_grad():
            valid_loss, valid_accuracy, valid_auc = test(model, params, optimizer, adj_exercise_kc, valid_kc_data, valid_exercise_data, valid_exercise_respose_data)
            print('loss : %3.5f, valid auc : %3.5f, valid accuracy : %3.5f' % ( valid_loss, valid_auc, valid_accuracy))
            test_loss, test_accuracy, test_auc = test(model, params, optimizer, adj_exercise_kc, test_kc_data, test_exercise_data, test_exercise_respose_data)
            print("test_loss: %.4f\t test_auc: %.4f\t test_accuracy: %.4f\t " % (test_loss, test_auc, test_accuracy))

        all_train_auc[idx + 1] = train_auc
        all_train_accuracy[idx + 1] = train_accuracy
        all_train_loss[idx + 1] = train_loss
        all_valid_loss[idx + 1] = valid_loss
        all_valid_accuracy[idx + 1] = valid_accuracy
        all_valid_auc[idx + 1] = valid_auc
        #
        # output the epoch with the best validation auc
        # if valid_auc > best_valid_auc:
        #     print('%3.4f to %3.4f' % (best_valid_auc, valid_auc))
        #     best_valid_auc = valid_auc
        if test_auc > best_test_auc:
            print('auc: %3.4f to %3.4f' % (best_test_auc, test_auc))
            best_test_auc = test_auc

        if test_accuracy > best_test_acc:
            print('acc: %3.4f to %3.4f' % (best_test_acc, test_accuracy))
            best_test_acc = test_accuracy
        #         best_epoch = idx+1
        #         best_valid_acc = valid_accuracy
        #         best_valid_loss = valid_loss
        #         test_loss, test_accuracy, test_auc = test(model, params, optimizer, test_q_data, test_qa_data)
        #         print("test_auc: %.4f\ttest_accuracy: %.4f\ttest_loss: %.4f\t" % (test_auc, test_accuracy, test_loss))
        print('best_test_auc:  %3.5f' % (best_test_auc))
        print('best_test_acc:  %3.5f' % (best_test_acc))
    # print("best outcome: best epoch: %.4f" % (best_epoch))
    # print("valid_auc: %.4f\tvalid_accuracy: %.4f\tvalid_loss: %.4f\t" % (best_valid_auc, best_valid_acc, best_valid_loss))
    # print("test_auc: %.4f\ttest_accuracy: %.4f\ttest_loss: %.4f\t" % (test_auc, test_accuracy, test_loss))



if __name__ == "__main__":
    main()
