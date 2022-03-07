import torch
import torch.nn as nn
import numpy as np
import utils as utils
import embedd_loss as embedd_loss
from torch.autograd import Variable
import torch.nn.functional as F
class MODEL(nn.Module):

    def __init__(self, n_exercise, batch_size, exercise_embed_dim, hidden_dim, layer_num, Lamda, params,student_num=None):
        super(MODEL, self).__init__()
        self.n_exercise = n_exercise
        self.n_kc = params.n_knowledge_concept
        self.batch_size = batch_size
        self.exercise_embed_dim = exercise_embed_dim
        self.student_num = student_num
        self.nheads = params.num_heads
        self.alpha = params.alpha
        self.dropout = params.dropout
        self.params = params
        self.mode = params.mode
        self.Lamda = Lamda

        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.lstm = nn.LSTM(self.exercise_embed_dim * 2, self.hidden_dim, self.layer_num, batch_first=True, dropout=self.dropout)
        self.fc1 = nn.Linear(self.exercise_embed_dim + self.hidden_dim, self.hidden_dim, bias=True)
        self.fc2 = nn.Linear(self.hidden_dim, 1, bias=True)
        self.rel = nn.ReLU()

        self.exercise_embed = nn.Parameter(torch.randn(self.n_exercise, self.exercise_embed_dim))
        self.kc_embed = nn.Parameter(torch.randn(self.n_kc, self.exercise_embed_dim))

        self.exercise_kc_attentions = [
            Exercise_KC_GraphAttentionLayer(self.exercise_embed_dim, self.exercise_embed_dim, dropout=self.dropout, alpha=self.alpha, mode=self.mode,
                                            concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.exercise_kc_attentions):
            self.add_module('exercise_kc_attention_{}'.format(i), attention)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal(self.fc1.weight)
        nn.init.kaiming_normal(self.fc2.weight)
        nn.init.kaiming_normal(self.exercise_embed)
        nn.init.kaiming_normal(self.kc_embed)
        nn.init.constant(self.fc1.bias, 0)
        nn.init.constant(self.fc2.bias, 0)

    def forward(self, adj_exercise_kc, kc_data, exercise_data, exercise_respond_data, target, student_id=None):

        batch_size = exercise_data.shape[0]
        seqlen = exercise_data.shape[1]
        exercise_node_embedding = self.exercise_embed
        kc_node_mebedding = self.kc_embed
        exercise_embedding = torch.cat([att(exercise_node_embedding, kc_node_mebedding, adj_exercise_kc) for att in self.exercise_kc_attentions], dim=1).view(self.n_exercise, self.exercise_embed_dim ,self.nheads).mean(2)
        exercise_embedding_add_zero = torch.cat(
            [utils.varible(torch.zeros(1, exercise_embedding.shape[1]), self.params.gpu), exercise_embedding], dim=0)
        slice_exercise_data = torch.chunk(exercise_data, seqlen, 1)
        slice_exercise_embedd_data = []
        for i, single_slice_exercise_data_index in enumerate(slice_exercise_data):
            single_slice_exercise_embedd_data = torch.index_select(exercise_embedding_add_zero, 0,
                                                                   single_slice_exercise_data_index.squeeze(1))
            slice_exercise_embedd_data.append(single_slice_exercise_embedd_data)
        slice_exercise_respond_data = torch.chunk(exercise_respond_data, seqlen, 1)

        zeros = torch.zeros_like(exercise_embedding)
        cat1 = torch.cat((zeros, exercise_embedding), -1)
        cat2 = torch.cat((exercise_embedding, zeros), -1)
        response_embedding = torch.cat((cat1, cat2), -2)
        response_embedding_add_zero = torch.cat([utils.varible(torch.zeros(1, response_embedding.shape[1]), self.params.gpu), response_embedding],
                                                dim=0)
        slice_respond_embedd_data = []
        for i, single_slice_respond_data_index in enumerate(slice_exercise_respond_data):
            single_slice_respond_embedd_data = torch.index_select(response_embedding_add_zero, 0, single_slice_respond_data_index.squeeze(1))
            slice_respond_embedd_data.append(single_slice_respond_embedd_data)


        lstm_input = torch.cat([slice_respond_embedd_data[i].unsqueeze(1) for i in range(seqlen)], 1)
        assessment_exercises = torch.cat([slice_exercise_embedd_data[i].unsqueeze(1) for i in range(seqlen)], 1)
        assessment_exercises = assessment_exercises[:, 1 : , :]
        h0 = utils.varible(Variable(torch.zeros(self.layer_num, lstm_input.size(0), self.hidden_dim)), self.params.gpu)
        c0 = utils.varible(Variable(torch.zeros(self.layer_num, lstm_input.size(0), self.hidden_dim)), self.params.gpu)
        learn_state, (hn, cn) = self.lstm(lstm_input, (h0, c0))
        learn_state = learn_state[:, : seqlen - 1, :]

        y = self.rel(self.fc1(torch.cat((learn_state, assessment_exercises), -1)))
        pred = self.fc2(y)
        pred = pred.squeeze(-1).view(batch_size * (seqlen - 1), -1)
        target_1d = target
        mask = target_1d.ge(0)
        pred_1d = pred.view(-1, 1)
        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        predict_loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)
        kc_exercises_embedd_loss = embedd_loss.kc_exercises_embedd_loss(adj_exercise_kc, kc_node_mebedding, exercise_embedding)
        loss = predict_loss + self.Lamda * kc_exercises_embedd_loss

        return loss, torch.sigmoid(filtered_pred), filtered_target

class Exercise_KC_GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, mode, concat=True):
        super(Exercise_KC_GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.mode =mode
        self.W1 = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.kaiming_normal(self.W1)

        if mode == 1 or mode == 3:
            self.reduceDim = nn.Linear(in_features * 2, self.out_features, bias=True)
            nn.init.kaiming_normal(self.reduceDim.weight)
            nn.init.constant(self.reduceDim.bias, 0)
        if mode != 1:
            self.E = nn.Parameter(torch.empty(size=(in_features, out_features)))
            nn.init.kaiming_normal(self.E)
        if mode ==4:
            self.U = nn.Parameter(torch.empty(size=(in_features, 1)))
            nn.init.kaiming_normal(self.U)

        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.kaiming_normal(self.a)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, exercise_h, kc_h, adj_exercise_kc):
        if self.concat:
            kc_Wh = torch.mm(kc_h, self.W1)
            exercise_Wh = torch.mm(exercise_h, self.W1)
            a_input = self._prepare_attentional_mechanism_input(kc_Wh, exercise_Wh)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj_exercise_kc > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            new_kc_embed = torch.matmul(attention, kc_Wh)

            if self.mode == 1:
                exercises_embedd = torch.cat((new_kc_embed, exercise_h), dim=1)
                exercises_embedd = self.reduceDim(exercises_embedd)
            if self.mode == 2:
                exercise_Eh = torch.mm(exercise_h, self.E)
                exercises_embedd = new_kc_embed.mul(exercise_Eh)
            if self.mode == 3:
                exercise_Eh = torch.mm(exercise_h, self.E)
                exercises_embedd = torch.cat((new_kc_embed, new_kc_embed.mul(exercise_Eh)), dim=1)
                exercises_embedd = self.reduceDim(exercises_embedd)
            if self.mode == 4:
                u = torch.mm(exercise_h, self.U)
                d_kt = torch.mm(new_kc_embed, self.E)
                exercises_embedd = new_kc_embed + u * d_kt
            return F.elu(exercises_embedd)

    def _prepare_attentional_mechanism_input(self, kc_Wh, exercise_Wh):
        N_kc = kc_Wh.size()[0]
        N_exercise = exercise_Wh.size()[0]
        Wh_repeated_in_chunks = exercise_Wh.repeat_interleave(N_kc, dim=0)
        Wh_repeated_alternating = kc_Wh.repeat(N_exercise, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N_exercise, N_kc, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'