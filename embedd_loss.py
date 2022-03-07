import torch
def kc_exercises_embedd_loss(adj_exercise_kc, kc_node_mebedding, exercise_embedding):
    exercise_kc_similarity = torch.matmul(exercise_embedding, kc_node_mebedding.t())
    exercise_kc_similarity = torch.sigmoid(exercise_kc_similarity)
    loss_exercise_connected_kc = 1 - exercise_kc_similarity
    loss_exercise_disconnected_kc = exercise_kc_similarity
    zero_vec = torch.zeros_like(exercise_kc_similarity)
    loss_exercise_connected_kc = torch.where(adj_exercise_kc > 0, loss_exercise_connected_kc, zero_vec)
    loss_exercise_disconnected_kc = torch.where(adj_exercise_kc <= 0, loss_exercise_disconnected_kc, zero_vec)
    embedd_loss = loss_exercise_connected_kc + loss_exercise_disconnected_kc
    return embedd_loss.mean()