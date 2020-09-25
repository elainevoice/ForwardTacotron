import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph._shortest_path import dijkstra
from scipy.sparse.csr import csr_matrix
import torch

from models.aligner import Aligner
from utils.files import unpickle_binary
from utils.text import phonemes, text_to_sequence, sequence_to_text
from utils.text.cleaners import german_cleaners
from utils import hparams as hp

hp.configure('hparams.py')  # Load hparams from file

text_dict = unpickle_binary('data/text_dict.pkl')
mel = np.load('data/mel/02075.npy')
text = text_dict['02075']

device = torch.device('cpu')
model = Aligner(n_mels=80, lstm_dim=256, num_symbols=len(phonemes)).to(device)
model.eval()
model.load('checkpoints/asvoice_newdurs_tts.aligner/latest_weights.pyt')

print(f'loaded aligner step {model.get_step()}')
mel = torch.tensor(mel)
seq = text_to_sequence(text)
seq = torch.tensor(seq)
pred = model(mel.unsqueeze(0).transpose(1, 2))
pred = torch.softmax(pred, dim=-1)
pred = pred.detach()[0].numpy()
target = seq.numpy()

target_len = target.shape[0]
pred_len = pred.shape[0]
print(pred.shape)
pred_max = np.zeros((pred_len, target_len))

for i in range(pred_len):
    weight = 1. - pred[i, target]
    pred_max[i] = weight

def to_node_index(i, j, cols):
    return cols * i + j

def from_node_index(node_index, cols):
    return node_index // cols, node_index % cols

def to_adj_matrix(mat):
    rows = mat.shape[0]
    cols = mat.shape[1]

    row_ind = []
    col_ind = []
    data = []

    for i in range(rows):
        for j in range(cols):

            node = to_node_index(i, j, cols)

            if j < cols - 1:
                right_node = to_node_index(i, j + 1, cols)
                weight_right = mat[i, j + 1]
                row_ind.append(node)
                col_ind.append(right_node)
                data.append(weight_right)

            if i < rows -1:
                bottom_node = to_node_index(i + 1, j, cols)
                weight_bottom = mat[i + 1, j]
                row_ind.append(node)
                col_ind.append(bottom_node)
                data.append(weight_bottom)

            if i < rows -1 and j < cols-1:
                bottom_node = to_node_index(i + 1, j + 1, cols)
                weight_bottom = mat[i + 1, j + 1]
                row_ind.append(node)
                col_ind.append(bottom_node)
                data.append(weight_bottom)

    #print(f'max row_ind {max(row_ind)} max col_ind {max(col_ind)} dim {ro}')
    adj_mat = coo_matrix((data, (row_ind, col_ind)), shape=(rows * cols, rows * cols))
    return adj_mat.tocsr()

adj_matrix = to_adj_matrix(pred_max)

dist_matrix, predecessors = dijkstra(csgraph=adj_matrix, directed=True, indices=0, return_predecessors=True)

path = []
pr_index = predecessors[-1]
while pr_index != 0:
    path.append(pr_index)
    pr_index = predecessors[pr_index]
path.reverse()
# append first and last
path = [0] + path + [dist_matrix.size-1]
cols = pred_max.shape[1]

mel_text = {}
durations = np.zeros(seq.shape[0])
print(f'dur shape {durations.shape}')

for node_index in path:
    i, j = from_node_index(node_index, cols)

    this_k = target[j]
    letter = sequence_to_text([target[j]])
    next_k = None
    next_letter = None
    if j + 1 < len(target):
        next_k = target[j+1]
        next_letter = sequence_to_text([target[j+1]])

    this_prob = pred[i, this_k]
    next_prob = pred[i, next_k]

    pred_letter = sequence_to_text([np.argmax(pred[i], axis=-1)])

    pred_letter_s1 = sequence_to_text([pred[i].argsort()[-1]])
    pred_letter_s2 = sequence_to_text([pred[i].argsort()[-2]])
    pred_letter_s3 = sequence_to_text([pred[i].argsort()[-3]])

    print(f'{i} {j} {letter} {pred_letter} {pred_max[i, j]} | {letter} {this_prob} | {next_letter} {next_prob} | ')
    mel_text[i] = j

for j in mel_text.values():
    durations[j] += 1

print(f'durs: {durations}')

#for i in range(len(durations)):
#    print(f'{text[i]} {durations[i]} ')
#print(durations)
#print(sum(durations))
#print(mel.shape)
#indices = []
#for r in result:
#    indices.append(target[r])

#print(indices)
#print(sequence_to_text(indices))
#print()
#max_indices = np.argmax(pred, axis=-1)
#print(max_indices)
#print(sequence_to_text(max_indices))
#print(text)
#print(sequence_to_text(target))
#print(tokenizer.decode(result))

#print(dist_matrix)
#print(predecessors)
#print(adj_matrix)

#print(adj_matrix)
#print(pred_max[:10, :10])

#print(target)
#print(tokenizer.decode(target.tolist()))
#print(pred)