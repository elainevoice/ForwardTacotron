from pathlib import Path

import numpy as np
from utils import hparams as hp
from utils.dataset import filter_max_len
from utils.files import unpickle_binary, pickle_binary
from utils.paths import Paths
from utils.text import whitespace_index, text_to_sequence, clean_text

if __name__ == '__main__':
    hp.configure('hparams.py')  # Load hparams from file
    dataset = unpickle_binary('data/train_dataset.pkl')
    dataset = filter_max_len(dataset)
    text_dict = unpickle_binary('data/text_dict.pkl')
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)
    dataset_new = dataset[:]
    text_dict_new = {a: b for a, b in text_dict.items()}
    for index, (item_id, mel_len) in enumerate(dataset):
        dur = np.load(paths.alg / f'{item_id}.npy', allow_pickle=False)
        mel = np.load(paths.mel / f'{item_id}.npy', allow_pickle=False)
        pitch = np.load(paths.phon_pitch / f'{item_id}.npy', allow_pickle=False)
        dur_cum = np.cumsum(dur).astype(np.int)
        text = text_dict[item_id]
        x = text_to_sequence(text)
        w = np.where(np.array(x) == whitespace_index)[0].tolist()
        w = [0] + w + [len(dur)-1]
        if len(w) > 3:
            for i in range(3):
                inds = np.random.choice(w, size=2, replace=False)
                l, r = np.min(inds)+1, np.max(inds)
                ml, mr = dur_cum[l], dur_cum[r]
                x = x[l:r]
                dur = dur[l:r]
                pitch = pitch[l:r]
                mel = mel[:, ml:mr]
                mel_len = mr-ml
                item_id_new = item_id + f'_{i}'
                np.save(paths.alg / f'{item_id_new}.npy', dur, allow_pickle=False)
                np.save(paths.mel / f'{item_id_new}.npy', mel, allow_pickle=False)
                np.save(paths.phon_pitch / f'{item_id_new}.npy', pitch, allow_pickle=False)
                dataset_new.append((item_id_new, mel_len))
                text_dict_new[item_id_new] = text[l:r]
                print(f'{index} / {len(dataset)} {item_id_new} {l} {r} {text[l:r]}')

    pickle_binary(dataset_new, paths.data / 'train_dataset.pkl')
    pickle_binary(tex_dict, paths.data / 'text_dict.pkl')
