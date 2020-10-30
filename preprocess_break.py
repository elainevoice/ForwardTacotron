from pathlib import Path

import numpy as np
from utils import hparams as hp
from utils.dataset import filter_max_len
from utils.files import unpickle_binary, pickle_binary
from utils.paths import Paths
from utils.text import comma_index, doubledot_index, text_to_sequence, clean_text

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
        dur_cum = np.cumsum(np.pad(dur, (1, 0)))
        text = text_dict[item_id]
        x = text_to_sequence(text)

        w_comma = np.where(np.array(x) == comma_index)[0].tolist()
        w_dd = np.where(np.array(x) == doubledot_index)[0].tolist()
        indices = []
        for w in w_comma + w_dd:
            indices.append((0, w))
            indices.append((w+2, len(dur)-1))
        for i, (l, r) in enumerate(indices):
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
    pickle_binary(text_dict_new, paths.data / 'text_dict.pkl')
