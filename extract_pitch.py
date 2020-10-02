from pathlib import Path

from utils import hparams as hp
import parselmouth
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd

from utils.dataset import get_tts_datasets, filter_max_len
from utils.display import plot_mel, progbar, stream
from utils.dsp import melspectrogram
from utils.files import unpickle_binary
from utils.paths import Paths


def interpolate(data, invalid=None):
    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid,
                                    return_distances=False,
                                    return_indices=True)
    return data[tuple(ind)]


def normalize(phoneme_pitches):
    nonzeros = np.concatenate([v[np.where(v != 0.0)[0]]
                               for item_id, v in phoneme_pitches])
    mean, std = np.mean(nonzeros), np.std(nonzeros)
    for item_id, v in phoneme_pitches:
        zero_idxs = np.where(v == 0.0)[0]
        v -= mean
        v /= std
        v[zero_idxs] = 0.0
    return mean, std

if __name__ == '__main__':

    MAX_FREQ = 250.
    hp.configure('hparams.py')
    train_data = unpickle_binary('data/train_dataset.pkl')
    val_data = unpickle_binary('data/val_dataset.pkl')
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)
    all_data = train_data + val_data
    all_data = filter_max_len(all_data)

    phoneme_pitches = []
    text_dict = unpickle_binary('data/text_dict.pkl')
    # adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/0b27e359a5869cd23294c1707c92f989c0bf201e/PyTorch/SpeechSynthesis/FastPitch/extract_mels.py
    for prog_idx, (item_id, mel_len) in enumerate(all_data, 1):
        dur = np.load(paths.alg / f'{item_id}.npy')
        pitch = np.load(paths.raw_pitch / f'{item_id}.npy')
        durs_cum = np.cumsum(np.pad(dur, (1, 0)))

        pitch_char = np.zeros((dur.shape[0],), dtype=np.float)
        for idx, a, b in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
            values = pitch[a:b][np.where(pitch[a:b] != 0.0)[0]]
            values = values[np.where(values < MAX_FREQ)[0]]
            pitch_char[idx] = np.mean(values) if len(values) > 0 else 0.0
        #print(f'{item_id} {pitch_char}')
        #print(f'{item_id} {text_dict[item_id]} {dur} {pitch_char}')
        phoneme_pitches.append((item_id, pitch_char))
        bar = progbar(prog_idx, len(all_data))
        message = f'{bar} {prog_idx}/{len(all_data)} '
        stream(message)

    mean, std = normalize(phoneme_pitches)
    print(f'mean {mean} std {std} ')

    """
    phoneme_pitches_interp = []
    for item_id, phoneme_pitch in phoneme_pitches:
        print(f'{item_id} {phoneme_pitch}')
        phoneme_pitch[np.where(phoneme_pitch==0.0)[0]] = np.nan
        phoneme_pitch_interp = interpolate(phoneme_pitch)
        phoneme_pitches_interp.append((item_id, phoneme_pitch_interp))
        print(f'dilated: {item_id} {phoneme_pitch_interp}')
    """
    print('done. Saving.')

    for item_id, phoneme_pitch in phoneme_pitches:

        np.save(paths.phon_pitch / f'{item_id}.npy', phoneme_pitch, allow_pickle=False)




    #plt.figure()
    #fig = plot_mel(mel)
    #plt.twinx()
    #draw_pitch(pitch, time_norm=(mel.shape[1] + 3) / snd.duration)
    #plt.savefig(f"/tmp/pitch_new_{file.stem}.png")