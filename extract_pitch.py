from pathlib import Path

from utils import hparams as hp
import parselmouth
import librosa
import numpy as np
import matplotlib.pyplot as plt

from utils.dataset import get_tts_datasets, filter_max_len
from utils.display import plot_mel
from utils.dsp import melspectrogram
from utils.files import unpickle_binary
from utils.paths import Paths

if __name__ == '__main__':
    hp.configure('hparams.py')
    train_data = unpickle_binary('data/train_dataset.pkl')
    val_data = unpickle_binary('data/val_dataset.pkl')
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)
    all_data = train_data + val_data
    all_data = filter_max_len(all_data)

    for item_id, mel_len in all_data:
        dur = np.load(paths.alg / f'{item_id}.npy')
        pitch = np.load(paths.raw_pitch / f'{item_id}.npy')
        dur_cum = np.cumsum(dur)
        phoneme_pitch = np.zeros(dur_cum.shape[0])
        for i in range(dur_cum.shape[0]-1):
            left, right = dur_cum[i], dur_cum[i+1]
            pitch_vals = pitch[left:right]
            pitch_vals = pitch_vals[pitch_vals != 0.0]
            pitch_mean = np.mean(pitch_vals) if len(pitch_vals) > 0.0 else 0.0
            phoneme_pitch[i] = pitch_mean

        print(dur_cum)
        print(phoneme_pitch)
        np.save(paths.phon_pitch / f'{item_id}.npy', phoneme_pitch)



    #plt.figure()
    #fig = plot_mel(mel)
    #plt.twinx()
    #draw_pitch(pitch, time_norm=(mel.shape[1] + 3) / snd.duration)
    #plt.savefig(f"/tmp/pitch_new_{file.stem}.png")