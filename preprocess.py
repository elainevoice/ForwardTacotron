import argparse
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
import traceback

import pandas as pd
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.model_selection import train_test_split

from utils.display import *
from utils.dsp import *
from utils.files import get_files, pickle_binary
from utils.paths import Paths
from utils.text import clean_text
from utils.text.recipes import libri_tts, ljspeech


# Helper functions for argument types
def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n


parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--path', '-p', help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--extension', '-e', metavar='EXT', default='.wav', help='file extension to search for in dataset folder')
parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers, default=cpu_count()-1, help='The number of worker threads to use for preprocessing')
parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
args = parser.parse_args()

hp.configure(args.hp_file)  # Load hparams from file
if args.path is None:
    args.path = hp.wav_path

extension = args.extension
path = args.path


def convert_file(path: Path):
    y = load_wav(path)
    y = trim_long_silences(y)
    peak = np.abs(y).max()
    if hp.peak_norm or peak > 1.0:
        y /= peak

    mel = melspectrogram(y)
    if hp.voc_mode == 'RAW':
        quant = encode_mu_law(y, mu=2**hp.bits) if hp.mu_law else float_2_label(y, bits=hp.bits)
    elif hp.voc_mode == 'MOL':
        quant = float_2_label(y, bits=16)

    m_p = preprocess_wav(y, source_sr=hp.sample_rate)
    return mel.astype(np.float32), quant.astype(np.int64), m_p


class Preprocessor:

    def __init__(self, paths, text_dict):
        self.paths = paths
        self.text_dict = text_dict

    def process_wav(self, path: Path):
        try:
            wav_id = path.stem
            m, x, y_p = convert_file(path)
            np.save(self.paths.mel/f'{wav_id}.npy', m, allow_pickle=False)
            np.save(self.paths.quant/f'{wav_id}.npy', x, allow_pickle=False)
            text = self.text_dict[wav_id]
            text = clean_text(text)
            return wav_id, m.shape[-1], text, y_p
        except Exception as e:
            try:
                traceback.print_stack(e)
            except:
                pass
            return None, None, None, None



if __name__ == '__main__':

    wav_files = get_files(path, extension)
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)
    voice_encoder = VoiceEncoder()

    print(f'\n{len(wav_files)} {extension[1:]} files found in "{path}"\n')

    if len(wav_files) == 0:

        print('Please point wav_path in hparams.py to your dataset,')
        print('or use the --path option.\n')

    else:
        n_workers = max(1, args.num_workers)
        simple_table([
            ('Sample Rate', hp.sample_rate),
            ('Bit Depth', hp.bits),
            ('Mu Law', hp.mu_law),
            ('Hop Length', hp.hop_length),
            ('CPU Usage', f'{n_workers}/{cpu_count()}'),
            ('Num Validation', hp.n_val)
        ])
        print('Creating dict...')
        text_dict, speaker_dict = ljspeech(path)

        speakers = sorted(list(set(speaker_dict.values())))
        speaker_token_dict = {sp_id: i for i, sp_id in enumerate(speakers)}
        process_wav = Preprocessor(paths, text_dict).process_wav
        pool = Pool(processes=n_workers)
        dataset = []
        cleaned_texts = []
        print('\nCreating mels...')

        for i, (item_id, length, cleaned_text, m_p) in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
            if item_id is not None:
                semb = voice_encoder.embed_utterance(m_p)
                np.save(paths.semb/f'{item_id}.npy', semb, allow_pickle=False)

                if item_id in text_dict:
                    speaker_id = speaker_dict[item_id]
                    dataset += [(item_id, speaker_id, int(length))]
                    cleaned_texts += [(item_id, cleaned_text)]
                else:
                    print(f'Entry not found for id: {item_id}')
            bar = progbar(i, len(wav_files))
            message = f'{bar} {i}/{len(wav_files)} '
            stream(message)

        text_dict = {id: text for id, text in cleaned_texts}
        pickle_binary(text_dict, paths.data/'text_dict.pkl')

        dataset.sort()
        df = pd.DataFrame(data=dataset, columns=['item_id', 'speaker_id', 'length'])
        value_counts = df['speaker_id'].value_counts()
        to_remove = value_counts[value_counts < hp.min_speaker_count].index
        df = df[~df.speaker_id.isin(to_remove)]
        train_df, val_df = train_test_split(df, test_size=hp.n_val, random_state=42, stratify=df[['speaker_id']])

        train_dataset = list(train_df[['item_id', 'length']].itertuples(index=False, name=None))
        val_dataset = list(val_df[['item_id', 'length']].itertuples(index=False, name=None))
        val_dataset.sort(key=lambda d: -int(d[1]))

        # make sure certain speaker ids in hparams are first in the val dataset
        val_first, val_second = [], []
        first_val_speaker_ids = set(hp.val_speaker_ids)
        for v_id, v_len in val_dataset:
            val_speaker_id = speaker_dict[v_id]
            if val_speaker_id in first_val_speaker_ids:
                val_first.append((v_id, v_len))
                first_val_speaker_ids.remove(val_speaker_id)
            else:
                val_second.append((v_id, v_len))
        val_dataset = val_first + val_second

        text_dict = {id: text for id, text in cleaned_texts}

        print('\naveraging speaker embeddings...')
        avg_sembs = {}
        for item_id, _ in train_dataset + val_dataset:
            semb = np.load(str(paths.data/'semb'/f'{item_id}.npy'))
            s_id = speaker_dict[item_id]
            avg_semb = avg_sembs.get(s_id, np.zeros(semb.shape))
            avg_semb += semb
            avg_sembs[s_id] = avg_semb
        s_id_counter = Counter([speaker_dict[s_id] for s_id, _ in train_dataset + val_dataset])
        print(s_id_counter)
        for s_id, avg_semb in avg_sembs.items():
            avg_semb = avg_semb / s_id_counter[s_id]
            avg_semb = avg_semb / np.linalg.norm(avg_semb, 2)
            avg_sembs[s_id] = avg_semb

        pickle_binary(avg_sembs, paths.data/'speaker_emb_dict.pkl')
        pickle_binary(text_dict, paths.data/'text_dict.pkl')
        pickle_binary(speaker_dict, paths.data/'speaker_dict.pkl')
        pickle_binary(speaker_token_dict, paths.data/'speaker_token_dict.pkl')
        pickle_binary(train_dataset, paths.data/'train_dataset.pkl')
        pickle_binary(val_dataset, paths.data/'val_dataset.pkl')

        print('\n\nCompleted. Ready to run "python train_tacotron.py" or "python train_wavernn.py". \n')
