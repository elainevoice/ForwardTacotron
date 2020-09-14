from utils.files import unpickle_binary, pickle_binary
from utils.paths import Paths
import shutil

def get_data(path):
    train_data = unpickle_binary(path / 'train_dataset.pkl')
    val_data = unpickle_binary(path / 'val_dataset.pkl')
    text_dict = unpickle_binary(path / 'text_dict.pkl')
    speaker_dict = unpickle_binary(path / 'speaker_dict.pkl')
    speaker_token_dict = unpickle_binary(path / 'speaker_token_dict.pkl')
    return {
        'train_data': train_data,
        'val_data': val_data,
        'text_dict': text_dict,
        'speaker_dict': speaker_dict,
        'speaker_token_dict': speaker_token_dict
    }


if __name__ == '__main__':
    paths_en = Paths('data_en', 'dummy', 'dummy')
    paths_de = Paths('data_de', 'dummy', 'dummy')
    paths_merged = Paths('data_en_de', 'dummy', 'dummy')

    data_en = get_data(paths_en.data)
    data_de = get_data(paths_de.data)

    train_data = data_de['train_data'] + data_en['train_data']
    val_data = data_de['val_data'] + data_en['val_data']
    text_dict = {**data_de['text_dict'], **data_en['text_dict']}
    speaker_dict = {**data_de['speaker_dict'], **data_en['speaker_dict']}
    speakers = sorted(list(set(speaker_dict.values())))
    speaker_token_dict = {sp_id: i for i, sp_id in enumerate(speakers)}
    print(speaker_token_dict)

    pickle_binary(train_data, paths_merged.data / 'train_data.pkl')
    pickle_binary(val_data, paths_merged.data / 'val_data.pkl')
    pickle_binary(text_dict, paths_merged.data / 'text_dict.pkl')
    pickle_binary(speaker_dict, paths_merged.data / 'speaker_dict.pkl')
    pickle_binary(speaker_token_dict, paths_merged.data / 'speaker_token_dict.pkl')

    for item_id, _ in data_de['train_data'] + data_de['val_data']:
        try:
            shutil.copy(paths_de.mel/f'{item_id}.npy', paths_merged.mel/f'{item_id}.npy')
            shutil.copy(paths_de.alg/f'{item_id}.npy', paths_merged.alg/f'{item_id}.npy')
        except Exception as e:
            print(e)
    for item_id, _ in data_en['train_data'] + data_en['val_data']:
        try:
            shutil.copy(paths_en.mel/f'{item_id}.npy', paths_merged.mel/f'{item_id}.npy')
            shutil.copy(paths_en.alg/f'{item_id}.npy', paths_merged.alg/f'{item_id}.npy')
        except Exception as e:
            print(e)
