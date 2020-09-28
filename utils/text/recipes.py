from multiprocessing.pool import Pool

from utils.display import progbar, stream
from utils.files import get_files
from pathlib import Path
from typing import Union, Tuple


def ljspeech(path: Union[str, Path]):
    csv_file = get_files(path, extension='.csv')

    assert len(csv_file) == 1

    text_dict = {}
    speaker_dict = {}

    with open(csv_file[0], encoding='utf-8') as f :
        for line in f :
            split = line.split('|')
            text_dict[split[0]] = split[1]
            speaker_dict[split[0]] = 'karlsson'

    return text_dict, speaker_dict


def vctk(path: Union[str, Path], n_workers, extension='.txt') -> Tuple[dict, dict]:
    files = list(Path(path).glob('**/*' + extension))
    text_dict = {}
    speaker_id_dict = {}
    pool = Pool(processes=n_workers)
    for i, (file, text) in enumerate(pool.imap_unordered(read_line, files), 1):
        bar = progbar(i, len(files))
        message = f'{bar} {i}/{len(files)} '
        text_id = file.name.replace(extension, '')
        speaker_id = file.parent.stem
        text_dict[text_id] = text
        speaker_id_dict[text_id] = speaker_id
        stream(message)
    return text_dict, speaker_id_dict


def libri_tts(path: Union[str, Path], n_workers, extension='.normalized.txt') -> Tuple[dict, dict]:
    files = list(Path(path).glob('**/*' + extension))
    text_dict = {}
    speaker_id_dict = {}
    pool = Pool(processes=n_workers)
    for i, (file, text) in enumerate(pool.imap_unordered(read_line, files), 1):
        bar = progbar(i, len(files))
        message = f'{bar} {i}/{len(files)} '
        text_id = file.name.replace(extension, '')
        speaker_id = int(file.parent.parent.stem)
        text_dict[text_id] = text
        speaker_id_dict[text_id] = speaker_id
        stream(message)
    return text_dict, speaker_id_dict


def read_line(file: Path) -> Tuple[Path, str]:
    with open(str(file), encoding='utf-8') as f:
        line = f.readlines()[0]
    return file, line
