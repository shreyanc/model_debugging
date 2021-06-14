from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

from datasets.dataset_base import DatasetBase
from datasets.dataset_utils import load_audio
import pandas as pd

from utils import *
from paths import *
import logging

logger = logging.getLogger()


def load_pmemo_song_ids(seed):
    ann = pd.read_csv(path_pmemo_static_annotations)
    ids = ann.musicId.values
    tr, te = train_test_split(ids, test_size=0.1, random_state=seed)
    te, va = train_test_split(te, test_size=0.2, random_state=seed)
    return tr, va, te


def compute_pmemo_splits(tsize=0.2, seed=0):
    meta = pd.read_csv(path_pmemo_metadata)
    annotations = pd.read_csv(path_pmemo_static_annotations)
    artists = meta['artist']
    artist_value_counts = artists.value_counts()
    single_artists = artist_value_counts.index[artist_value_counts == 1]

    if isinstance(tsize, int):
        test_set_size = tsize
    else:
        test_set_size = int(tsize * len(meta))

    assert len(single_artists) >= test_set_size, "Single artist test set size is greater than number of single artists in dataset."
    single_artists = single_artists.sort_values()

    rand_state = check_random_state(seed)
    selected_artists = rand_state.choice(single_artists, test_set_size, replace=False)
    selected_tracks_for_test = meta[meta['artist'].isin(selected_artists)]

    test_song_ids = selected_tracks_for_test['musicId'].values
    train_song_ids = annotations[~annotations['musicId'].isin(test_song_ids)]['musicId'].values

    return train_song_ids, test_song_ids


def get_pmemo_label_stats():
    ann = pd.read_csv(path_pmemo_static_annotations)
    return ann.agg(['mean', 'std', 'min', 'max'])


class PMEmoDataset(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(name='pmemo', annotations=path_pmemo_static_annotations, **kwargs)

    def _load_annotations(self, annotations):
        ann = pd.read_csv(annotations)
        return ann

    def _get_label_names_from_annotations_df(self):
        return self.annotations.columns[1:]

    def _get_dataset_label_stats(self):
        return get_pmemo_label_stats()

    def _get_id_col(self):
        return self.annotations.columns[0]

    def _get_audio_path_from_id(self, songid):
        return os.path.join(path_pmemo_audio_dir, str(songid) + '.mp3')


class PMEmoGenericDatasetNpy(Dataset):
    def __init__(self, cache_dir=path_cache_fs,
                 name="pmemo",
                 aud_len_samples=16000,
                 **kwargs):
        self.dset_name = name
        self.cache_dir = cache_dir
        self.aud_len_samples = aud_len_samples
        self.dataset_cache_dir = os.path.join(cache_dir, name, 'npy')
        self.song_paths_list = list_files_deep(path_pmemo_audio_dir, full_paths=True)

    def __getitem__(self, ind):
        song_path = self.song_paths_list[ind]
        song_id = song_path.split('/')[-1].split('.')[0]
        npy_path = os.path.join(self.dataset_cache_dir, str(song_id) + '.npy')
        labels = []

        if not os.path.exists(os.path.dirname(npy_path)):
            os.makedirs(os.path.dirname(npy_path))

        try:
            aud = np.load(npy_path)
        except FileNotFoundError as e:
            aud = load_audio(song_path, 16000)
            np.save(npy_path, aud)

        random_idx = int(np.floor(np.random.random(1) * (len(aud) - self.aud_len_samples)))
        aud = torch.tensor(np.array(aud[random_idx:random_idx + self.aud_len_samples]))
        return song_path, aud, labels

    def __len__(self):
        return len(self.song_paths_list)


if __name__ == '__main__':
    seed = 311
    tr, te = compute_pmemo_splits(seed=seed)
    np.save(os.path.join(path_pmemo_root, 'splits', f'train_ids_seed={seed}'), tr)
    np.save(os.path.join(path_pmemo_root, 'splits', f'test_ids_seed={seed}'), te)
    # ids = load_pmemo_split('test', 42)
    # print(ids)
