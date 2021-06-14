import torch
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

from datasets.dataset_base import DatasetBase
from datasets.dataset_utils import slice_func, normalize_spec, get_dataset_stats
from helpers.audio import MadmomAudioProcessor
import pandas as pd
import numpy as np
from torchvision.transforms import Normalize

from helpers.specaugment import SpecAugment
from utils import *
from paths import *
import logging

logger = logging.getLogger()


def load_midlevel_exclude_piano():
    meta = pd.read_csv(path_midlevel_metadata, sep=';')
    meta_piano = pd.read_csv(path_midlevel_metadata_piano, sep=',')
    no_piano = list(set(meta['song id'].values) - set(meta_piano['song id'].values))
    return np.array(no_piano), meta_piano['song id'].values


def load_midlevel_domain_split(source):
    if source == 'piano':
        return load_midlevel_exclude_piano()
    meta = pd.read_csv(path_midlevel_metadata, sep=';')
    assert source in meta['Source'].unique(), print(f"Unknown source {source}. Available: {meta['Source'].unique()}")

    sc_ids = meta[meta['Source'] == source]['song id']
    not_sc_ids = meta[~meta['song id'].isin(sc_ids)]['song id']

    assert len(sc_ids) > 0 and len(not_sc_ids) > 0, print(f"Can't split with {source}")
    return not_sc_ids.values, sc_ids.values


def load_midlevel_extra(instrument):
    meta = pd.read_csv(path_midlevel_metadata, sep=';')
    meta_solo_instruments = pd.read_csv(path_midlevel_metadata_instruments, sep=',')
    ids_selected = meta_solo_instruments[meta_solo_instruments['Domain'] == instrument]['song id'].values
    ids_not_selected = list(set(meta['song id'].values) - set(ids_selected))
    return ids_not_selected, ids_selected


def load_midlevel_aljanaki(ids=None, exclude=None, seed=None, tsize=0.08):
    rand_state = check_random_state(seed)
    meta = pd.read_csv(path_midlevel_metadata, sep=';')
    annotations = pd.read_csv(path_midlevel_annotations)
    if ids is not None:
        meta = meta[meta['song id'].isin(ids)]
        annotations = annotations[annotations['song_id'].isin(ids)]

    assert meta['song id'].equals(annotations['song_id']), "Song IDs in metadata file does not equal those in annotations file."

    artists = meta['Artist']
    if exclude is not None:
        meta = meta.drop(meta[meta.Source == exclude].index)
    if isinstance(tsize, int):
        test_set_size = tsize
    else:
        test_set_size = int(tsize * len(meta))
    artist_value_counts = artists.value_counts()
    single_artists = artist_value_counts.index[artist_value_counts == 1]
    assert len(single_artists) >= test_set_size, "Single artist test set size is greater than number of single artists in dataset."

    single_artists = single_artists.sort_values()
    selected_artists = rand_state.choice(single_artists, test_set_size, replace=False)
    selected_tracks_for_test = meta[meta['Artist'].isin(selected_artists)]

    test_song_ids = selected_tracks_for_test['song id'].values
    train_song_ids = annotations[~annotations['song_id'].isin(test_song_ids)]['song_id'].values

    return train_song_ids, test_song_ids


aljanaki_split = load_midlevel_aljanaki


def get_label_medians():
    annotations = pd.read_csv(path_midlevel_annotations)
    return annotations.median().values[1:]


class MidlevelDataset(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(name='midlevel', annotations=path_midlevel_annotations, **kwargs)

    def _load_annotations(self, annotations):
        ann = pd.read_csv(annotations)
        ann.columns = ann.columns.str.replace(' ', '')
        return ann

    def _get_label_names_from_annotations_df(self):
        return self.annotations.columns[1:]

    def _get_dataset_label_stats(self):
        return self.annotations.agg(['mean', 'std', 'min', 'max'])

    def _get_id_col(self):
        return self.annotations.columns[0]

    def _get_audio_path_from_id(self, songid):
        return os.path.join(path_midlevel_audio_dir, str(songid) + '.mp3')


class MidlevelBinaryDataset(MidlevelDataset):
    def __init__(self, *args, **kwargs):
        super(MidlevelBinaryDataset, self).__init__(*args, **kwargs)
        self.midlevel_medians = get_label_medians() / 10

    def __getitem__(self, ind):
        song_id = self.song_id_list[ind]
        audio_path = os.path.join(path_midlevel_audio_dir, str(song_id) + '.mp3')

        x = self._get_spectrogram(audio_path, self.dataset_cache_dir).spec

        slice_length = self.processor.times_to_frames(self.duration)
        x_sliced, start_time, end_time = slice_func(x, slice_length, self.processor, mode='random')
        x_sliced = normalize_spec(x_sliced, dset_name=self.dset_name, aud_processor=self.processor)

        x_aug = self.augment(x_sliced).astype(np.float32)

        midlevels = self.annotations.loc[self.annotations.song_id == song_id]
        midlevels = midlevels.drop(labels=['song_id'] + self.drop_labels, axis=1).to_numpy(dtype=np.float32)
        midlevels = midlevels / 10

        binary_midlevels = midlevels > self.midlevel_medians
        binary_midlevels = binary_midlevels.astype(int)

        return audio_path, torch.from_numpy(x_aug), torch.from_numpy(binary_midlevels).squeeze()


class MidlevelGenericDataset(Dataset):
    def __init__(self, cache_dir=path_cache_fs,
                 name='midlevel_generic',
                 duration=15,
                 annotations_df=None,
                 audio_paths_column_name='path',
                 aud_processor=None,
                 slice_mode='start',
                 **kwargs):

        self.dset_name = name
        if aud_processor is not None:
            self.processor = aud_processor
        else:
            self.processor = MadmomAudioProcessor(fps=31.3)
        self.duration = duration  # seconds
        self.dataset_cache_dir = os.path.join(cache_dir, name)
        self.aud_len = duration
        self.annotations = annotations_df
        self.audio_paths_column_name = audio_paths_column_name
        self.audio_paths_list = self.annotations[audio_paths_column_name].tolist()
        self.slice_mode = slice_mode

        if isinstance(kwargs.get('augment'), SpecAugment):
            self.augment = kwargs['augment']
        elif kwargs.get('augment') == 'none' or kwargs.get('augment') is None:
            self.augment = lambda x: x
        else:
            logger.info(f"WARNING: No spec augment function assigned -- got {kwargs.get('augment')}; should be SpecAugment instance or None or 'none'!")
            self.augment = lambda x: x

        if kwargs.get('label_noise'):
            self.label_noise_std = kwargs['label_noise']
        else:
            self.label_noise_std = 0.0

        if kwargs.get('normalizing_dset'):
            logger.info(f"WARNING (MidlevelGenericDataset - {self.dset_name}): Using {kwargs.get('normalizing_dset')} mean and std using default audio_processor parameters!")
            self.norm_mean, self.norm_std = get_dataset_stats(kwargs.get('normalizing_dset'))

        self.kwargs = kwargs

    def __getitem__(self, ind):
        audio_path = self.audio_paths_list[ind]

        x = self._get_spectrogram(audio_path, self.dataset_cache_dir).spec

        slice_length = self.processor.times_to_frames(self.duration)
        x_sliced, start_time, end_time = slice_func(x, slice_length, self.processor, mode=self.slice_mode)
        if self.kwargs.get('normalizing_dset'):
            x_sliced = normalize_spec(x_sliced, mean=self.norm_mean, std=self.norm_std)
        else:
            x_sliced = normalize_spec(x_sliced, dset_name=self.dset_name, aud_processor=self.processor)

        x_aug = self.augment(x_sliced).astype(np.float32)

        midlevels = self.annotations.loc[self.annotations[self.audio_paths_column_name] == audio_path]
        midlevels = midlevels.drop(labels=self.audio_paths_column_name, axis=1).to_numpy(dtype=np.float32).squeeze()
        midlevels = np.add(midlevels, self.label_noise_std * np.random.randn(len(midlevels)))

        return audio_path, torch.from_numpy(x_aug), torch.from_numpy(midlevels).squeeze()

    def _get_spectrogram(self, audio_path, dataset_cache_dir):
        specpath = os.path.join(dataset_cache_dir, self.processor.get_params.get("name"),
                                str(os.path.basename(audio_path).split('.')[0]) + '.specobj')
        specdir = os.path.split(specpath)[0]
        if not os.path.exists(specdir):
            print(specdir, os.path.exists(specdir))
            os.makedirs(specdir)
        try:
            return pickleload(specpath)
        except:
            print(f"Calculating spectrogram for {audio_path} and saving to {specpath}")
            spec_obj = self.processor(audio_path)
            pickledump(spec_obj, specpath)
            return spec_obj

    def __len__(self):
        return len(self.audio_paths_list)


class MidlevelGenericDatasetNpy(Dataset):
    def __init__(self, cache_dir=path_cache_fs,
                 name="midlevel",
                 df=None,
                 song_ids=None,
                 aud_len_samples=16000,
                 binarizer=None,
                 slice_mode='start',
                 **kwargs):
        self.dset_name = name
        self.binarizer = binarizer
        self.cache_dir = cache_dir
        self.aud_len_samples = aud_len_samples
        self.dataset_cache_dir = os.path.join(cache_dir, name, 'npy')
        self.song_ids = song_ids
        self.slice_mode = slice_mode
        # self.labels = read_file(path_mtgjamendo_raw_30s_labels_50artists)

    def __getitem__(self, ind):
        song_path = os.path.join(self.dataset_cache_dir, 'audio', str(self.song_ids[ind])+'.npy')
        # labels = get_tags_for_song(song_path, self.labels)['tags']
        # if self.binarizer is not None:
        #     labels = self.binarizer.transform(labels)
        npy_path = os.path.join(self.dataset_cache_dir, song_path)
        aud = np.load(npy_path)
        random_idx = int(np.floor(np.random.random(1) * (len(aud) - self.aud_len_samples)))
        aud = torch.tensor(np.array(aud[random_idx:random_idx + self.aud_len_samples]))
        labels = []
        return song_path, aud, labels

    def __len__(self):
        return len(self.song_ids)

if __name__ == '__main__':
    load_midlevel_aljanaki()
