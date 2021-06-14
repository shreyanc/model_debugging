import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.dataset_utils import slice_func
from utils import pickleload, pickledump
from paths import *
import logging
from helpers.audio import MadmomAudioProcessor

logger = logging.getLogger()

class DatasetBase(Dataset):
    def __init__(self, cache_dir=path_cache_fs,
                 name='',
                 duration=15,
                 select_song_ids=None,
                 aud_processor=None,
                 annotations=None,
                 normalize_labels=False,
                 scale_labels=True,
                 normalize_loudness=False,
                 padding='loop',
                 **kwargs):

        if aud_processor is not None:
            self.processor = aud_processor
        else:
            self.processor = MadmomAudioProcessor(fps=31.3, normalize_loudness=normalize_loudness)

        self.dataset_cache_dir = os.path.join(cache_dir, name)
        self.dset_name = name
        self.duration = duration  # seconds
        assert padding in ['zero', 'loop'], print(f"Padding should be in ['zero', 'loop'], is {padding}")
        self.padding = padding  # in slice_func, determines how to handle spectrogram slice when requested length is longer than spectrogram length

        try:
            self.annotations = self._load_annotations(annotations)
        except Exception as e:
            print("Exception occured while parsing annotations: ", e)
            assert isinstance(annotations, pd.DataFrame), print("annotations should be a pd.DataFrame object")
            self.annotations = annotations

        self.label_names = self._get_label_names_from_annotations_df()
        self.id_col = self._get_id_col()

        if select_song_ids is not None:
            selected_set = self.annotations[self.annotations[self.id_col].isin([int(i) for i in select_song_ids])]
        else:
            selected_set = self.annotations
        self.song_id_list = selected_set[self.id_col].tolist()

        self.normalize_labels = normalize_labels
        self.scale_labels = scale_labels

        self.label_stats = self._get_dataset_label_stats()

        if kwargs.get('slice_mode') is None:
            self.slice_mode = 'random'
        else:
            assert kwargs['slice_mode'] in ['random', 'start', 'end', 'middle']
            self.slice_mode = kwargs['slice_mode']

    def __getitem__(self, ind):
        song_id = self.song_id_list[ind]
        audio_path = self._get_audio_path_from_id(song_id)

        x = self._get_spectrogram(audio_path, self.dataset_cache_dir).spec

        slice_length = self.processor.times_to_frames(self.duration)
        x_sliced, start_time, end_time = slice_func(x, slice_length, self.processor, mode=self.slice_mode, padding=self.padding)
        x_sliced = x_sliced.astype(np.float32)

        labels = self._get_labels(song_id)

        return audio_path, torch.from_numpy(x_sliced), labels

    def _get_spectrogram(self, audio_path, dataset_cache_dir):
        specpath = os.path.join(dataset_cache_dir, self.processor.get_params.get("name"), str(os.path.basename(audio_path).split('.')[0]) + '.specobj')
        specdir = os.path.split(specpath)[0]
        if not os.path.exists(specdir):
            os.makedirs(specdir)
        try:
            return pickleload(specpath)
        except Exception as e:
            print(f"Could not load {specpath} -- {e}")
            print(f"Calculating spectrogram for {audio_path} and saving to {specpath}")
            spec_obj = self.processor(audio_path)
            pickledump(spec_obj, specpath)
            return spec_obj

    def __len__(self):
        return len(self.song_id_list)

    def _load_annotations(self, annotations):
        pass

    def _get_label_names_from_annotations_df(self):
        raise NotImplementedError

    def _get_dataset_label_stats(self):
        pass

    def _get_id_col(self):
        raise NotImplementedError

    def _get_audio_path_from_id(self, songid):
        raise NotImplementedError

    def _get_labels(self, song_id):
        labels = self.annotations.loc[self.annotations[self.id_col] == song_id][self.label_names]
        # ann = self.get_labels_df()
        # labels = ann.loc[ann[self.id_col] == song_id][self.label_names]

        if self.normalize_labels:
            labels -= self.label_stats.loc['mean'][self.label_names].values
            labels /= self.label_stats.loc['std'][self.label_names].values

        if self.scale_labels:
            labels -= self.label_stats.loc['min'][self.label_names].values
            labels /= self.label_stats.loc['max'][self.label_names].values - self.label_stats.loc['min'][self.label_names].values
            labels *= 2
            labels -= 1

        return torch.from_numpy(labels.values).squeeze()

    def get_labels_df(self):
        ann = self.annotations.copy()
        if self.normalize_labels:
            ann[self.label_names] -= self.label_stats.loc['mean'][self.label_names].values
            ann[self.label_names] /= self.label_stats.loc['std'][self.label_names].values

        if self.scale_labels:
            ann[self.label_names] -= self.label_stats.loc['min'][self.label_names].values
            ann[self.label_names] /= self.label_stats.loc['max'][self.label_names].values - self.label_stats.loc['min'][self.label_names].values
            ann[self.label_names] *= 2
            ann[self.label_names] -= 1

        return ann
