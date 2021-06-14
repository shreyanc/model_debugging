from datasets.dataset_base import DatasetBase

from utils import *
from paths import *
import logging

logger = logging.getLogger()

def get_deam_label_stats(full_songs_subset=False):
    if full_songs_subset:
        ann = pd.read_csv(path_deam_annotations_static_2001_2058)
    else:
        ann = pd.read_csv(path_deam_annotations_static_1_2000)

    ann = ann[['song_id', 'arousal_mean', 'valence_mean']]
    return ann.agg(['mean', 'std', 'min', 'max'])


class DEAMDataset(DatasetBase):
    def __init__(self, **kwargs):
        if kwargs.get('annotations') is None:
            kwargs['annotations'] = path_deam_annotations_static_1_2000
        super().__init__(name='deam', **kwargs)

    def _load_annotations(self, annotations):
        ann = pd.read_csv(annotations)
        ann = ann[['song_id', 'arousal_mean', 'valence_mean']]
        return ann

    def _get_label_names_from_annotations_df(self):
        return ['arousal_mean', 'valence_mean']

    def _get_dataset_label_stats(self):
        return self.annotations.agg(['mean', 'std', 'min', 'max'])

    def _get_id_col(self):
        return 'song_id'

    def _get_audio_path_from_id(self, songid):
        return os.path.join(path_deam_audio_dir, str(songid) + '.mp3')

