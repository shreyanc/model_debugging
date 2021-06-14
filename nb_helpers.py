import os
import re
import time
import warnings

import librosa
import numpy as np
import pandas as pd
import torch
from audioLIME import lime_audio
from audioLIME.factorization_spleeter import SpleeterPrecomputedFactorization, SpleeterFactorization
from matplotlib import pyplot as plt
from tqdm import tqdm
import soundfile as sf

from datasets.dataset_utils import slice_func
from paths import path_pmemo_root, path_deam_root
from utils import audio_processor, ml_names, list_files_deep


def predict_fn_ml_emo(x_array, model):
    if torch.is_tensor(x_array):
        output = model(x_array.unsqueeze(0).unsqueeze(0))
    else:
        output = torch.tensor([0, 0, 0])  # TODO : implement

    emo_preds = output['output'].cpu().detach().numpy()
    ml_preds = output['concepts'].cpu().detach().numpy()
    return ml_preds, emo_preds


def find_index_of_song_id(songid, ids):
    idx = np.array(sorted(ids))
    return np.where(idx == songid)


def get_preds(dataset, model):
    sngs = []
    prds = []
    lbls = []
    mls = []

    for i in tqdm(range(len(dataset))):
        test_piece_path, spectrogram, labels = dataset[i]
        ml, emo = predict_fn_ml_emo(spectrogram, model)
        prds.append(emo.squeeze())
        mls.append(ml.squeeze())
        lbls.append(labels.squeeze().numpy())
        sngs.append(test_piece_path)

    return sngs, prds, lbls, mls


def get_explainer_predict_fn_ml(model, aud_len):
    def fn(x_array):
        slice_length = audio_processor.times_to_frames(aud_len)
        n_bands = 149
        x = torch.zeros(len(x_array), 1, n_bands, slice_length)
        for i in range(len(x_array)):
            spec = audio_processor.process(x_array[i]).spec
            spec_sliced, start_time, end_time = slice_func(spec, slice_length, audio_processor, mode='random', padding='loop')
            x[i] = torch.Tensor(spec_sliced).unsqueeze(0).unsqueeze(0)

        inputs = x
        output = model(inputs)
        ml_preds = output['concepts'].cpu().detach().numpy()
        return ml_preds

    return fn


def load_audio(aud_path, target_sr):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        waveform, _ = librosa.load(aud_path, mono=True, sr=target_sr)
    return waveform


def generate_explanations(test_pieces, ml_idx_list, model, audio_dir, aud_len, use_effects=True, use_precomupted=True):
    explanations_list = []
    explanations_weights_list = []
    explanations_audio_paths_list = []
    original_audio_paths_list = []
    neighborhood_labels = []

    spleeter_precomputed_factorizations_pmemo = '/home/shreyan/mounts/cp/projects/midlevel_explanations/source_separation/spleeter_precomputed_factorization/pmemo/'
    for i, test_piece in tqdm(enumerate(test_pieces), total=len(test_pieces)):
        test_piece_path = os.path.join(audio_dir, test_piece)
        data_provider = load_audio(test_piece_path, target_sr=22050)

        if use_precomupted:
            spleeter_factorization = SpleeterPrecomputedFactorization(test_piece_path,
                                                                      target_sr=22050,
                                                                      temporal_segmentation_params=1,
                                                                      composition_fn=None,
                                                                      spleeter_sources_path=spleeter_precomputed_factorizations_pmemo,
                                                                      recompute=False
                                                                      )
        else:
            spleeter_factorization = SpleeterFactorization(data_provider,
                                                           temporal_segmentation_params=1,
                                                           composition_fn=None,
                                                           model_name='spleeter:5stems')

        aud, ins = spleeter_factorization.initialize_components()
        explainer = lime_audio.LimeAudioExplainer(verbose=False, absolute_feature_sort=False)

        explanation = explainer.explain_instance(factorization=spleeter_factorization,
                                                 predict_fn=get_explainer_predict_fn_ml(model, aud_len),
                                                 # top_labels=1,
                                                 num_reg_targets=7,
                                                 num_samples="exhaustive",
                                                 batch_size=8
                                                 )

        neighborhood_labels.append(explanation.neighborhood_labels)
        song_id = test_piece_path.split('/')[-1].split('.')[0]
        out_dir = f"../misc/spleeter_output/{song_id}"
        num_components = 1
        ret_dict = {}

        if use_effects:
            label = ml_idx_list[i]
        else:
            label = np.argmax(explanation.neighborhood_labels.std())

        top_components, component_indeces, weights = explanation.get_sorted_components(label,
                                                                                       positive_components=True,
                                                                                       negative_components=False,
                                                                                       num_components=num_components,
                                                                                       return_indeces=True)

        ex = [ins[i] for i in component_indeces]
        # print(ml_names[label], ex)
        explanations_list.append(ex)
        if len(weights) < num_components:
            weights = np.hstack([weights, np.array([0] * (num_components - len(weights)))])
            top_components.append(top_components[0])
        else:
            weights = weights[:num_components]
            top_components = top_components[:num_components]
        weights = weights / sum(weights)
        # print(weights, '\n')
        explanations_weights_list.append(weights)
        exp_mix = top_components[0]  # *weights[0] + top_components[1]*weights[1]

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        sf.write(os.path.join(out_dir, f"{ml_names[label]}_explanation.wav"), exp_mix, 22050)
        explanations_audio_paths_list.append(os.path.join(out_dir, f"{ml_names[label]}_explanation.wav"))
        if len(component_indeces) > 0:
            ret_dict.update({ml_names[label]: ins[component_indeces[0]]})

        sf.write(os.path.join(out_dir, f"original.wav"), spleeter_factorization._original_mix, 22050)
        original_audio_paths_list.append(os.path.join(out_dir, f"original.wav"))

    return explanations_list, explanations_weights_list, explanations_audio_paths_list, original_audio_paths_list, neighborhood_labels


def get_preds_df_for_model(model_path, model, dset_name, dataset, cache_dir=None):
    if cache_dir is None:
        preds_cache_path = f'./cached_material/{model_path.split("/")[-1].split(".")[0]}_{dset_name}_preds_df.csv'
    else:
        preds_cache_path = os.path.join(cache_dir, f'{model_path.split("/")[-1].split(".")[0]}_{dset_name}_preds_df.csv')

    if not os.path.exists(preds_cache_path):
        songs, preds, labels, midls = get_preds(dataset, model)

        preds_arr = np.vstack(preds)
        labels_arr = np.vstack(labels)
        midls_arr = np.vstack(midls)

        song_ids = [s.split('/')[-1] for s in songs]
        preds_df = pd.DataFrame(zip(song_ids, preds_arr[:, 0], preds_arr[:, 1], labels_arr[:, 0], labels_arr[:, 1]),
                                columns=["song_ids", "ar_preds", "va_preds", "ar_labels", "va_labels"])
        preds_df.insert(loc=preds_df.shape[1], column="ar_err", value=preds_df['ar_preds'] - preds_df['ar_labels'])
        preds_df.insert(loc=preds_df.shape[1], column="va_err", value=preds_df['va_preds'] - preds_df['va_labels'])
        ml_df = pd.DataFrame(midls_arr, columns=ml_names)
        preds_df = pd.concat([preds_df, ml_df], axis=1)
        # preds_df.head()

        # genres_df = pd.read_csv('/home/shreyan/mounts/fs/home@fs/RUNS/genres/199ed_2021-02-27_00-36-50_dset=pmemo/pmemo_tags.csv')
        if 'pmemo' in dset_name:
            dset_root = path_pmemo_root
            genres_df = pd.read_csv(os.path.join(dset_root, f'pmemo_genre_tags_musicnn.csv'))
        elif 'deam' in dset_name:
            dset_root = path_deam_root
            genres_df = pd.read_csv(os.path.join(dset_root, f'deam_genre_tags_musicnn.csv'))
        else:
            dset_root = None
            genres_df = None
        # genres_df.head()
        preds_df = preds_df.merge(genres_df, how='inner', on=['song_ids'])
        # preds_df.head()

        preds_df.to_csv(preds_cache_path, index=False)

    else:
        preds_df = pd.read_csv(preds_cache_path)

    return preds_df


def get_top_errors(num=5, of='valence', err_type='over', preds_df=None, plot=False):
    if of == 'valence':
        err_col = 'va_err'
    elif of == 'arousal':
        err_col = 'ar_col'
    else:
        err_col = of

    top_errors = preds_df.sort_values(by=err_col, ascending=(err_type == 'under')).iloc[:num]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(np.arange(0, num), top_errors['va_labels'], 'o', label='actual')
        ax.plot(np.arange(0, num), top_errors['va_preds'], 'D', label='predicted')
        ax.vlines(np.arange(0, num), top_errors['va_labels'], top_errors['va_preds'])
        ax.axhline(y=0, color='k', linewidth=0.4)
        ax.set_xticks(np.arange(0, num))
        ax.set_xticklabels(labels=top_errors['song_ids'], rotation=45)
        ax.legend()
        plt.show()

    return top_errors


def plot_fraction_of_genre_vs_error(preds_df, emotion, genre, title=None, num_quantiles=10):
    if emotion == 'valence':
        err_col = 'va_err'
    else:
        err_col = 'ar_err'

    quantiles = preds_df[[err_col]].quantile(np.arange(0.1, 1.0, 1 / num_quantiles)).values.squeeze()

    from collections import Counter
    genre_percent = []
    for i in range(len(quantiles) - 1):
        q1, q2 = quantiles[i], quantiles[i + 1]
        z = preds_df[(q1 < preds_df[err_col]) & (preds_df[err_col] < q2)]
        genre_percent.append(Counter(z['tags'])[genre] / len(z))

    import matplotlib
    matplotlib.rcParams.update({'errorbar.capsize': 8})
    matplotlib.rcParams.update({'font.size': 15})
    matplotlib.rcParams.update({'lines.linewidth': 3})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([(quantiles[i] + quantiles[i + 1]) / 2 for i in range(len(quantiles) - 1)], genre_percent, marker='o', ms=15)
    ax.set_xlabel(f'{emotion} error')
    ax.set_ylabel(f'fraction of songs tagged "{genre}"')
    ax.set_title(title)
    plt.savefig('./plots/quantiles.png')
    # plt.show()
    # return fig


def get_model_path(models_root=None, split=None, avdset=None, run=None, hash_id=None):
    if models_root is None:
        models_root = '/home/shreyan/mounts/fs/home@fs/RUNS/limejuice_train_explainable_ml_av'

    all_models = os.listdir(models_root)

    if split is not None:
        assert avdset in ['deam', 'pmemo', 'combined']
        m = [model_path for model_path in all_models if f'split={split}' in model_path and f'avdset={avdset}' in model_path]
        if len(m) != 1:
            if hash_id is not None:
                selected_model = [model_path for model_path in all_models if hash_id in model_path]
                assert len(selected_model) == 1, print(f"multiple models exist with hash {hash_id}!")
                selected_model_root = selected_model[0]
            else:
                print(f"hash id needed, as multple models exist with split {split} and avdset {avdset}")
                raise Exception
        else:
            selected_model_root = m[0]

    else:
        assert hash_id is not None
        selected_model = [model_path for model_path in all_models if hash_id in model_path]
        assert len(selected_model) == 1, print(f"multiple models exist with hash {hash_id}!")
        selected_model_root = selected_model[0]
        re_split = re.search('split=([0-9]+)', selected_model_root)
        if re_split:
            split = re_split.group(1)
        else:
            split = 0

    all_runs = list_files_deep(os.path.join(models_root, selected_model_root, 'saved_models'), full_paths=True)
    selected_run = [run_name for run_name in all_runs if f'_{run}' in run_name]
    selected_model_path = selected_run[0]

    return selected_model_path, os.path.join(models_root, selected_model_root), int(split)


def plot_weights(w):
    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    im = ax.imshow(w, cmap='bwr', vmax=np.abs(w).max(), vmin=-np.abs(w).max())
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels=('arousal', 'valence'))
    ax.set_xticks(np.arange(0, 7))
    ax.set_xticklabels(labels=ml_names, rotation='vertical')
    fig.colorbar(im)
    plt.title("Weights")
    plt.show()
