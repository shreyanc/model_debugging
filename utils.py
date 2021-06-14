
import torch
import pickle
import numpy as np

from helpers.audio import MadmomAudioProcessor
from paths import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_processor = MadmomAudioProcessor(fps=31.3)

ml_names = ['melody', 'articulation', 'rhythm_complexity', 'rhythm_stability', 'dissonance', 'tonal_stability', 'minorness']

def num_if_possible(s):
    try:
        return int(s)
    except ValueError:
        pass

    try:
        return float(s)
    except ValueError:
        pass

    if s == 'True':
        return True
    if s == 'False':
        return False

    return s

def list_files_deep(dir_path, full_paths=False, filter_ext=None):
    all_files = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(dir_path, '')):
        if len(filenames) > 0:
            for f in filenames:
                if full_paths:
                    all_files.append(os.path.join(dirpath, f))
                else:
                    all_files.append(f)

    if filter_ext is not None:
        return [f for f in all_files if os.path.splitext(f)[1] in filter_ext]
    else:
        return all_files


def save(model, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    try:
        torch.save(model.module.state_dict(), path)
    except AttributeError:
        torch.save(model.state_dict(), path)


def pickledump(data, fp):
    d = os.path.dirname(fp)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(fp, 'wb') as f:
        pickle.dump(data, f)


def pickleload(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)


def dumptofile(data, fp):
    d = os.path.dirname(fp)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(fp, 'w') as f:
        print(data, file=f)


def print_dict(dict, round):
    for k, v in dict.items():
        print(f"{k}:{np.round(v, round)}")


def log_dict(logger, dict, round=None, delimiter='\n'):
    log_str = ''
    for k, v in dict.items():
        if isinstance(round, int):
            try:
                log_str += f"{k}: {np.round(v, round)}{delimiter}"
            except:
                log_str += f"{k}: {v}{delimiter}"
        else:
            log_str += f"{k}: {v}{delimiter}"
    logger.info(log_str)


def load_model(model_weights_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    model.to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_weights_path))
    else:
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()


def inf(dl):
    """Infinite dataloader"""
    while True:
        for x in iter(dl): yield x


def choose_rand_index(arr, num_samples):
    return np.random.choice(arr.shape[0], num_samples, replace=False)

