import numpy as np
from madmom.audio import SignalProcessor, FramedSignalProcessor, LogarithmicFilteredSpectrogramProcessor
from madmom.processors import Processor, SequentialProcessor


class Spectrogram():
    def __init__(self, spec, params_dict=None):
        self.spec = spec
        if params_dict is not None:
            self.params_dict = params_dict

    @property
    def params(self):
        return self.params_dict


class MadmomAudioProcessor(Processor):
    def __init__(self, sr=22050, hop_size=512, fps=None, frame_size=2048, num_mel_bands=149, num_bands_oct=24, mel=False, **kwargs):
        self.sr, self.n_fft, self.hop_length, self.n_mels = sr, frame_size, hop_size, num_mel_bands
        self.frame_size = self.n_fft

        if fps is not None:
            self.hop_length = sr // fps

        self.sig_proc = SignalProcessor(num_channels=1, sample_rate=sr, norm=True)
        self.fsig_proc = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size, fps=fps, origin='future')
        if mel:
            from madmom.audio.filters import MelFilterbank
            self.spec_proc = LogarithmicFilteredSpectrogramProcessor(num_bands=num_mel_bands, filterbank=MelFilterbank)
        else:
            self.n_mels = num_bands_oct
            self.name = "madmom_octfb"
            self.spec_proc = LogarithmicFilteredSpectrogramProcessor(num_bands=num_bands_oct, fmin=kwargs.get("fmin", 20),
                                                                     fmax=kwargs.get("fmax", 16000))

        self.name = "madmom" + '_' + str(sr) + '_' + str(self.n_fft) + '_' + str(self.n_fft) + '_' + str(int(self.hop_length))

        # if kwargs.get('normalize_loudness') is not None:
        self.normalize_loudness = kwargs.get('normalize_loudness', False)

    def process(self, file_path, **kwargs):
        sig = np.trim_zeros(self.sig_proc.process(file_path))
        fsig = self.fsig_proc.process(sig)
        spec = self.spec_proc.process(fsig)
        return Spectrogram(spec.transpose(), params_dict=self.get_params)

    def process_waveform(self, wf):
        sig = self.sig_proc(np.trim_zeros(wf))
        fsig = self.fsig_proc.process(sig)
        spec = self.spec_proc.process(fsig)
        return Spectrogram(spec.transpose(), params_dict=self.get_params)

    def times_to_frames(self, times):
        return np.floor(np.array(times) * self.sr / self.hop_length).astype(int)

    def frames_to_times(self, frames):
        return frames * self.hop_length / self.sr

    @property
    def get_params(self):
        param_dict = {"sr": self.sr,
                      "n_fft": self.n_fft,
                      "hop_length": self.hop_length,
                      "n_mels": self.n_mels,
                      "name": self.name}
        return param_dict

