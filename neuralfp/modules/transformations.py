from audiomentations import Compose,Shift,PitchShift,TimeStretch,AddImpulseResponse,FrequencyMask,TimeMask,ClippingDistortion,AddBackgroundNoise,Gain


class TransformNeuralfp:
    
    def __init__(self, ir_dir, noise_dir, sample_rate):
        self.sample_rate = sample_rate
        self.train_transform_i = Compose([
            Shift(min_fraction=-0.1, max_fraction=0.1, rollover=False),
            # PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            # AddImpulseResponse(ir_path=ir_dir, p=0.6),
            FrequencyMask(min_frequency_band=0.1, max_frequency_band=0.5,p=0.8),
            TimeMask(min_band_part=0.1, max_band_part=0.5),
            # ClippingDistortion(),
            # AddBackgroundNoise(sounds_path=noise_dir, min_snr_in_db=0, max_snr_in_db=10,p=0.9),
            # Gain(),
            # Mp3Compression()
            ])
        
        self.train_transform_j = Compose([
            Shift(min_fraction=-0.1, max_fraction=0.1, rollover=False),
            # PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            AddImpulseResponse(ir_path=ir_dir, p=0.6),
            FrequencyMask(min_frequency_band=0.1, max_frequency_band=0.5,p=0.8),
            TimeMask(min_band_part=0.1, max_band_part=0.5),
            ClippingDistortion(),
            # AddBackgroundNoise(sounds_path=noise_dir, min_snr_in_db=0, max_snr_in_db=5,p=0.9),
            Gain(),
            # Mp3Compression()
            ])
            
    def __call__(self, x):
        return self.train_transform_i(x,self.sample_rate), self.train_transform_j(x,self.sample_rate)