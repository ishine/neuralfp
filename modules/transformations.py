from audiomentations import Compose,Shift,PitchShift,TimeStretch,AddImpulseResponse,FrequencyMask,TimeMask,ClippingDistortion,AddBackgroundNoise,Gain


class TransformNeuralfp:
    
    def __init__(self, ir_dir, noise_dir):
        
        self.train_transform = Compose([
            Shift(min_fraction=-0.2, max_fraction=0.2, rollover=False),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            AddImpulseResponse(ir_path=ir_dir, p=0.8),
            FrequencyMask(min_frequency_band=0.1, max_frequency_band=0.5,p=0.8),
            TimeMask(min_band_part=0.1, max_band_part=0.5),
            ClippingDistortion(),
            AddBackgroundNoise(sounds_path=noise_dir, min_snr_in_db=0, max_snr_in_db=10,p=0.8),
            Gain(),
            # Mp3Compression()
            ])
            
    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
