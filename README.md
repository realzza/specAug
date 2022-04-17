# specAug

This is a local implementation of spectrogram augmentation on the fly in training stage. Originated from [SpecAugment](https://github.com/DemisEom/SpecAugment).

## Files
[`augmentation.py`](augmentation.py) currently supports `frequecy_mask`, `time_mask` and `combined`. More local spec augment methods will be added in the future.

[`aug_dataset.py`](aug_dataset.py) contains random local augmentation.