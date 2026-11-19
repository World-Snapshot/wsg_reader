# wsg_reader
Separate the read code of wsg and make it into the most concise read tool. The original code can be found at [this link](https://github.com/World-Snapshot/wsg).

## Abstract

We know that if we want to estimate information from RGB and then inject it into WSG, this process is of course quite complex and requires different environments. For instance, the optical flow environment, or the human pose environment. However, if we only read and visualize, then only a few packages are needed to view WSG and WSV files:

```python
conda create -n wsg_reader python=3.12 -y
conda activate wsg_reader

pip install torch torchvision
pip install numpy scikit-image matplotlib tqdm opencv-python ffmpeg bpy psutil Pillow matplotlib pandas pyarrow datasets
```

## Note

- But, for this repo, it is recommended to use the environment of the WaveGen: It is already a combination of the [WaveGen](https://github.com/World-Snapshot/WaveGen) + [ControlWave](https://github.com/World-Snapshot/ControlWave) + [wave2pixel](https://github.com/World-Snapshot/wave2pixel) + [wsg_reader](https://github.com/World-Snapshot/wsg_reader) environments.

- Additionally, this warehouse has been applied to the [wave2pixel](https://github.com/World-Snapshot/wave2pixel) and [The results of Augustus](https://github.com/World-Snapshot/The-results-of-Augustus) and other repositories, but generally speaking, they are not the latest versions.

## Other

I did consider packaging this as a pip package, but making the code more accessible would make it easier to modify and optimize, so I didn't do that.

## License

The original WSG/WSV reading, decoding, and visualization code and
documentation in this repository are released under the
[WSM Non-commercial Research License](LICENSE). Use is limited to academic,
research, experimental, personal, and other noncommercial purposes. For
commercial-use rights, contact Rutgers at
`compliance@research.rutgers.edu` and reference Rutgers Docket #2027-020.
Any Rutgers-owned model artifacts distributed with this repository, including
trained or fine-tuned weights, checkpoints, adapters, and `.pth` files, are
covered by the same license.

Third-party dependencies, pretrained models, datasets, and adapted components
retain their respective licenses and terms. The RU-NCRL does not relicense or
claim ownership of those third-party materials.
