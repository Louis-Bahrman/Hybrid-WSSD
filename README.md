# A Hybrid Model for Weakly-Supervised Speech Dereverberation


[Paper](https://www.doi.org/10.1109/ICASSP49660.2025.10888095) &nbsp;
[Arxiv](https://doi.org/10.48550/arXiv.2502.06839) &nbsp;
[HAL](https://hal.science/hal-04931672/) &nbsp;
[Code](https://github.com/Louis-Bahrman/Hybrid-WSSD) &nbsp;
[Poster](docs/poster.pdf) &nbsp;
[Website](https://louis-bahrman.github.io/Hybrid-WSSD/)


This repository contains a Python program, `dereverberate.py`, to  dereverberate an audio file using our proposed supervision paradigms.
It also contains training code to apply this framework to new models and datasets.

## Installation

1. Clone this repository

```
git clone https://github.com/Louis-Bahrman/Hybrid-WSSD.git
cd Hybrid-WSSD
```

2. Install required dependencies

```
conda env create -f environment.yaml
conda activate hybrid_wssd
```

3. Download weights

```
wget 'https://partage.imt.fr/index.php/s/4oFHc7ik6nkP6br/download/icassp_logs.zip'
unzip icassp_logs.zip
```

## Usage
See:
```
python dereverberate.py -h
```

## Training new models or datasets

See [framework_details.md](framework_details.md)

## Citing

If you use this work in your research or business, please cite it using the following BibTeX entry:

```
@INPROCEEDINGS{10888095,
  author={Bahrman, Louis and Fontaine, Mathieu and Richard, Gaël},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={A Hybrid Model for Weakly-Supervised Speech Dereverberation}, 
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10888095}}
```