# NeuralLoss: A Learnable Pretrained Surrogate Loss for Learning to Rank

This is the code for **NeuralLoss: A Learnable Pretrained Surrogate Loss for Learning to Rank**. The code is based on [allRank](https://github.com/allegro/allRank).

## Requirements

- torch==2.4.1
- numpy==2.0.1
- scipy==1.14.1
- scikit-learn==1.5.2
- tensorboardX==2.6.2.2
- flatten_dict==0.4.2
- attrs==24.2.0


## Running

- Run `python run.py` directly or `python run.py --fold fold_id` to specify the fold_id.


## Configuration

- Configuration file of NeuralLoss is `neural_loss/config.py`.
- Configuration files of allRank are in the fold `configs`.

## Reference

```
@ARTICLE{10969820,
author={Liu, Chen and Jiang, Cailan and Zhou, Lixin},
journal={ IEEE Transactions on Knowledge \& Data Engineering },
title={{ NeuralLoss: A Learnable Pretrained Surrogate Loss for Learning to Rank }},
year={2025},
volume={37},
number={07},
ISSN={1558-2191},
pages={4179-4192},
doi={10.1109/TKDE.2025.3562450},
url = {https://doi.ieeecomputersociety.org/10.1109/TKDE.2025.3562450},
publisher={IEEE Computer Society},
address={Los Alamitos, CA, USA},
month=jul}

```