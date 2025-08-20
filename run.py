"""
requirements:
    torch
    numpy
    scipy
    scikit-learn
    tensorboardX
    flatten_dict
    tqdm
    attrs
"""
from neural_loss import train_loss_model, NeuralRankNDCG, NeuralRankRecall, LossModelConfig
import random
import numpy as np
import torch
from ranker import Ranker, load_config
from os import path
import neural_loss
from copy import deepcopy
import logging
import sys
import argparse
import os


# torch.autograd.set_detect_anomaly(True)


class Logger:
    def __init__(self):
        log_format = "[%(levelname)s] %(asctime)s - %(message)s"
        log_dateformat = "%H:%M:%S"
        logging.basicConfig(format=log_format, datefmt=log_dateformat, stream=sys.stdout, level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.episode = None
        self.phase = None
        self.stage = None
        self.epoch = None

    def info(self, message):
        msg = ''
        if self.epoch:
            msg = f"Epoch {self.epoch:<2}"
        if self.stage:
            msg = f"{self.stage:<5}, {msg}" if msg else f"{self.stage:<5}"
        if self.phase:
            msg = f"{self.phase:<10}, {msg}" if msg else f"{self.phase:<10}"
        self.logger.info(f"{msg}  --  {message}" if msg else message)

def main(loss_model_config: LossModelConfig, rank_config, logger, device, tag):
    # ********* rank model ***********
    # alter_model: None, 'mlp', 'set_transformer', 'attsets', 'mil_att', 'mil_gate_att'. None for Context-Aware Ranker
    ranker = Ranker(rank_config, alter_model=None, device=device, logger=logger)
    print('\n')

    # ********************************************************
    # *               training loss model                    *
    # ********************************************************
    logger.phase = 'LOSS-MODEL'

    if not os.path.exists('init_loss_model.pt'):
        train_data = neural_loss.load_dataset(loss_model_config.metric,
                                            loss_model_config.num_seqs,
                                            loss_model_config.repeat,
                                            loss_model_config.min_seq_len,
                                            loss_model_config.max_seq_len,
                                            loss_model_config.random_len,
                                            loss_model_config.target_type,
                                            stage='train',
                                            at_k=loss_model_config.at_k,
                                            logger=logger)
        val_data = neural_loss.load_dataset(loss_model_config.metric,
                                            loss_model_config.num_seqs_val,
                                            1,
                                            loss_model_config.min_seq_len,
                                            loss_model_config.max_seq_len,
                                            loss_model_config.random_len,
                                            loss_model_config.target_type,
                                            stage='val',
                                            at_k=loss_model_config.at_k,
                                            logger=logger)
        test_data = neural_loss.load_dataset(loss_model_config.metric,
                                            loss_model_config.num_seqs_test,
                                            1,
                                            loss_model_config.min_seq_len,
                                            loss_model_config.max_seq_len,
                                            loss_model_config.random_len,
                                            loss_model_config.target_type,
                                            stage='test',
                                            at_k=loss_model_config.at_k,
                                            logger=logger)
        train_ds = neural_loss.RankDataset(train_data, loss_model_config.padd_idx)
        val_ds = neural_loss.RankDataset(val_data, loss_model_config.padd_idx)
        test_ds = neural_loss.RankDataset(test_data, loss_model_config.padd_idx)
        train_dl = neural_loss.RankDataLoader(train_ds, loss_model_config.batch_size, True)
        val_dl = neural_loss.RankDataLoader(val_ds, loss_model_config.eval_batch_size, False)
        test_dl = neural_loss.RankDataLoader(test_ds, loss_model_config.eval_batch_size, False)

    loss_model = neural_loss.Model(model_dim=loss_model_config.model_dim,
                                   num_heads=loss_model_config.num_heads,
                                   num_layers=loss_model_config.num_layers,
                                   embedding_num=loss_model_config.embed_num,
                                   padd_idx=loss_model_config.padd_idx,
                                   out_active=loss_model_config.out_active)

    if os.path.exists('init_loss_model.pt'):
        loss_model.load_state_dict(torch.load('init_loss_model.pt', weights_only=True))
    else:
        loss_model = train_loss_model(loss_model_config, loss_model, train_dl, val_dl, test_dl, device, logger, tag)
        torch.save(loss_model.state_dict(), 'init_loss_model.pt')


    if loss_model_config.metric == 'dcg':
        metric_loss_model = NeuralRankNDCG(deepcopy(loss_model))
    elif loss_model_config.metric == 'avg_recall':
        metric_loss_model = NeuralRankRecall(deepcopy(loss_model))
    else:
        raise ValueError('metric must be dcg or avg_recall')


    # ********************************************************
    # *                 training ranking model               *
    # ********************************************************
    print()
    logger.phase = 'RANK-MODEL'
    ranking_results_on_test = ranker.train(metric_loss_model,
                            from_scratch=True,
                            tag=tag)

    return ranking_results_on_test



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', help='fold id', default=None)
    args = parser.parse_args()

    seeds = [1, ]
    # seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 42]


    # --- ranking task
    config_file_name = path.join('configs','config-ndcg_MQ2008.json')
    loss_metric='dcg'
    rank_job_dir='MQ2008'



    # ********* ranking model configuration ***********
    rank_run_id = 'run_1'     # experiment id, results are saved in <job_dir>/results/<run_id>
    rank_config = load_config(config_file_name, rank_job_dir, rank_run_id)

    # set fold through command line
    if args.fold is not None:
        rank_config.data.path = rank_config.data.path.replace('Fold1', f'Fold{args.fold}')
        print(rank_config.data.path)


    # ********* loss model configuration ***********
    loss_model_config = LossModelConfig(metric=loss_metric, max_seq_len=rank_config.data.slate_length)

    results = {}
    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        print('\n\n\n')
        print('=' * 80)
        print(f'\n  seed: {seed}\n')
        print('=' * 80)
        print()

        fold = args.fold or ''
        ranking_results = main(loss_model_config, rank_config, Logger(), device, fold)

        results[f'seed-{seed}'] = ranking_results


    print(f'\nfold{args.fold}: [')
    for seed, result in results.items():
        print(f'{result},')
    print(']')
