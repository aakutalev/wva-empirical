import datetime
import logging
import sys
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from pydoc import locate

import joblib
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

SCRIPT_NAME = "evaluate"
logger = logging.getLogger(SCRIPT_NAME)

student = {0:  0.,
           1:  12.7062, 2:  4.3027, 3:  3.1824, 4:  2.7764, 5:  2.5706,
           6:  2.4469,  7:  2.3646, 8:  2.3060, 9:  2.2622, 10: 2.2281,
           11: 2.2010,  12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314,
           16: 2.1199,  17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860}


class MyMNIST(Dataset):
    def __init__(self, inputs, targets):
        self.data = inputs
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def train_model(model, train_set, test_sets, batch_size=100, epochs=1, validation_samples=0):
    """
    Single dataset training
    """
    if validation_samples > 0:
        train_set = MyMNIST(train_set[0].data[:-validation_samples], train_set[0].targets[:-validation_samples])

    num_iters = int(np.ceil(train_set.data.shape[0] * epochs / batch_size))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model.train()
    idx = 0
    for epoch in range(epochs):
        for inputs, labels in iter(train_loader):
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            model.step(inputs=inputs, labels=labels)
            idx += 1
            if idx % 67 == 0:
                print(f'Training {idx}/{num_iters} iterations done.\r', end='')
    logger.info(f'Training {num_iters}/{num_iters} iterations done. ')

    model.eval()
    accuracy = 0.
    with torch.no_grad():
        for t, test_set in enumerate(test_sets):
            inputs = torch.tensor(test_set[1].data, device=model.device)
            logits = model.forward(inputs)
            results = logits.max(-1).indices
            accuracy += np.mean(results.cpu().numpy() == test_set[1].targets)
    accuracy /= len(test_sets)
    logger.info(f'Mean accuracy on {len(test_sets)} test sets is {accuracy:0.6f}')
    return accuracy


def continual_learning(
        model,
        mnist_datasets,
        lmbda,
        batch_size,
        epoch_num,
        validation_samples=0,
        importances_on_validation_samples=True
):
    """
    Continual model training on several datasets
    """
    model.reset()
    logger.info('Model has been cleaned.')
    test_datasets = []
    accuracies = []
    task_num = len(mnist_datasets)
    for idx, dataset in enumerate(mnist_datasets):
        test_datasets.append(dataset)
        model.open_lesson(lmbda)
        accuracy = train_model(model, dataset, test_datasets, batch_size, epoch_num, validation_samples)
        accuracies.append(accuracy)
        if idx != task_num - 1:
            if validation_samples > 0:
                if importances_on_validation_samples:
                    model.close_lesson(dataset[0].data[-validation_samples:], dataset[0].targets[-validation_samples:])
                else:
                    model.close_lesson(dataset[0].data[:-validation_samples], dataset[0].targets[:-validation_samples])
            else:
                model.close_lesson(dataset[0].data, dataset[0].targets)
            logger.info(f'Importances have been calculated.')
            #maxs, means = [], []
            #for imp in model.importances:
            #    maxs.append(imp.max().item())
            #    means.append(imp.mean().item())
            #logger.info(f"Max importance {max(maxs)}, max mean importance {max(means)}.")
    return accuracies


def permute_mnist(mnist):
    idxs = list(range(mnist[0].data.shape[1]))
    np.random.shuffle(idxs)
    mnist2 = []
    for dataset in mnist:
        perm_dataset = deepcopy(dataset)
        for i in range(perm_dataset.data.shape[1]):
            perm_dataset.data[:, i] = dataset.data[:, idxs[i]]
        mnist2.append(perm_dataset)
    return tuple(mnist2)


def experiments_run(config: dict):
    # network structure and training parameters
    experiment_name = config["experiment_id"]
    lambdas = config["lambdas"]
    net_struct = config["net_struct"]
    learning_rate = config["learning_rate"]
    N = config["repeats"]
    batch_size = config["batch_size"]
    epoch_num = config["epoch_num"]
    device = config.get("device") or "cpu"
    Model = locate(config["model_class"])
    validation_samples = config.get("validation_samples") or 0
    importances_on_validation_samples = config.get("importances_on_validation_samples", True)
    empirical_fisher = config.get("empirical_fisher", False)

    # setup logger to output to console and file
    log_format = config.get("log_format") or "%(asctime)s [%(levelname)s] %(message)s"
    log_file = config.get("log_file") or f"./{experiment_name}.log"
    log_level = config.get("log_level") or logging.INFO
    logging.basicConfig(filename=log_file, level=log_level, format=log_format)

    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    device = torch.device(device)
    logger.info(f"Operating device is {device}")

    dataset_file = 'datasets.dmp'
    try:
        mnist_datasets = joblib.load(dataset_file)
        logger.info('Dataset has been loaded from cache.')
    except FileNotFoundError:
        logger.info('Dataset cache not found. Creating new one.')
        # download and transform mnist dataset
        train_data = datasets.MNIST('../mnist_data', download=True, train=True)
        train_inputs = train_data.data.numpy()
        train_inputs = (train_inputs.reshape(train_inputs.shape[0], -1) / 255).astype(np.float32)
        train_labels = train_data.targets.numpy()
        train_dataset = MyMNIST(train_inputs, train_labels)

        test_data = datasets.MNIST('../mnist_data', download=True, train=False)
        test_inputs = test_data.data.numpy()
        test_inputs = (test_inputs.reshape(test_inputs.shape[0], -1) / 255).astype(np.float32)
        test_labels = test_data.targets.numpy()
        test_dataset = MyMNIST(test_inputs, test_labels)

        mnist = (train_dataset, test_dataset)
        mnist0 = mnist
        mnist1 = permute_mnist(mnist)
        mnist2 = permute_mnist(mnist)
        mnist3 = permute_mnist(mnist)
        mnist4 = permute_mnist(mnist)
        mnist5 = permute_mnist(mnist)
        mnist6 = permute_mnist(mnist)
        mnist7 = permute_mnist(mnist)
        mnist8 = permute_mnist(mnist)
        mnist9 = permute_mnist(mnist)

        mnist_datasets = [mnist0, mnist1, mnist2, mnist3, mnist4, mnist5, mnist6, mnist7, mnist8, mnist9]
        joblib.dump(mnist_datasets, dataset_file, compress=3)

    exp_result_file = f"{experiment_name}.dmp"
    try:
        experiments = joblib.load(exp_result_file)
    except FileNotFoundError:
        logger.info('Experiment cache not found. Creating new one.')
        experiments = defaultdict(list)

    model = Model(shape=net_struct, learning_rate=learning_rate, device=device, empirical_fisher=empirical_fisher)

    start_time = datetime.datetime.now()
    time_format = "%Y-%m-%d %H:%M:%S"
    logger.info(f'Continual learning start at {start_time:{time_format}}')

    lm = []
    for l in lambdas:
        if isinstance(l, list) or isinstance(l, tuple):
            if len(l) == 3:
                lm += list(np.arange(l[0], l[1], l[2]))
            else:
                raise ValueError(f'Wrong range of lambdas: {l}')
        else:
            lm.append(l)
    for lmbda in lambdas:
        exps = experiments[lmbda]
        len_exp = len(exps)
        K = max(0, N - len_exp)
        if K==0:
            accs = np.array(exps)[:, len(mnist_datasets)-1]
            accs_len = len(accs)
            res_str = f"{accs.mean():0.3f}:{accs.std() * student[accs_len-1] / np.sqrt(accs_len):0.3f}"
        else:
            res_str = ""
        logger.info(f'For lambda={lmbda} {len_exp} experiments done, {K} experiments queued. {res_str}')
        for i in range(len_exp+1, N+1):
            iter_start_time = datetime.datetime.now()
            logger.info(f'{i}-th experiment on lambda={lmbda} started at {iter_start_time:{time_format}}')
            accuracies = continual_learning(
                model, mnist_datasets,
                lmbda=lmbda,
                batch_size=batch_size,
                epoch_num=epoch_num,
                validation_samples=validation_samples,
                importances_on_validation_samples=importances_on_validation_samples,
            )
            exps.append(accuracies)
            joblib.dump(experiments, exp_result_file, compress=1)
            logger.info(f'{i}-th experiment time spent {datetime.datetime.now() - iter_start_time}')
            logger.info(f'For now total time spent {datetime.datetime.now() - start_time}')

    logger.info(f'Done for lambdas {lambdas}')
    lambdas = sorted(experiments.keys())
    experiments = {k: experiments[k] for k in lambdas}
    joblib.dump(experiments, exp_result_file, compress=1)
    logger.info(f'Sorted lambdas.')


if __name__ == "__main__":
    args = sys.argv[1:]
    config_file = args[0] if len(args) > 0 else "config.yml"
    with open(config_file, "r") as reader:
        config = yaml.safe_load(reader)
    experiments_run(config)
