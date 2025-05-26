"""Calculate the mean and std of results from multiple log files."""

import argparse

import numpy as np


def _calculate_mean_and_std(long_dict):
    for name in long_dict:
        for label in long_dict[name]:
            long_dict[name][label] = (
                round(np.mean(long_dict[name][label]), 3),
                round(np.std(long_dict[name][label]), 3)
            )
    return long_dict


def _print_results(long_dict, measure):
    print(measure)
    for item in long_dict.items():
        print(item)

def main():
    parser = argparse.ArgumentParser(description='General results path.')
    parser.add_argument(
        '--results_path',
        type=str,
        help='Path to the results.',
        required=True,
    )
    parser.add_argument(
        '--n_array',
        type=str,
        help='Number of arrays.',
        default=10
    )
    args = parser.parse_args()

    mos_labels = ['mos', 'noi', 'col', 'dis', 'loud']
    datasets = [
        'NISQA_VAL_SIM',
        'NISQA_VAL_LIVE',
        'NISQA_TEST_LIVETALK',
        'NISQA_TEST_FOR',
        'NISQA_TEST_P501'
    ]
    results_pcc = {dataset: {
        mos_label: [] for mos_label in mos_labels
    } for dataset in datasets}
    results_mse = {dataset: {
        mos_label: [] for mos_label in mos_labels
    } for dataset in datasets}

    for i in range(args.n_array):
        new_path = f'{args.results_path}_{i}/train.log'
        with open(new_path, 'r') as f:
            for _, line in enumerate(f):
                if '[31]' in line:
                    dataset_name = line.split('[')[1][:-1]
                    mos_name = line.split('[')[2][:-1]
                    if dataset_name not in datasets:
                        continue
                    results_pcc[dataset_name][mos_name].append(
                        float(line.split('LCC = ')[-1][:5])
                    )
                    results_mse[dataset_name][mos_name].append(
                        np.sqrt(float(line.split('MSE = ')[-1][:5]))
                    )
    _print_results(_calculate_mean_and_std(results_pcc), 'pcc')
    _print_results(_calculate_mean_and_std(results_mse), 'rmse')


if __name__ == '__main__':
    main()