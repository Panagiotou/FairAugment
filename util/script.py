import subprocess
import argparse

def main(dataset):
    sampling_methods = ['class', 'class_protected', 'protected', 'same_class']
    
    for method in sampling_methods:
        command = ['python', 'ml_efficiency_class_balance.py', '--dataset', dataset, '--sampling_method', method]
        subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset name', required=True)
    args = parser.parse_args()

    main(args.dataset)