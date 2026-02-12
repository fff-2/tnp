import argparse
import time
import os
import os.path as osp

def get_args():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--expid', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--wandb-project', type=str, default='contextual-bandits')
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)

    # wheel
    parser.add_argument("--cmab_data", choices=["wheel"], default="wheel")
    parser.add_argument("--cmab_wheel_delta", type=float, default=0.5)
    parser.add_argument("--cmab_mode", choices=["train", "eval", "plot", "evalplot"], default="train")
    parser.add_argument('--cmab_num_bs', type=int, default=10)
    parser.add_argument("--cmab_train_update_freq", type=int, default=1)
    parser.add_argument("--cmab_train_num_batches", type=int, default=1)
    parser.add_argument("--cmab_train_batch_size", type=int, default=8)
    parser.add_argument("--cmab_train_seed", type=int, default=0)
    parser.add_argument("--cmab_train_reward", type=str, default="all")
    parser.add_argument("--cmab_eval_method", type=str, default="ucb")
    parser.add_argument("--cmab_eval_num_contexts", type=int, default=2000)
    parser.add_argument("--cmab_eval_seed_start", type=int, default=0)
    parser.add_argument("--cmab_eval_seed_end", type=int, default=49)
    parser.add_argument("--cmab_plot_seed_start", type=int, default=0)
    parser.add_argument("--cmab_plot_seed_end", type=int, default=49)

    # Model
    parser.add_argument('--model', type=str, default="tnpa")

    # Training
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000)

    args = parser.parse_args()

    if args.expid is None:
        if args.cmab_mode == 'train':
            args.expid = time.strftime('%Y%m%d-%H%M')
        else:
            # For eval modes, default to the latest timestamped run
            from runner import results_path
            task_dir = osp.join(results_path, args.cmab_data, f'train-{args.cmab_train_reward}-R', args.model)
            if osp.isdir(task_dir):
                dirs = sorted(
                    [d for d in os.listdir(task_dir) if osp.isdir(osp.join(task_dir, d))],
                    reverse=True
                )
                if dirs:
                    # Find the first dir that has subdirs (actual expids are nested)
                    for d in dirs:
                        subdirs = sorted(
                            [s for s in os.listdir(osp.join(task_dir, d)) if osp.isdir(osp.join(task_dir, d, s))],
                            reverse=True
                        )
                        if subdirs:
                            args.expid = subdirs[0]
                            print(f'Using latest expid: {args.expid}')
                            break
            if args.expid is None:
                args.expid = time.strftime('%Y%m%d-%H%M')

    return args