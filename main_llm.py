import argparse
import math
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from datasets import FS1000Dataset, RGDataset, FisVDataset
from models.llm_aqa_model import LLM_AQA
from models.loss import LossFun
from test_llm import test_epoch
from train_llm import train_epoch


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def get_optim(model, args):
    if args.optim == 'sgd':
        optim = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optim = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optim}")
    return optim


def get_scheduler(optim, args):
    if args.lr_decay is None:
        return None

    if args.lr_decay == 'cos':
        schedulers = []
        milestones = []

        # Warm up linearly from 0 to args.lr for the first few epochs.
        if args.warmup > 0:
            def warmup_lambda(epoch):
                return float(epoch + 1) / float(args.warmup)

            schedulers.append(torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lambda))
            milestones.append(args.warmup)

        # Run cosine annealing for the remaining epochs.
        cos_epochs = args.epoch - args.warmup
        if cos_epochs > 0:
            schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim,
                    T_max=cos_epochs,
                    eta_min=args.lr * args.decay_rate,
                )
            )
            if args.warmup > 0:
                milestones.append(args.warmup + cos_epochs)

        if len(schedulers) == 1:
            return schedulers[0]

        return torch.optim.lr_scheduler.SequentialLR(
            optim,
            schedulers=schedulers,
            milestones=milestones[:-1],
        )

    if args.lr_decay == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optim,
            milestones=[args.epoch - 30],
            gamma=args.decay_rate,
        )

    raise ValueError(f"Unknown scheduler: {args.lr_decay}")


def compute_average(metric_list):
    """Average metrics using Fisher Z for correlations and arithmetic mean for MSE."""
    r_values = [x[0] for x in metric_list]
    z_list = [0.5 * (math.log(1 + r) - math.log(1 - r)) for r in r_values]
    zz = np.mean(z_list)
    coef_avg = (np.exp(zz) - np.exp(-zz)) / (np.exp(zz) + np.exp(-zz))
    mse_avg = np.mean([x[1] for x in metric_list])
    return coef_avg, mse_avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, default='data/fs1000/video_features')
    parser.add_argument('--audio-path', type=str, default='data/fs1000/audio_features')
    parser.add_argument('--flow-path', type=str, default='data/fs1000/flow_features')
    parser.add_argument('--dataset', type = str, choices=['FS1000', 'FisV', 'RG'], default='FS1000')
    parser.add_argument('--clip-num', type=int, default=95)
    parser.add_argument('--train-label-path', type=str, default='data/fs1000/train.txt')
    parser.add_argument('--test-label-path', type=str, default='data/fs1000/val.txt')
    parser.add_argument('--action-type', type=str, default='TES')
    parser.add_argument('--model-name', type=str, default='llm_aqa', help='Name used to save model and logs')
    parser.add_argument('--ckpt', default=None, help="Checkpoint for pretrained model")
    parser.add_argument('--test', action='store_true', help="Only evaluate without training")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--lr-decay', type=str, default='cos', help='Learning rate scheduler')
    parser.add_argument('--decay-rate', type=float, default=0.1, help='Learning rate decay factor')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--in_dim', type=int, default=768)
    parser.add_argument('--num_fusion_tokens', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--alpha_mse', type=float, default=10)
    parser.add_argument('--alpha_con', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--score_range', type=int, default=None)
    parser.add_argument('--llm-path', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--use-lora', action='store_true', default=True)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default='outputs')

    args = parser.parse_args()
    setup_seed(args.seed)

    # Silence tokenizer parallelism warnings for cleaner logs.
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    log_dir = os.path.join(args.output_dir, 'logs', args.model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load datasets.
    Dataset = FS1000Dataset
    if args.dataset == 'RG':
        Dataset = RGDataset
    elif args.dataset == 'FisV':
        Dataset = FisVDataset
    elif args.dataset == 'FS1000':
        Dataset = FS1000Dataset
        
    train_data = Dataset(
        args.video_path,
        args.audio_path,
        args.flow_path,
        args.train_label_path,
        clip_num=args.clip_num,
        action_type=args.action_type,
        args=args,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
    )

    test_data = Dataset(
        args.video_path,
        args.audio_path,
        args.flow_path,
        args.test_label_path,
        clip_num=args.clip_num,
        action_type=args.action_type,
        train=False,
        args=args,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print('=============Load dataset successfully=============')

    # Build the model.
    model = LLM_AQA(
        in_dim=args.in_dim,
        clip_num=args.clip_num,
        dropout=args.dropout,
        llm_path=args.llm_path,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        num_fusion_tokens=args.num_fusion_tokens,
    ).to(device)

    loss_fn = LossFun(args.alpha_mse, args.alpha, args.margin, args.alpha_con)

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint)

    print('=============Load model successfully=============')

    # Evaluation-only mode.
    if args.test:
        combinations = [
            [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ]
        results_for_avg = []
        modalities = ['V', 'A', 'F']

        for mask in combinations:
            combination_name = ''.join([modalities[j] for j in range(3) if mask[j] == 1])
            test_loss, coef = test_epoch(0, model, test_loader, None, device, mask, args)
            print(f"Combination {combination_name}: Test Loss: {test_loss:.4f}\tTest Coef: {coef:.3f}")
            if mask != [1, 1, 1]:
                results_for_avg.append((coef, test_loss))

        avg_coef, avg_loss = compute_average(results_for_avg)
        print("Average metrics (excluding VAF):")
        print(f"Average Test Loss: {avg_loss:.4f}\tAverage Test Coef: {avg_coef:.3f}")
        raise SystemExit(0)

    logger = SummaryWriter(log_dir)
    best_coef, best_epoch, best_coef_mse = -1, -1, 100000.0
    best_mse, best_epoch2, best_mse_coef = 100000.0, -1, -1

    optim = get_optim(model, args)
    scheduler = get_scheduler(optim, args)

    print('=============Begin training=============')

    for epc in range(args.epoch):
        avg_loss, train_coef = train_epoch(epc, model, loss_fn, train_loader, optim, logger, device, args)
        if scheduler is not None:
            scheduler.step()

        test_loss, test_coef = test_epoch(epc, model, test_loader, logger, device, [1, 1, 1], args)

        if round(test_coef, 3) > round(best_coef, 3) or (
            round(test_coef, 3) == round(best_coef, 3) and test_loss < best_coef_mse
        ):
            best_coef, best_epoch, best_coef_mse = test_coef, epc, test_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, args.model_name + '_best.pkl'))

        if round(test_loss, 3) < round(best_mse, 3) or (
            round(test_loss, 3) == round(best_mse, 3) and test_coef > best_mse_coef
        ):
            best_mse, best_epoch2, best_mse_coef = test_loss, epc, test_coef
            torch.save(model.state_dict(), os.path.join(ckpt_dir, args.model_name + '_best_mse.pkl'))

        print(
            f'Epoch: {epc}\tLoss: {avg_loss:.4f}\tTrain Coef: {train_coef:.3f}\t'
            f'Test Loss: {test_loss:.4f}\tTest Coef: {test_coef:.3f}'
        )

    print(
        'Best Test Coef: {:.3f} (MSE: {:.3f})\tBest Coef Epoch: {}\t'
        'Best Test MSE: {:.3f} (Coef: {:.3f})\tBest MSE Epoch: {}'.format(
            best_coef, best_coef_mse, best_epoch, best_mse, best_mse_coef, best_epoch2
        )
    )

    print("=============Loading best model based on Coef=============")
    checkpoint = torch.load(os.path.join(ckpt_dir, args.model_name + '_best.pkl'), map_location=device)
    model.load_state_dict(checkpoint)
    combinations = [
        [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ]
    results_for_avg = []
    modalities = ['V', 'A', 'F']

    for mask in combinations:
        combination_name = ''.join([modalities[j] for j in range(3) if mask[j] == 1])
        test_loss, coef = test_epoch(0, model, test_loader, None, device, mask, args)
        print(f"Combination {combination_name}: Test Loss: {test_loss:.4f}\tTest Coef: {coef:.3f}")
        if mask != [1, 1, 1]:
            results_for_avg.append((coef, test_loss))

    avg_coef, avg_loss = compute_average(results_for_avg)
    print("Average metrics (excluding VAF):")
    print(f"Average Test Loss: {avg_loss:.4f}\tAverage Test Coef: {avg_coef:.3f}")

    print("=============Loading best model based on MSE=============")
    checkpoint = torch.load(os.path.join(ckpt_dir, args.model_name + '_best_mse.pkl'), map_location=device)
    model.load_state_dict(checkpoint)
    results_for_avg = []

    for mask in combinations:
        combination_name = ''.join([modalities[j] for j in range(3) if mask[j] == 1])
        test_loss, coef = test_epoch(0, model, test_loader, None, device, mask, args)
        print(f"Combination {combination_name}: Test Loss: {test_loss:.4f}\tTest Coef: {coef:.3f}")
        if mask != [1, 1, 1]:
            results_for_avg.append((coef, test_loss))

    avg_coef, avg_loss = compute_average(results_for_avg)
    print("Average metrics (excluding VAF):")
    print(f"Average Test Loss: {avg_loss:.4f}\tAverage Test Coef: {avg_coef:.3f}")
