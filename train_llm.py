import numpy as np
from scipy.stats import spearmanr
import torch

from utils import AverageMeter


def generate_modality_mask():
    """Generate a mask combination that keeps at least one modality."""
    combinations = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ]
    idx = torch.randint(0, len(combinations), [1])
    return combinations[idx[0]]


def train_epoch(epoch, model, loss_fn, train_loader, optim, logger, device, args):
    model.train()
    preds = np.array([])
    labels = np.array([])
    losses = AverageMeter('loss', logger)

    for video_feat, audio_feat, flow_feat, label in train_loader:
        video_feat = video_feat.to(device)
        audio_feat = audio_feat.to(device)
        flow_feat = flow_feat.to(device)
        label = label.float().to(device)

        # Randomly mask modalities during training to improve robustness.
        mask = generate_modality_mask()

        out = model(video_feat, audio_feat, flow_feat, mask)
        pred = out['output']
        loss = loss_fn(pred, label, out['output_main'], out['output_aux'], out['embed'], args)

        optim.zero_grad()
        loss.backward()
        # Clip gradients for more stable optimization.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        losses.update(loss.item(), label.shape[0])

        if len(preds) == 0:
            preds = pred.detach().cpu().numpy()
            labels = label.detach().cpu().numpy()
        else:
            preds = np.concatenate((preds, pred.detach().cpu().numpy()), axis=0)
            labels = np.concatenate((labels, label.detach().cpu().numpy()), axis=0)

    coef, _ = spearmanr(preds, labels)
    if logger is not None:
        logger.add_scalar('train coef', coef, epoch)
    avg_loss = losses.done(epoch)
    return avg_loss, coef
