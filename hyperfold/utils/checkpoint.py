"""Checkpoint writing with automatic pruning of non-interval .pth files."""
import os

from ..data.utils import write_pkl


def write_checkpoint(
        ckpt_path: str,
        model,
        conf,
        optimizer,
        epoch,
        step,
        logger=None,
        use_torch=True,
        save_interval=50000,
        metrics=None,
    ):
    """Serialize experiment state and stats to a pickle file.

    Non-interval ``.pth`` files in the checkpoint directory are pruned, so
    only the checkpoints at each ``save_interval`` step (and "best" files)
    are retained.

    Args:
        ckpt_path: Path to save checkpoint.
        model: Model state dict.
        conf: Experiment configuration.
        optimizer: Optimizer state dict.
        epoch: Training epoch at time of checkpoint.
        step: Training steps at time of checkpoint.
        logger: Optional logger for the serialization message.
        use_torch: Serialize with torch.save instead of pickle.
        save_interval: Interval of steps to retain checkpoints.
        metrics: Optional metrics to store in the checkpoint.
    """
    ckpt_dir = os.path.dirname(ckpt_path)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Delete non-interval .pth files (retain only the ones at each save_interval step)
    for fname in os.listdir(ckpt_dir):
        if "best" in fname:
            continue
        if fname.endswith('.pth'):
            step_in_fname = ''.join(filter(str.isdigit, fname))  # Extract step number from filename
            if step_in_fname and int(step_in_fname) % save_interval != 0:
                os.remove(os.path.join(ckpt_dir, fname))

    if logger is not None:
        logger.info(f'Serializing experiment state to {ckpt_path}')
    else:
        print(f'Serializing experiment state to {ckpt_path}')
    write_pkl(
        ckpt_path,
        {
            'model': model,
            'conf': conf,
            'optimizer': optimizer,
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
        },
        use_torch=use_torch)
