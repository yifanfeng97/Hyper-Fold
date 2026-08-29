"""Classification datasets for Hyper-Fold: EC-number prediction and fold
classification.

Both datasets return lightweight ``ProteinData`` containers exposing the
fields the Fold-Conv models consume (``x`` residue-type ids, ``pos`` C-alpha
coordinates, ``seq`` sequence indices, ``ori`` orientation frames, ``batch``
graph ids, ``y`` targets), so the models run on a PyG-like API with no
``torch_geometric`` dependency. Orientation frames are computed at load time
in NumPy with the same math as ``hyperfold.model.foldconv.orientation``.

File formats follow the GearNet/CDConv benchmark releases; loading needs
only NumPy:

- EC (``ECDataset``): ``root/coordinates/{name}.npy`` C-alpha coordinates,
  ``root/{split}.fasta`` sequences, ``root/nrPDB-EC_annot.tsv`` multi-hot EC
  annotations, and ``root/nrPDB-EC_test.csv`` for the sequence-identity
  test-set masking (``percent``).
- Fold (``FoldDataset``, SCOPe 1.75 / Fold3D): ``root/class_map.txt``,
  ``root/{split}.txt`` (name <...> fold-string), and per-protein coordinates
  under ``root/{split}/{name}.hdf5`` (GearNet Zenodo release) or
  ``root/{split}/{name}.npz`` (NumPy-converted copy, see
  :func:`FoldDataset.convert_hdf5_to_npz`). The original ``.hdf5`` layout is
  read through a lazy ``h5py`` import; ``h5py`` is an optional dependency and
  is never needed when the ``.npz`` layout is used.

Structure-forcing augmentation (fold training): Gaussian coordinate noise
(``noise_std``, paper: sigma = 0.15 Angstrom) and residue-type masking to the
unknown token X (``mask_prob``, paper: p = 0.15), applied on the train split
only. The EC dataset applies Gaussian coordinate noise with sigma = 0.05
Angstrom on the train split, matching the reference pipeline.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "ProteinData",
    "collate_fn",
    "orientation",
    "ECDataset",
    "FoldDataset",
]


_aa = "ACDEFGHIKLMNPQRSTVWYX"
_aa_to_id = {aa: i for i, aa in enumerate(_aa)}

#: Residue-type id of the unknown token X (used by residue-type masking).
UNKNOWN_AA_ID = 20


class ProteinData:
    """Lightweight data container mimicking PyG Data for our use case."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to(self, device):
        for key, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
            elif hasattr(value, 'to') and callable(value.to):
                setattr(self, key, value.to(device))
        return self

    def __repr__(self):
        attrs = ", ".join(
            f"{k}={tuple(v.shape) if isinstance(v, torch.Tensor) else v}"
            for k, v in vars(self).items()
        )
        return f"{self.__class__.__name__}({attrs})"


def collate_fn(batch):
    """Collate a list of ProteinData into a batched ProteinData."""
    xs, poss, seqs, oris, ys, batch_vec = [], [], [], [], [], []
    for i, d in enumerate(batch):
        n = d.x.shape[0]
        xs.append(d.x)
        poss.append(d.pos)
        seqs.append(d.seq)
        oris.append(d.ori)
        ys.append(d.y)
        batch_vec.append(torch.full((n,), i, dtype=torch.long))

    return ProteinData(
        x=torch.cat(xs, dim=0),
        pos=torch.cat(poss, dim=0),
        seq=torch.cat(seqs, dim=0),
        ori=torch.cat(oris, dim=0),
        y=torch.stack(ys, dim=0),
        batch=torch.cat(batch_vec, dim=0),
    )


def _normalize_l2(x, axis=1, eps=1e-12):
    """NumPy L2 normalization matching sklearn.preprocessing.normalize."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    out = np.zeros_like(x)
    np.divide(x, norm, out=out, where=norm > 0)
    return out


def orientation(pos):
    """Compute local orientation frames from CA coordinates.

    Same math as ``hyperfold.model.foldconv.orientation`` (the torch version
    used inside the models), evaluated in NumPy at dataset load time.
    """
    u = _normalize_l2(pos[1:, :] - pos[:-1, :], axis=1)
    u1 = u[1:, :]
    u2 = u[:-1, :]
    b = _normalize_l2(u2 - u1, axis=1)
    n = _normalize_l2(np.cross(u2, u1), axis=1)
    o = _normalize_l2(np.cross(b, n), axis=1)
    ori = np.stack([b, n, o], axis=1)
    return np.concatenate([
        np.expand_dims(ori[0], 0),
        ori,
        np.expand_dims(ori[-1], 0)
    ], axis=0)


class ECDataset(Dataset):
    """Enzyme Commission (EC-number) multi-label classification dataset.

    Produces per-residue residue-type ids (21 classes incl. X), C-alpha
    positions (centered), sequence indices and orientation frames, plus a
    multi-hot target vector over the EC classes. ``weights`` holds the
    per-class BCE weights (n_samples / class frequency) used by the training
    script.
    """

    def __init__(self, root='data/EnzymeCommission_cdconv', percent=30,
                 random_seed=0, split='train', protein_subset=None,
                 coord_noise_std=0.05):
        self.random_state = np.random.RandomState(random_seed)
        self.split = split
        self.protein_subset = set(protein_subset) if protein_subset is not None else None
        self.coord_noise_std = float(coord_noise_std)

        valid_percents = {30, 40, 50, 70, 95, 100}
        if percent not in valid_percents:
            raise ValueError(
                f"Unsupported percent={percent}. Choose from {sorted(valid_percents)}."
            )

        npy_dir = os.path.join(root, 'coordinates')
        fasta_file = os.path.join(root, split + '.fasta')

        # Mask test set according to sequence identity cutoff.
        test_set = set()
        if split == "test":
            with open(os.path.join(root, "nrPDB-EC_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    elif percent == 100:
                        test_set.add(arr[0])

        # Load fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    if split == "train" and self.protein_subset is not None and (protein_name not in self.protein_subset):
                        continue
                    amino_ids = [_aa_to_id[amino] for amino in line.rstrip()]
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        self.data = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name + ".npy"))
            center = np.sum(pos, axis=0, keepdims=True) / pos.shape[0]
            pos = pos - center
            ori = orientation(pos)
            self.data.append((protein_name, pos, ori, amino_ids.astype(int)))

        # Load EC annotations.
        level_idx = 1
        ec_cnt = 0
        ec_num = {}
        ec_annotations = {}
        self.labels = {}
        with open(os.path.join(root, 'nrPDB-EC_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1:
                    arr = line.rstrip().split('\t')
                    for ec in arr:
                        ec_annotations[ec] = ec_cnt
                        ec_num[ec] = 0
                        ec_cnt += 1
                elif idx > 2:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        for ec in arr[level_idx].split(','):
                            if len(ec) > 0:
                                protein_labels.append(ec_annotations[ec])
                                ec_num[ec] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(ec_annotations)
        n_samples = len(self.data)
        self.weights = np.zeros((ec_cnt,), dtype=np.float32)
        for ec, idx in ec_annotations.items():
            self.weights[idx] = n_samples / max(ec_num[ec], 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        protein_name, pos, ori, amino = self.data[idx]
        label = np.zeros((self.num_classes,), dtype=np.float32)
        if len(self.labels[protein_name]) > 0:
            label[self.labels[protein_name]] = 1.0

        if self.split == "train":
            pos = pos + self.random_state.normal(0.0, self.coord_noise_std,
                                                 pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        return ProteinData(
            x=torch.from_numpy(amino),
            y=torch.from_numpy(label),
            ori=torch.from_numpy(ori),
            seq=torch.from_numpy(seq),
            pos=torch.from_numpy(pos),
        )


class FoldDataset(Dataset):
    """SCOPe 1.75 fold classification dataset (Fold3D).

    Single-label classification over 1,195 fold classes with three test
    scenarios (test_fold / test_superfamily / test_family) sharing the same
    train/valid splits. Produces the same ``ProteinData`` fields as
    :class:`ECDataset`; ``y`` is a scalar class index.

    Structure-forcing augmentation (train split only): Gaussian coordinate
    noise with std ``noise_std`` (paper: 0.15 Angstrom) and residue-type
    masking to the unknown token X with probability ``mask_prob`` (paper:
    0.15).

    Per-protein coordinates are read from ``{split}/{name}.hdf5`` (original
    GearNet release; requires the optional ``h5py`` package) or, preferably,
    from a NumPy-converted ``{split}/{name}.npz`` copy with arrays
    ``amino_pos`` [1, L, 3] and ``amino_types`` [L]. Use
    :meth:`convert_hdf5_to_npz` once to create the ``.npz`` layout.
    """

    SPLITS = ("train", "valid", "test_fold", "test_superfamily", "test_family")

    def __init__(self, root='data/fold3d/new_fold3d', random_seed=0,
                 split='train', noise_std=0.05, mask_prob=0.0):
        if split not in self.SPLITS:
            raise ValueError(
                f"Unknown split '{split}'. Choose from {self.SPLITS}."
            )
        self.random_state = np.random.RandomState(random_seed)
        self.split = split
        self.noise_std = float(noise_std)
        self.mask_prob = float(mask_prob)

        with open(os.path.join(root, 'class_map.txt'), 'r') as f:
            class_map = {}
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                fold_str, idx = line.split('\t')
                class_map[fold_str] = int(idx)
        self.num_classes = len(class_map)

        self.data = []
        with open(os.path.join(root, split + '.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip('\n').split('\t')
                if len(arr) < 2:
                    continue
                name, fold_str = arr[0], arr[-1]
                label = class_map[fold_str]

                pos, amino = self._load_arrays(root, split, name)

                # A few samples carry residue ids outside [0, 19] (-1 or up to
                # 24); map them to the unknown-residue slot 20 (X), matching
                # the EC pipeline's 21-entry embedding convention.
                amino = np.where((amino < 0) | (amino > 19), UNKNOWN_AA_ID,
                                 amino)

                center = np.sum(pos, axis=0, keepdims=True) / pos.shape[0]
                pos = pos - center
                ori = orientation(pos)
                self.data.append((name, pos, ori, amino, label))

    @staticmethod
    def _load_arrays(root, split, name):
        """Load (pos [L, 3], amino [L]) from the .npz or .hdf5 layout."""
        npz_path = os.path.join(root, split, name + '.npz')
        if os.path.exists(npz_path):
            with np.load(npz_path) as z:
                pos = np.asarray(z['amino_pos'][0], dtype=np.float64)
                amino = np.asarray(z['amino_types'], dtype=np.int64)
            return pos, amino

        hdf5_path = os.path.join(root, split, name + '.hdf5')
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                f"Reading '{hdf5_path}' requires the optional 'h5py' package. "
                "Install it once and pre-convert the dataset with "
                "FoldDataset.convert_hdf5_to_npz(root) so that loading needs "
                "only NumPy."
            ) from e
        with h5py.File(hdf5_path, 'r') as h5:
            pos = np.asarray(h5['amino_pos'][0], dtype=np.float64)
            amino = np.asarray(h5['amino_types'][()], dtype=np.int64)
        return pos, amino

    @staticmethod
    def convert_hdf5_to_npz(root, splits=SPLITS):
        """Convert the GearNet .hdf5 layout to NumPy .npz files (in place).

        Requires the optional ``h5py`` package at conversion time only; the
        resulting ``{split}/{name}.npz`` files load with NumPy alone.
        """
        import h5py

        n = 0
        for split in splits:
            split_dir = os.path.join(root, split)
            if not os.path.isdir(split_dir):
                continue
            for fname in os.listdir(split_dir):
                if not fname.endswith('.hdf5'):
                    continue
                hdf5_path = os.path.join(split_dir, fname)
                npz_path = os.path.join(
                    split_dir, fname[:-len('.hdf5')] + '.npz')
                if os.path.exists(npz_path):
                    continue
                with h5py.File(hdf5_path, 'r') as h5:
                    amino_pos = np.asarray(h5['amino_pos'])
                    amino_types = np.asarray(h5['amino_types'])
                np.savez_compressed(npz_path, amino_pos=amino_pos,
                                    amino_types=amino_types)
                n += 1
        return n

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, pos, ori, amino, label = self.data[idx]

        if self.split == "train":
            # Structure-forcing augmentation: Gaussian coordinate noise plus
            # residue-type masking to the unknown token X.
            pos = pos + self.random_state.normal(0.0, self.noise_std,
                                                 pos.shape)
            if self.mask_prob > 0.0:
                mask = self.random_state.rand(amino.shape[0]) < self.mask_prob
                amino = np.where(mask, UNKNOWN_AA_ID, amino)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        return ProteinData(
            x=torch.from_numpy(amino),
            y=torch.tensor(label, dtype=torch.long),
            ori=torch.from_numpy(ori),
            seq=torch.from_numpy(seq),
            pos=torch.from_numpy(pos),
        )
