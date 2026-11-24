#!/usr/bin/env python
"""
Pool an akita_data.py dataset across its train/valid/test splits into a single
ordered collection of windows, ready to be fed to a prediction script.

This module does NOT load a model and does NOT compute metrics. It only:
  - validates the dataset (geometry, target count, split strides)
  - builds one window table (chrom, start, end, split, global row index)
  - pools the targets into a single array in that same row order
  - exposes a pooled tf.data iterator over sequences in that same row order

Pooling across splits is only valid when the dataset was built with
stride_train == stride_test == seq_length, so that every window is
non-overlapping and the split labels carry no meaning. The module checks this
and refuses to pool otherwise.

Typical use as a CLI:

    python pool_data.py --data_dir .../neuron/all --out_prefix .../pooled/neuron

Typical use as a library (from a prediction script):

    from pool_data import load_pooled
    pooled = load_pooled(data_dir)
    pooled.windows          # DataFrame, one row per window, in prediction order
    pooled.targets          # (n_windows, target_length, num_targets) float16
    for x in pooled.iter_sequences(batch_size=4):
        ...                 # batches in the same order as pooled.windows
"""

import os
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from basenji import dataset


SPLITS = ['train', 'valid', 'test']


# ----------------------------------------------------------------------
# container
# ----------------------------------------------------------------------

@dataclass
class PooledData:
    """Windows, targets and metadata pooled across splits, in one row order."""
    data_dir: str
    windows: pd.DataFrame                 # chrom,start,end,split,row
    targets: Optional[np.ndarray]         # (n_windows, target_length, num_targets)
    target_names: List[str]
    stats: dict
    options: dict = field(default_factory=dict)
    _datasets: list = field(default_factory=list, repr=False)

    @property
    def n_windows(self):
        return len(self.windows)

    @property
    def target_length(self):
        return self.stats['target_length']

    @property
    def num_targets(self):
        return self.stats['num_targets']

    def iter_sequences(self, batch_size=4):
        """Yield batches of one-hot sequences in the same order as .windows.

        Streams from the TFRecords rather than materialising the full array,
        which would be ~4 MB per window.
        """
        for ds in self._datasets:
            for batch in ds.dataset:
                # SeqDataset in eval mode yields (sequence, target) tuples
                yield batch[0] if isinstance(batch, (tuple, list)) else batch

    def select(self, chroms=None, exclude_chroms=None, splits=None):
        """Boolean mask over windows, for use by a downstream script."""
        m = np.ones(self.n_windows, dtype=bool)
        if chroms:
            m &= self.windows.chrom.isin(chroms).values
        if exclude_chroms:
            m &= ~self.windows.chrom.isin(exclude_chroms).values
        if splits:
            m &= self.windows.split.isin(splits).values
        return m


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------

def _read_options(data_dir):
    path = f'{data_dir}/options.json'
    return json.load(open(path)) if os.path.exists(path) else {}


def _read_target_names(data_dir, num_targets):
    path = f'{data_dir}/targets.txt'
    if os.path.exists(path):
        tdf = pd.read_csv(path, sep='\t', index_col=0)
        for col in ('identifier', 'description'):
            if col in tdf.columns:
                return tdf[col].astype(str).tolist()
    return [f'target{i}' for i in range(num_targets)]


def validate(data_dir, require_equal_strides=True, verbose=True):
    """Check dataset geometry and stride settings. Returns (stats, options)."""
    stats = json.load(open(f'{data_dir}/statistics.json'))
    options = _read_options(data_dir)

    seq_length = stats['seq_length']
    target_length = stats['target_length']
    num_targets = stats['num_targets']
    pool_width = stats.get('pool_width')
    crop_bp = stats.get('crop_bp', options.get('crop_bp'))
    diagonal_offset = stats.get('diagonal_offset',
                                options.get('diagonal_offset', 2))

    problems = []

    # target_length must be consistent with an upper-triangular flattening
    n_bins = None
    for cand in range(diagonal_offset + 1, 4096):
        if len(np.triu_indices(cand, diagonal_offset)[0]) == target_length:
            n_bins = cand
            break
    if n_bins is None:
        problems.append(
            f'target_length={target_length} is not a valid upper-triangular '
            f'length for any bin count at diagonal_offset={diagonal_offset}')
    elif pool_width and crop_bp is not None:
        expected = (seq_length - 2 * crop_bp) // pool_width
        if expected != n_bins:
            problems.append(
                f'inferred n_bins={n_bins} but seq_length/crop/pool_width '
                f'imply {expected}')

    # strides: pooling across splits requires non-overlapping windows everywhere
    st_train = options.get('stride_train')
    st_test = options.get('stride_test')
    if st_train is not None and st_test is not None:
        # akita_data.py stores strides as a fraction of seq_length in some
        # versions and as bp in others; normalise to bp
        def as_bp(v):
            return int(round(v * seq_length)) if v <= 1 else int(v)
        st_train_bp, st_test_bp = as_bp(st_train), as_bp(st_test)
        if verbose:
            print(f'strides: train={st_train_bp} test={st_test_bp} '
                  f'seq_length={seq_length}')
        if require_equal_strides and not (st_train_bp == st_test_bp == seq_length):
            problems.append(
                f'stride_train={st_train_bp}, stride_test={st_test_bp}, but '
                f'pooling across splits requires both to equal seq_length='
                f'{seq_length}. Windows in the train split overlap, so pooled '
                f'windows would not be independent.')
    elif verbose:
        print('WARNING: strides not found in options.json — verify manually '
              'that this dataset was built with non-overlapping windows')

    if verbose:
        print(f'seq_length={seq_length} target_length={target_length} '
              f'n_bins={n_bins} num_targets={num_targets} '
              f'pool_width={pool_width} crop_bp={crop_bp} '
              f'diagonal_offset={diagonal_offset}')

    if problems:
        raise ValueError('dataset validation failed:\n  - ' +
                         '\n  - '.join(problems))
    return stats, options


# ----------------------------------------------------------------------
# pooling
# ----------------------------------------------------------------------

def load_pooled(data_dir, load_targets=True, batch_size=4,
                require_equal_strides=True, verbose=True):
    """Pool train/valid/test into one ordered PooledData object.

    Row order is: all train windows (in sequences.bed order), then all valid,
    then all test. iter_sequences() and .targets follow the same order as
    .windows, so a downstream prediction script can zip them safely.
    """
    stats, options = validate(data_dir, require_equal_strides, verbose)
    target_names = _read_target_names(data_dir, stats['num_targets'])

    bed = pd.read_csv(f'{data_dir}/sequences.bed', sep='\t', header=None,
                      names=['chrom', 'start', 'end', 'split'])

    win_parts, tgt_parts, datasets = [], [], []

    for split in SPLITS:
        sub = bed[bed.split == split].reset_index(drop=True)
        if len(sub) == 0:
            if verbose:
                print(f'{split:6s}: empty, skipping')
            continue

        ds = dataset.SeqDataset(data_dir, split_label=split,
                                batch_size=batch_size, mode='eval')

        n_ds = getattr(ds, 'num_seqs', None)
        if n_ds is not None and n_ds != len(sub):
            raise AssertionError(
                f'{split}: sequences.bed has {len(sub)} rows but the TFRecords '
                f'contain {n_ds}. Row order/count mismatch — per-chromosome '
                f'stratification downstream would be silently wrong.')

        if load_targets:
            t = ds.numpy(return_inputs=False)
            if t.shape[0] != len(sub):
                raise AssertionError(
                    f'{split}: sequences.bed has {len(sub)} rows but targets '
                    f'array has {t.shape[0]}.')
            tgt_parts.append(t.astype(np.float16))

        win_parts.append(sub)
        datasets.append(ds)
        if verbose:
            print(f'{split:6s}: {len(sub):5d} windows')

    if not win_parts:
        raise ValueError(f'no windows found in {data_dir}')

    windows = pd.concat(win_parts, ignore_index=True)
    windows['row'] = np.arange(len(windows))

    targets = np.concatenate(tgt_parts, axis=0) if tgt_parts else None

    if verbose:
        print(f'pooled : {len(windows):5d} windows across '
              f'{windows.chrom.nunique()} chromosomes')
        if targets is not None:
            print(f'targets: {targets.shape} {targets.dtype}')

    return PooledData(data_dir=data_dir, windows=windows, targets=targets,
                      target_names=target_names, stats=stats, options=options,
                      _datasets=datasets)


# ----------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------

def report(pooled):
    """Print per-chromosome window counts and per-channel target diagnostics."""
    w = pooled.windows
    print('\n=== windows per chromosome ===')
    tab = (w.groupby('chrom')
             .agg(n_windows=('row', 'size'),
                  splits=('split', lambda s: ','.join(sorted(set(s)))))
             .reset_index())
    tab['chrom_key'] = tab.chrom.str.replace('chr', '', regex=False)
    tab = tab.sort_values('chrom_key', key=lambda s: s.map(
        lambda v: (0, int(v)) if v.isdigit() else (1, v)))
    print(tab[['chrom', 'n_windows', 'splits']].to_string(index=False))

    # every chromosome should sit in exactly one split
    multi = tab[tab.splits.str.contains(',')]
    if len(multi):
        print('\nWARNING: these chromosomes span multiple splits:')
        print(multi[['chrom', 'splits']].to_string(index=False))

    if pooled.targets is None:
        return

    print('\n=== target channels ===')
    for i, name in enumerate(pooled.target_names):
        col = pooled.targets[:, :, i].astype(np.float32)
        finite = np.isfinite(col)
        print(f'{name:14s} mean={np.nanmean(col):+.4f} sd={np.nanstd(col):.4f} '
              f'min={np.nanmin(col):+.3f} max={np.nanmax(col):+.3f} '
              f'nonfinite={1 - finite.mean():.2%}')
        if not np.isfinite(np.nanstd(col)) or np.nanstd(col) == 0:
            print(f'  WARNING: {name} has zero/undefined variance — the cool '
                  f'file may have failed to load')


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True,
                    help='akita_data.py output directory')
    ap.add_argument('--out_prefix', required=True,
                    help='prefix for pooled outputs')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--no_targets', action='store_true',
                    help='only build the window table, skip target loading')
    ap.add_argument('--allow_unequal_strides', action='store_true',
                    help='pool even if windows overlap (not recommended)')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    pooled = load_pooled(args.data_dir,
                         load_targets=not args.no_targets,
                         batch_size=args.batch_size,
                         require_equal_strides=not args.allow_unequal_strides)
    report(pooled)

    pooled.windows.to_csv(f'{args.out_prefix}_windows.tsv',
                          sep='\t', index=False)
    with open(f'{args.out_prefix}_meta.json', 'w') as fh:
        json.dump({'data_dir': os.path.abspath(args.data_dir),
                   'n_windows': pooled.n_windows,
                   'target_names': pooled.target_names,
                   'stats': pooled.stats}, fh, indent=2)

    if pooled.targets is not None:
        np.savez_compressed(f'{args.out_prefix}_targets.npz',
                            targets=pooled.targets)
        print(f'\nwrote {args.out_prefix}_targets.npz '
              f'{pooled.targets.shape}')

    print(f'wrote {args.out_prefix}_windows.tsv and _meta.json')


if __name__ == '__main__':
    main()
