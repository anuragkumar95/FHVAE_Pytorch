from torch.utils.data import Dataset
import os
from collections import OrderedDict
import numpy as np
import json
from pathlib import Path
import pandas as pd


class Segment(object):
    """Represents an EEG segment."""

    def __init__(self, seq, start, end):
        self.seq = seq
        self.start = start
        self.end = end

    def __str__(self):
        return f"{self.seq}, {self.start}, {self.end}"

    def __repr__(self):
        return str(self)


class BaseDataset(Dataset):
    def __init__(
        self,
        csv: Path,
        min_len: int = 1,
        mvn_path: str = None,
        seg_len: int = 32,
        seg_shift: int = 24,
        rand_seg: bool = False,
        sequence_list=None,
    ):
        """
        Args:
            feat_scp:  Feature scp path
            len_scp:   Sequence-length scp path
            min_len:   Keep sequence no shorter than min_len
            mvn_path:       Path to file storing the mean and variance of the sequences
                                for normalization
            seg_len:   Segment length
            seg_shift: Segment shift if seg_rand is False; otherwise randomly
                                extract floor(seq_len/seg_shift) segments per sequence
            rand_seg: If True, randomly extract segments
        """
        
        self.csv = csv

        self.seg_len = seg_len
        self.seg_shift = seg_shift
        self.rand_seg = rand_seg

        self.seqs = []
        for _, row in self.csv.iterrows():
            seq_id = f"{row['subj']}_{row['sess']}_{row['trial']}"
            self.seqs.append(seq_id)
        self.seq2idx = dict([(seq, i) for i, seq in enumerate(self.seqs)])
        self.idx2seq = dict([(i, seq) for i, seq in enumerate(self.seqs)])
        self.seq_lens = self.csv['lens'].tolist()

        self.segs, self.seq_nsegs = self._make_segs(
            self.seqs, self.seq_lens, self.seg_len, self.seg_shift, self.rand_seg
        )

    def apply_mvn(self, feats):
        """Apply mean and variance normalization."""
        if self.mvn_params is None:
            return feats
        else:
            return (feats - self.mvn_params["mean"]) / self.mvn_params["std"]

    def _mvn_prep(self, mvn_path):
        if mvn_path is not None:
            if not os.path.exists(mvn_path):
                self.mvn_params = self._compute_mvn()
                with open(mvn_path, "w") as f:
                    json.dump(self.mvn_params, f)
            else:
                with open(mvn_path) as f:
                    self.mvn_params = json.load(f)
        else:
            self.mvn_params = None

    def _compute_mvn(self):
        """Compute mean and variance normalization."""
        n, x, x2 = 0.0, 0.0, 0.0
        for _, row in self.csv.iterrows():
            feat_path = row['eeg_path']
            with open(feat_path, "rb") as f:
                feat = np.load(f)
            x += np.sum(feat, axis=1, keepdims=True)
            x2 += np.sum(feat ** 2, axis=1, keepdims=True)
            n += feat.shape[0]
        mean = x / n
        std = np.sqrt(x2 / n - mean ** 2)
        return {"mean": mean.tolist(), "std": std.tolist()}

    def undo_mvn(self, feats):
        """Undo mean and variance normalization."""
        if self.mvn_params is None:
            return feats
        else:
            return feats * self.mvn_params["std"] + self.mvn_params["mean"]

    def __len__(self):
        #return len(self.seqlist)
        """Better to override further down to get custom length access"""
        raise NotImplementedError()

    def __getitem__(self, index):
        """Returns key(sequence), feature, and number of segments."""
        raise NotImplementedError()

    def _make_seq_lists(self, seqlist):
        """Return lists of all sequences and the corresponding features and lengths."""
        keys, feats, lens = [], [], []
        for seq in seqlist:
            keys.append(seq)
            feats.append(self.feats[seq])
            lens.append(self.lens[seq])

        return keys, feats, lens

    def _make_segs(
        self,
        seqs: list,
        lens: list,
        seg_len: int = 20,
        seg_shift: int = 8,
        rand_seg: bool = False,
    ):
        """Make segments from a list of sequences.

        Args:
            seqs:      List of sequences
            lens:      List of sequence lengths
            seg_len:   Segment length
            seg_shift: Segment shift if rand_seg is False; otherwise randomly
                           extract floor(seq_len/seg_shift) segments per sequence
            rand_seg:  If True, randomly extract segments
        """
        segs = []
        nsegs = []
        for seq, l in zip(seqs, lens):
            nseg = (l - seg_len) // seg_shift + 1
            nsegs.append(nseg)
            if rand_seg:
                starts = np.random.choice(range(l - seg_len + 1), nseg)
            else:
                starts = np.arange(nseg) * seg_shift
            for start in starts:
                end = start + seg_len
                segs.append(Segment(seq, start, end))
        return segs, nsegs

class NumpyEEGDataset(BaseDataset):
    def __init__(
        self,
        csv: Path,
        min_len: int = 1,
        mvn_path: str = None,
        seg_len: int = 20,
        seg_shift: int = 8,
        rand_seg: bool = False,
        split='train',
    ):
        """
        Args:
            csv:       Path the csv file containing seq information
            min_len:   Keep sequence no shorter than min_len
            seg_len:   Segment length
            seg_shift: Segment shift if seg_rand is False; otherwise randomly
                                extract floor(seq_len/seg_shift) segments per sequence
            rand_seg: If True, randomly extract segments
        """
        print(f"Preparing {split} dataset. If creating the val/test dataset, make sure the bs is set to 1.")
        csv = pd.read_csv(csv)
        grouped = csv.groupby('split')
        for name, group in grouped:
            if name == split:
                csv = group.reset_index(drop=True)
        del grouped

        super().__init__(
            csv, min_len, mvn_path, seg_len, seg_shift, rand_seg
        )

        self._mvn_prep(mvn_path)
        self.n_seqs = len(self.seqs)
        self.split = split
        self.eeg_paths = list(self.csv['eeg_path'])
        assert len(self.eeg_paths) == len(self.seqs)
        print(f"{split} {self.__class__.__name__} created with {len(self.segs)} segments and {self.num_seqs} seqs.")

    def __len__(self, ):  
        if self.split == 'train':
            return len(self.segs)
        else:
            return len(self.seqs)

    def __getitem__(self, index):
        """Returns key(sequence), feature, and number of segments."""
        if self.split == 'train':
            seg = self.segs[index]
            idx = self.seq2idx[seg.seq]
            with open(self.eeg_paths[idx], "rb") as f:
                feat = np.load(f)[:, seg.start : seg.end]
            feat = self.apply_mvn(feat)
            nsegs = self.seq_nsegs[idx]
            feat = feat.T
        else:
            seq = self.seqs[index]
            segs = [seg for seg in self.segs if seg.seq == seq]
            with open(self.eeg_paths[index], "rb") as fp:
                f = np.load(fp)
            f = self.apply_mvn(f)
            feat = []
            for seg in segs:
                f_t = f[:, seg.start : seg.end].T
                feat.append(f_t[np.newaxis, :, :])
            feat = np.concatenate(feat, axis=0)
            idx = np.asarray([-1 for i in range(feat.shape[0])])
            nsegs = np.asarray([feat.shape[0] for i in range(feat.shape[0])])
        return idx, feat, nsegs