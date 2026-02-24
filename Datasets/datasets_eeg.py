from torch.utils.data import Dataset
import os
from collections import OrderedDict
import numpy as np
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

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
        print(f"{split} {self.__class__.__name__} created with {len(self.segs)} segments and {self.n_seqs} seqs.")

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
    

class Joint_AUD_EEG_Dataset(Dataset):
    def __init__(self, root_dir, csv, mvn_path, seg_len, seg_shift, rand_seg, split, subj_norm=False):
        super().__init__()
        print(f"Preparing {split} dataset. If creating the val/test dataset, make sure the bs is set to 1.")
        if isinstance(csv, str) or isinstance(csv, Path):
            self.csv = pd.read_csv(csv)
            grouped = self.csv.groupby('split')
            for name, group in grouped:
                if name == split:
                    self.csv = group.reset_index(drop=True)
            del grouped
        else:
            self.csv = csv

        self.root = root_dir

        self.spk_map = {
            "podcast": "spk-00",
            "audiobook_1":"spk-01",
            "audiobook_2":"spk-02",
            "audiobook_3":"spk-02",
            "audiobook_4": "spk-03",
            "audiobook_5": "spk-04",
            "audiobook_6": "spk-04",
            "audiobook_7": "spk-05",
            "audiobook_8": "spk-06",
            "audiobook_9": "spk-07",
            "audiobook_10": "spk-08",
            "audiobook_11": "spk-09",
            "audiobook_12": "spk-10",
            "audiobook_13": "spk-11",
            "audiobook_14": "spk-12",
        }

        self.spk2idx = {}
        for _, v in self.spk_map.items():
            if v not in self.spk2idx:
                curr_idx = len(self.spk2idx)
                self.spk2idx[v] = curr_idx

        self.mvn_params = {'eeg':{}, 'spec':{}}
        self.subj_mvn_params = {}
        self._mvn_prep(mvn_path)
    
        self.eeg_sr = 64
        self.aud_fr = 100 #Frame rate of spectrograms
        self.seg_len = seg_len
        self.seg_shift = seg_shift
        self.rand_seg = rand_seg
        self.split = split
        self.subj_norm = subj_norm

        self.seqs = []
        self.seq_lens = self.csv['lens'].to_list()
        self.eeg_paths = self.csv['eeg_path'].to_list()
        self.spec_paths = self.csv['sti_feature_path'].to_list()

        # Computing subject mean and variation
        print('Computing subject mean and variance')
        self._compute_subj_mvn()

        print(f"Loading spectograms for faster access...")
        self.specs = {}
        for path in tqdm(self.spec_paths):
            path = path.replace('ROOT', self.root)
            if path not in self.specs:
                with open(path, 'rb') as fp:
                    f = np.load(fp)
                self.specs[path] = f

        for _, row in self.csv.iterrows():
            seq_id = f"{row['subj']}_{row['sess']}_{row['trial']}"
            self.seqs.append(seq_id)
        self.seq2idx = dict([(seq, i) for i, seq in enumerate(self.seqs)])
        self.idx2seq = dict([(i, seq) for i, seq in enumerate(self.seqs)])

        self.esegs, self.asegs, self.seq_nsegs = self._make_segs(
            self.seqs, self.seq_lens, self.seg_len, self.seg_shift, self.rand_seg
        )
        self.n_seqs = len(self.seqs)
        print(f"Joint Dataset with {len(self.esegs)} segs and {len(self.seqs)} seqs")

    def _compute_subj_mvn(self, ):
        """Compute mean and variance normalization."""
        compute = {}
        for _, row in self.csv.iterrows():
            subj = row['subj']
            if subj not in self.subj_mvn_params:
                self.subj_mvn_params[subj] = {'mean':0, 'std':0}
            if subj not in compute:
                compute[subj] = {'x':0, 'x2':0, 'n':0}
            feat_path = row['eeg_path'].replace('ROOT', self.root)
            with open(feat_path, "rb") as f:
                feat = np.load(f)
            
            compute[subj]['x'] += np.sum(feat, axis=1, keepdims=True)
            compute[subj]['x2'] += np.sum(feat ** 2, axis=1, keepdims=True)
            compute[subj]['n'] += feat.shape[1]
        
        for subj in compute:
            mean = compute[subj]['x'] / compute[subj]['n']
            std = np.sqrt((compute[subj]['x2'] / compute[subj]['n']) - mean**2)
            self.subj_mvn_params[subj]['mean'] = mean
            self.subj_mvn_params[subj]['std'] = std

    def apply_subj_mvn(self, feat, subj):
        return (feat - self.subj_mvn_params[subj]["mean"]) / self.subj_mvn_params[subj]["std"]

    def apply_mvn(self, feats, feat_type='eeg'):
        """Apply mean and variance normalization."""
        if self.mvn_params is None:
            return feats
        else:
            return (feats - self.mvn_params[feat_type]["mean"]) / self.mvn_params[feat_type]["std"]

    def _mvn_prep(self, mvn_path):
        if mvn_path is not None:
            if not os.path.exists(mvn_path):
                for feat_type in ['eeg', 'spec']:
                    print(f"Computing {feat_type} mvn...")
                    self.mvn_params[feat_type] = self._compute_mvn(feat_type)
                with open(mvn_path, "w") as f:
                    json.dump(self.mvn_params, f)
            else:
                with open(mvn_path) as f:
                    self.mvn_params = json.load(f)
        else:
            self.mvn_params = None

    def _compute_mvn(self, feat_type):
        """Compute mean and variance normalization."""
        n, x, x2 = 0.0, 0.0, 0.0
        for _, row in self.csv.iterrows():
            if feat_type == 'eeg':
                feat_col_name = 'eeg_path'
            else:
                feat_col_name = 'sti_feature_path'
            feat_path = row[feat_col_name].replace('ROOT', self.root)
            with open(feat_path, "rb") as f:
                feat = np.load(f)
            x += np.sum(feat, axis=1, keepdims=True)
            x2 += np.sum(feat ** 2, axis=1, keepdims=True)
            n += feat.shape[1]
        mean = x / n
        std = np.sqrt(x2 / n - mean ** 2)
        return {"mean": mean.tolist(), "std": std.tolist()}

    def undo_mvn(self, feats, feat_type='eeg'):
        """Undo mean and variance normalization."""
        if self.mvn_params is None:
            return feats
        else:
            return feats * self.mvn_params[feat_type]["std"] + self.mvn_params[feat_type]["mean"]

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
        eeg_segs = []
        aud_segs = []
        nsegs = []
        for seq, l in zip(seqs, lens):
            nseg = (l - seg_len) // seg_shift + 1
            nsegs.append(nseg)
            if rand_seg:
                starts = np.random.choice(range(l - seg_len + 1), rand_seg)
            else:
                starts = np.arange(nseg) * seg_shift
            for eeg_start in starts:
                eeg_end = eeg_start + seg_len
                
                st_dur = eeg_start / self.eeg_sr
                en_dur = eeg_end / self.eeg_sr

                aud_start = int(st_dur * self.aud_fr)
                aud_end = int(en_dur * self.aud_fr)

                eeg_segs.append(Segment(seq, eeg_start, eeg_end))
                aud_segs.append(Segment(seq, aud_start, aud_end))
        return eeg_segs, aud_segs, nsegs
    
    def __len__(self, ):  
        if self.split == 'train':
            return len(self.esegs)
        else:
            return len(self.seqs)

    def __getitem__(self, index):
        """Returns key(sequence), feature, and number of segments."""
        if self.split == 'train':
            # Lookup eeg, spec segments and apply mvn
            eseg = self.esegs[index]
            aseg = self.asegs[index]
            idx = self.seq2idx[eseg.seq]
            subj = eseg.seq.split('_')[0]
            eeg_path = self.eeg_paths[idx].replace('ROOT', self.root)
            with open(eeg_path, "rb") as f:
                efeat = np.load(f)[:, eseg.start : eseg.end]
            sti_path = self.spec_paths[idx].replace('ROOT', self.root)
            sfeat = self.specs[sti_path][:, aseg.start : aseg.end]
            if self.subj_norm:
                efeat = self.apply_subj_mvn(efeat, subj)
            else:
                efeat = self.apply_mvn(efeat, feat_type='eeg')
            sfeat = self.apply_mvn(sfeat, feat_type='spec')
            
            # Get speaker labels
            for k in self.spk_map:
                if k in eseg.seq:
                    spk = self.spk_map[k]
                    s_label = self.spk2idx[spk]
                    break

            nsegs = self.seq_nsegs[idx]
            efeat, sfeat = efeat.T, sfeat.T
        else:
            seq = self.seqs[index]
            esegs = [seg for seg in self.esegs if seg.seq == seq]
            asegs = [seg for seg in self.asegs if seg.seq == seq]
            subj = seq.split('_')[0]
            eeg_path = self.eeg_paths[index].replace('ROOT', self.root)
            with open(eeg_path, "rb") as fp:
                efeat = np.load(fp)
            sti_path = self.spec_paths[index].replace('ROOT', self.root)
            sfeat = self.specs[sti_path]
            if self.subj_norm:
                efeat = self.apply_subj_mvn(efeat, subj)
            else:
                efeat = self.apply_mvn(efeat, feat_type='eeg')
            sfeat = self.apply_mvn(sfeat, feat_type='spec')

            EFEAT, SFEAT = [], []
            for seg in esegs:
                f_t = efeat[:, seg.start : seg.end].T
                EFEAT.append(f_t[np.newaxis, :, :])
            for seg in asegs:
                f_t = sfeat[:, seg.start : seg.end].T
                SFEAT.append(f_t[np.newaxis, :, :])
            efeat = np.concatenate(EFEAT, axis=0)
            sfeat = np.concatenate(SFEAT, axis=0)

            # Get speaker label
            for k in self.spk_map:
                if k in seq:
                    spk = self.spk_map[k]
                    s_label = self.spk2idx[spk]
                    break

            s_label = np.asarray([s_label for _ in range(efeat.shape[0])])
            idx = np.asarray([-1 for i in range(efeat.shape[0])])
            nsegs = np.asarray([efeat.shape[0] for _ in range(efeat.shape[0])])
        return idx, (efeat, sfeat), nsegs, s_label


class NumpyEEGSpkDataset(NumpyEEGDataset):
    def __init__(self, csv, min_len, mvn_path, seg_len, seg_shift, rand_seg, split):
        super().__init__(
            csv, min_len, mvn_path, seg_len, seg_shift, rand_seg, 'train'
        )
        self.spk_map = {
            "podcast": "spk-00",
            "audiobook_1":"spk-01",
            "audiobook_2":"spk-02",
            "audiobook_3":"spk-02",
            "audiobook_4": "spk-03",
            "audiobook_5": "spk-04",
            "audiobook_6": "spk-04",
            "audiobook_7": "spk-05",
            "audiobook_8": "spk-06",
            "audiobook_9": "spk-07",
            "audiobook_10": "spk-08",
            "audiobook_11": "spk-09",
            "audiobook_12": "spk-10",
            "audiobook_13": "spk-11",
            "audiobook_14": "spk-12",
        }

        self.spk2idx = {}
        for _, v in self.spk_map.items():
            if v not in self.spk2idx:
                curr_idx = len(self.spk2idx)
                self.spk2idx[v] = curr_idx

        self.stimuli = [Path(p).stem for p in self.csv['stimuli_path'].tolist()]

    def __len__(self, ):  
        return len(self.segs)

    def __getitem__(self, index):
        seg = self.segs[index]
        idx = self.seq2idx[seg.seq]
        stimuli = self.stimuli[idx].split('_')[2:]
        if len(stimuli) > 2:
            stimuli = "_".join(stimuli[:-1])
        else:
            stimuli = "_".join(stimuli)
        if 'podcast' in stimuli:
            stimuli = 'podcast'
        assert stimuli in self.spk_map, f"{stimuli} not found in {self.spk_map.keys()}..."
        spk = self.spk_map[stimuli]
        label = self.spk2idx[spk]
        with open(self.eeg_paths[idx], "rb") as f:
            feat = np.load(f)[:, seg.start : seg.end]
        feat = self.apply_mvn(feat)
        feat = feat.T
        return feat, label

class JointEEGAudSpkDataset(Joint_AUD_EEG_Dataset):
    def __init__(self, csv, min_len, mvn_path, seg_len, seg_shift, rand_seg, split):
        
        np.random.seed(777)

        self.spk_map = {
            "podcast": "spk-00",
            "audiobook_1":"spk-01",
            "audiobook_2":"spk-02",
            "audiobook_3":"spk-02",
            "audiobook_4": "spk-03",
            "audiobook_5": "spk-04",
            "audiobook_6": "spk-04",
            "audiobook_7": "spk-05",
            "audiobook_8": "spk-06",
            "audiobook_9": "spk-07",
            "audiobook_10": "spk-08",
            "audiobook_11": "spk-09",
            "audiobook_12": "spk-10",
            "audiobook_13": "spk-11",
            "audiobook_14": "spk-12",
        }
        
        super().__init__(
            csv, mvn_path, seg_len, seg_shift, rand_seg, 'train'
        )

        if split != 'all':
            np.random.shuffle(self.seqs)
            train_seqs = self.seqs[:int(len(self.seqs)*0.7)]
            val_seqs = self.seqs[int(len(self.seqs)*0.7):int(len(self.seqs)*0.85)]
            test_seqs = self.seqs[int(len(self.seqs)*0.85):]

            if split == 'train':
                seq_list = train_seqs
            if split == 'val':
                seq_list = val_seqs
            if split == 'test':
                seq_list = test_seqs
        
            self.esegs = [seg for seg in self.esegs if seg.seq in seq_list]
            self.asegs = [seg for seg in self.esegs if seg.seq in seq_list]

        print(f"Split {split} has {len(self.esegs)} seqs...")
        
        self.spk2idx = {}
        self.subj2idx = {}
        for _, v in self.spk_map.items():
            if v not in self.spk2idx:
                curr_idx = len(self.spk2idx)
                self.spk2idx[v] = curr_idx
        
        for _, row in self.csv.iterrows():
            subj = row['subj']
            if subj not in self.subj2idx:
                curr_idx = len(self.subj2idx)
                self.subj2idx[subj] = curr_idx

        self.stimuli = [Path(p).stem for p in self.csv['stimuli_path'].tolist()]

    def __len__(self, ):  
        return len(self.esegs)

    def __getitem__(self, index):
        eseg = self.esegs[index]
        aseg = self.asegs[index]
        idx = self.seq2idx[eseg.seq]
        subj = eseg.seq.split('_')[0]
        with open(self.eeg_paths[idx], "rb") as f:
            efeat = np.load(f)[:, eseg.start : eseg.end]
        stimuli = self.stimuli[idx].split('_')[2:]
        if len(stimuli) > 2:
            stimuli = "_".join(stimuli[:-1])
        else:
            stimuli = "_".join(stimuli)
        if 'podcast' in stimuli:
            stimuli = 'podcast'
        assert stimuli in self.spk_map, f"{stimuli} not found in {self.spk_map.keys()}..."
        spk = self.spk_map[stimuli]
        label = self.spk2idx[spk]
        subj_label = self.subj2idx[subj]
        sti_path = self.spec_paths[idx] 
        sfeat = self.specs[sti_path][:, aseg.start : aseg.end]
        efeat = self.apply_mvn(efeat, feat_type='eeg')
        sfeat = self.apply_mvn(sfeat, feat_type='spec')
        efeat, sfeat = efeat.T, sfeat.T
        return (efeat, sfeat), label, subj_label