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

class Joint_AUD_EEG_Dataset(Dataset):
    def __init__(self, root_dir, csv, mvn_path, seg_len, seg_shift, rand_seg, split, ds='all'):
        super().__init__()
        self.csv = pd.read_csv(csv)
        if ds != 'all':
            self.csv = self.csv[self.csv['dataset']==ds]
            
        print(f"Preparing {split} dataset. If creating the val/test dataset, make sure the bs is set to 1.")
        if isinstance(csv, str) or isinstance(csv, Path):
            grouped = self.csv.groupby('split')
            for name, group in grouped:
                if name == split:
                    self.csv = group.reset_index(drop=True)
            del grouped
        else:
            self.csv = csv

        self.root = root_dir

        self.speakers = self.csv['speaker'].tolist()
        self.spk2idx = {val: i for i, val in enumerate(self.csv['speaker'].unique())}

        self.stimuli = list(self.csv['stimuli'].unique())
        self.aud_seq2idx = {val: i for i, val in enumerate(self.stimuli)}
        self.aud_idx2seq = {i: val for i, val in enumerate(self.stimuli)}
        
        self.mvn_params = {}
        self._mvn_prep(mvn_path)
        
        self.aud_fr = 100 #Frame rate of spectrograms
        self.eeg_sr = 128
        self.seg_len = seg_len
        self.seg_shift = seg_shift
        self.rand_seg = rand_seg
        self.split = split

        self.seqs = []
        self.subjects = []
        self.eeg_sr_list = self.csv['sr'].to_list()
        self.seq_lens = self.csv['lens'].to_list()
        self.eeg_paths = self.csv['eeg_path'].to_list()
        self.spec_paths = self.csv['sti_feature_path'].unique().tolist()

        print(f"Loading spectograms for faster access...")
        self.specs = {}
        for path in tqdm(self.spec_paths):
            path = path.replace('ROOT', self.root)
            if path not in self.specs:
                with open(path, 'rb') as fp:
                    f = np.load(fp)
                self.specs[path] = f

        for _, row in self.csv.iterrows():
            seq_id = f"{row['subject']}_{row['sess']}_{row['trial']}"
            subj_id = f"{row['dataset']}_{row['subject']}"
            self.seqs.append(seq_id)
            self.subjects.append(subj_id)

        self.sub2idx = {val: i for i, val in enumerate(set(self.subjects))}

        self.eeg_seq2idx = dict([(seq, i) for i, seq in enumerate(self.seqs)])
        self.eeg_idx2seq = dict([(i, seq) for i, seq in enumerate(self.seqs)])

        self.esegs, self.asegs, self.seq_nsegs = self._make_segs(
            self.seqs, self.seq_lens, self.eeg_sr_list, self.seg_len, self.seg_shift, self.rand_seg
        )

        self.sub_nsegs = [0 for _ in self.sub2idx]
        self.spk_nsegs = [0 for _ in self.spk2idx]
        self.aud_nsegs = [0 for _ in self.aud_seq2idx]
        for i, spk in enumerate(self.speakers):
            spk_idx = self.spk2idx[spk]
            self.spk_nsegs[spk_idx] += self.seq_nsegs[i]
        for i, sub in enumerate(self.subjects):
            sub_idx = self.sub2idx[sub]
            self.sub_nsegs[sub_idx] += self.seq_nsegs[i]
        for i, sti in enumerate(self.csv['stimuli']):
            sti_idx = self.aud_seq2idx[sti]
            self.aud_nsegs[sti_idx] += self.seq_nsegs[i]
            
        self.n_eeg_seqs = len(self.seqs)
        self.n_aud_seqs = len(self.stimuli)
        self.n_subj = len(list(set(self.subjects)))
        self.n_speaker = len(list(set(self.speakers)))

        print(f"Joint Dataset with {len(self.esegs)} segs, {self.n_eeg_seqs} EEG seqs and {self.n_aud_seqs} audio seqs")
        print(f"Joint Dataset contains {self.n_subj} subjects and {self.n_speaker} stimulus.")

    def apply_mvn(self, feats, feat_type='eeg', ds='KUL'):
        """Apply mean and variance normalization."""
        if self.mvn_params is None:
            return feats
        else:
            return (feats - self.mvn_params[feat_type][ds]["mean"]) / self.mvn_params[feat_type][ds]["std"]

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
        print(f"Computing MVN...")
        x, x2, n = {'SPARR':0, 'KUL':0}, {'SPARR':0, 'KUL':0}, {'SPARR':0, 'KUL':0}
        for _, row in tqdm(self.csv.iterrows(), total=len(self.csv)):
            if feat_type == 'eeg':
                feat_col_name = 'eeg_path'
            else:
                feat_col_name = 'sti_feature_path'
            feat_path = row[feat_col_name].replace('ROOT', self.root)
            with open(feat_path, "rb") as f:
                feat = np.load(f, allow_pickle=True)
            if feat_type == 'eeg':
                if feat.shape[-1] == 64:
                    feat = feat.T
                x[row['dataset']] += np.sum(feat, axis=1, keepdims=True)
                x2[row['dataset']] += np.sum(feat ** 2, axis=1, keepdims=True)
                n[row['dataset']] += feat.shape[1]
            else:
                if feat.shape[-1] == 201:
                    feat = feat.T
                x[row['dataset']] += np.sum(feat, axis=1, keepdims=True)
                x2[row['dataset']] += np.sum(feat ** 2, axis=1, keepdims=True)
                n[row['dataset']] += feat.shape[1]
                
        mvn_params = {'SPARR':0, 'KUL':0}
        for ds in mvn_params:
            mean = x[ds] / n[ds]
            std = np.sqrt((x2[ds] / n[ds]) - mean ** 2)
            print(f"{ds} mean:{mean.shape, mean.mean()}, std:{std.shape, std.mean()}")
            mvn_params[ds] = {"mean": mean.tolist(), "std": std.tolist()}
        return mvn_params

    def undo_mvn(self, feats, feat_type='eeg', ds='KUL'):
        """Undo mean and variance normalization."""
        if self.mvn_params is None:
            return feats
        else:
            return feats * self.mvn_params[feat_type][ds]["std"] + self.mvn_params[feat_type][ds]["mean"]

    def _make_segs(
        self,
        seqs: list,
        lens: list,
        sr_list: list,
        seg_len: float = 0.5,
        seg_shift: float = 0.1,
        rand_seg: bool = False,
    ):
        """Make segments from a list of sequences.

        Args:
            seqs:      List of sequences
            lens:      List of sequence lengths
            seg_len:   Segment length in secs
            seg_shift: Segment shift if rand_seg is False; otherwise randomly 
                           extract floor(seq_len/seg_shift) segments per sequence (secs)
            rand_seg:  If True, randomly extract segments
        """
        eeg_segs = []
        aud_segs = []
        nsegs = []
        
        for i, (seq, l) in enumerate(zip(seqs, lens)):
            eeg_sr = sr_list[i]
      
            # change secs -> samples
            seg_len_samples = int(seg_len * eeg_sr)
            seg_shift_samples = int(seg_shift * eeg_sr)
            
            nseg = (l - seg_len_samples) // seg_shift_samples + 1
            nsegs.append(nseg)
            
            if rand_seg:
                starts = np.random.choice(range(l - seg_len_samples + 1), rand_seg)
            else:
                starts = np.arange(nseg) * seg_shift_samples
            
            for eeg_start in starts:
                eeg_end = eeg_start + seg_len_samples

                # change samples -> secs
                st_dur = eeg_start / eeg_sr
                en_dur = eeg_end / eeg_sr

                eeg_segs.append(Segment(seq, st_dur, en_dur))
                aud_segs.append(Segment(seq, st_dur, en_dur))
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
            
            e_idx = self.eeg_seq2idx[eseg.seq]
            sti = self.csv['stimuli'][e_idx]
            ds = self.csv['dataset'][e_idx]
            a_idx = self.aud_seq2idx[sti] 
            
            #subj = eseg.seq.split('_')[0]
            eeg_path = self.eeg_paths[e_idx].replace('ROOT', self.root)
            with open(eeg_path, "rb") as f:
                efeat = np.load(f)
                if efeat.shape[-1] == 64:
                    efeat = efeat.T
                # If EEG is at 64Hz upsample it to 128Hz
                eeg_start = int(eseg.start * self.eeg_sr_list[e_idx])
                eeg_end = int(eseg.end * self.eeg_sr_list[e_idx])
                efeat = efeat[:, eeg_start : eeg_end]
                if self.eeg_sr_list[e_idx] == 64:
                    efeat = scipy.signal.resample_poly(efeat, up=2, down=1, axis=-1)
                    
            sti_f_path = self.csv['sti_feature_path'][e_idx]
            sti_f_path = sti_f_path.replace('ROOT', self.root)
            aud_start = int(aseg.start * self.aud_fr)
            aud_end = int(aseg.end * self.aud_fr)
            sfeat = self.specs[sti_f_path][:, aud_start : aud_end]
            # with open(sti_f_path, "rb") as f:
            #     sfeat = np.load(f)
            #     if sfeat.shape[-1] == 201:
            #         sfeat = sfeat.T
            #     aud_start = int(aseg.start * self.aud_fr)
            #     aud_end = int(aseg.end * self.aud_fr)
            #     sfeat = sfeat[:, aud_start : aud_end]

            efeat = self.apply_mvn(efeat, feat_type='eeg', ds=ds)
            sfeat = self.apply_mvn(sfeat, feat_type='spec', ds=ds)
            
            # Get speaker labels
            spk = self.speakers[e_idx]
            spk_label = self.spk2idx[spk]
            spk_nsegs = self.spk_nsegs[spk_label]

            # Get subject labels
            sub = self.subjects[e_idx]
            sub_label = self.sub2idx[sub]
            sub_nsegs = self.sub_nsegs[sub_label]
            
            nsegs = self.seq_nsegs[e_idx]
            a_nsegs = self.aud_nsegs[a_idx]
            efeat, sfeat = efeat.T, sfeat.T
            
        else:
            seq = self.seqs[index]
            sti = self.csv['stimuli'][index]
            ds = self.csv['dataset'][index]
            a_idx = self.aud_seq2idx[sti] 
           
            esegs = [seg for seg in self.esegs if seg.seq == seq]
            asegs = [seg for seg in self.asegs if seg.seq == seq]
            #subj = seq.split('_')[0]
            
            eeg_path = self.eeg_paths[index].replace('ROOT', self.root)
            with open(eeg_path, "rb") as fp:
                efeat = np.load(fp)
                if efeat.shape[-1] == 64:
                    efeat = efeat.T
                # If EEG is at 64Hz upsample it to 128Hz
                if self.eeg_sr_list[index] == 64:
                    efeat = scipy.signal.resample_poly(efeat, up=2, down=1, axis=-1)
                    
            sti_f_path = self.csv['sti_feature_path'][index]
            sti_f_path = sti_f_path.replace('ROOT', self.root)
            sfeat = self.specs[sti_f_path]
            # with open(sti_f_path, "rb") as f:
            #     sfeat = np.load(f)

            efeat = self.apply_mvn(efeat, feat_type='eeg', ds=ds)
            sfeat = self.apply_mvn(sfeat, feat_type='spec', ds=ds)

            EFEAT, SFEAT = [], []
            for seg in esegs:
                eeg_start = int(seg.start * self.eeg_sr)
                eeg_end = int(seg.end * self.eeg_sr)
                f_t = efeat[:, eeg_start : eeg_end].T
                EFEAT.append(f_t[np.newaxis, :, :])
            for seg in asegs:
                aud_start = int(seg.start * self.aud_fr)
                aud_end = int(seg.end * self.aud_fr)
                f_t = sfeat[:, aud_start : aud_end].T
                SFEAT.append(f_t[np.newaxis, :, :])
            efeat = np.concatenate(EFEAT, axis=0)
            sfeat = np.concatenate(SFEAT, axis=0)

            # Get speaker labels
            spk = self.speakers[index]
            spk_label = self.spk2idx[spk]
            spk_nsegs = self.spk_nsegs[spk_label]

            # Get subject labels
            sub = self.subjects[index]
            sub_label = self.sub2idx[sub]
            sub_nsegs = self.sub_nsegs[sub_label]

            spk_label = np.asarray([spk_label for _ in range(efeat.shape[0])])
            sub_label = np.asarray([sub_label for _ in range(efeat.shape[0])])
            sub_nsegs = np.asarray([sub_nsegs for _ in range(efeat.shape[0])])
            spk_nsegs = np.asarray([spk_nsegs for _ in range(efeat.shape[0])])
            nsegs = np.asarray([efeat.shape[0] for _ in range(efeat.shape[0])])
            e_idx = np.asarray([-1 for i in range(efeat.shape[0])])
            nsegs = np.asarray([efeat.shape[0] for _ in range(efeat.shape[0])])
            a_nsegs = np.asarray([self.aud_nsegs[a_idx] for _ in range(efeat.shape[0])])
            a_idx = np.asarray([-1 for i in range(efeat.shape[0])])

        return e_idx, a_idx, efeat, sfeat, nsegs, a_nsegs, spk_nsegs, sub_nsegs, spk_label, sub_label