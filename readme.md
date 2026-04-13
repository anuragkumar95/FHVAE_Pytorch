This repository implements the Factorized Heirarchial Variational AutoEncoders in Pytorch. If you would like to use this work, please cite the following:
```
@inproceedings{hsu2017learning,
  title={Unsupervised Learning of Disentangled and Interpretable Representations from Sequential Data},
  author={Hsu, Wei-Ning and Zhang, Yu and Glass, James},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017},
}
```
[The original TF code for the FHVAE](https://github.com/wnhsu/FactorizedHierarchicalVAE).

The current code uses FHVAE to cluster EEG signals. We have prepared the datasets for both SparrKULee and KUL datasets. You would need to write your own custom code if you want to use this work for clustering audios or other signals. 

* To prepare the SparrKULee dataset, follow the instructions here : https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND
* To prepare the KUL dataset, follow the instructions here : https://zenodo.org/records/4004271

Store the KUL dataset at `ROOT_DIR/KUL_eeg/...`
Store the SparrKULee dataset at `ROOT_DIR/SparrKULee/...`

To run training, 
```
python train_FHVAE.py --conf ./conf/sup-fhvae_conf.json
```
<img width="1133" height="680" alt="cluster" src="https://github.com/user-attachments/assets/d6e6c6ba-7d41-4c4e-9227-4ac3e6e43953" />

