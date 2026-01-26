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

To run training, 
```
python train_FHVAE.py -c ./conf/fhvae_conf.json
```