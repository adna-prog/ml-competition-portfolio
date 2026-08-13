# Third-party notice — prospective RealMLP

`gen_prospective_realmlp_nb.py` and the generated
`s6e8_prospective_realmlp.ipynb` adapt the compact PyTorch RealMLP
implementation from:

- **Work:** RealMLP for Predicting Smartphone Addiction
- **Author:** Zhenrui.Weng
- **Source:** https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction
- **Original license:** Apache License 2.0

The adapted files have been modified substantially for this repository:
prospective development/holdout splitting, fold-local preprocessing,
inner cross-fitted target encoding, sealed-holdout prediction, artifact
validation, deterministic seed, and Kaggle P100 compatibility were added.
No trained model, OOF prediction, test prediction, or other fitted artifact
from the source notebook is redistributed or used.

The Apache License 2.0 text is included as `APACHE-2.0.txt` in this directory.
