# CSV-Net

## Interpretable Graph Learning on Preoperative Biopsy for Prediction of Pathological Complete Response to Neoadjuvant Therapy in Breast Cancer

<div align=left><img width="90%" src="overall.png"/></div>

## Installation
Clone the repo:
```bash
git clone https://github.com/Houwentai/CSV_Net && cd CSV_Net
```
Create a conda environment and activate it:
```bash
conda create -n env python=3.10
conda activate env
pip install -r requirements.txt
```

## Offline stage
- **WSI preprocessing and Feature Embedding** 
  - Please refer to [CLAM](https://github.com/mahmoodlab/CLAM)--Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021).

- **Prototype Extraction**
  ```bash
  python prototype.py
  ```

## Online stage
- **Construction of Phenotype Graph and Prototype Graph**
  - Please refer to **pp_graph_generation.ipynb**

- **Training of CSV-NET**
  - Running the following command-line for model training:
  ```bash
  python train.py
  ```

- **Validation of CSV-NET**
  - Running the following command-line for model inference and result statistics:
  ```bash
  python validation.py
  ```

## Saved model
We provide our trained CSV_Net model at [saved_model](https://github.com/Houwentai/CSV_Net/tree/main/saved_model), which performing as:
| Dataset | ROC-AUC |
| ----- |:--------:|
| Internel validation set | 0.845 |
| Externel validation set | 0.815 |

## Citation
- If you found our work useful in your research, please consider citing our work at:
```
TBD
```
