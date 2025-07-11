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
- **WSI Preprocessing** 
  - Please refer to [CLAM](https://github.com/mahmoodlab/CLAM)--Lu M Y, Williamson D F K, Chen T Y, et al. Data-efficient and weakly supervised computational pathology on whole-slide images[J]. Nature biomedical engineering, 2021, 5(6): 555-570.
 
- **Feature Embedding** 
  - Please refer to [UNI](https://github.com/mahmoodlab/UNI)--Chen R J, Ding T, Lu M Y, et al. Towards a general-purpose foundation model for computational pathology[J]. Nature Medicine, 2024, 30(3): 850-862.
 
- **Prototype Extraction**
  - Running the following command-line for prototype extraction:
  ```bash
  python prototype_extraction.py
  ```

## Online stage
- **Construction of Phenotype Graph and Prototype Graph**
  - Running the following command-line for the construction of phenotype graph and prototype graph:
  ```bash
  python pp_graph_construction.py
  ```
  
- **Training of CSV-NET**
  - Running the following command-line for model training:
  ```bash
  python training.py
  ```

- **Validation of CSV-NET**
  - Running the following command-line for model inference and result statistics:
  ```bash
  python validation.py
  ```

## Saved model
- Our trained CSV_Net model is avaliable at [saved_model](https://github.com/Houwentai/CSV_Net/tree/main/saved_model), which performing as:
  | Dataset | ROC-AUC |
  | ----- |:--------:|
  | Internel validation set | 0.845 |
  | Externel validation set | 0.815 |

## Citation
- If you found our work useful in your research, please consider citing our work at:
```
TBD
```
