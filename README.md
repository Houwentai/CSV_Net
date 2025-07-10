# CSV_Net

## Interpretable Graph Learning on Preoperative Biopsy for Prediction of Pathological Complete Response to Neoadjuvant Therapy in Breast Cancer

<div align=left><img width="70%" src="Overall.png"/></div>

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

## Data Preprocess
***gendata.ipynb*** shows how to transform a histological image into the hierarchical graphs. After the data processing is completed, put all hierarchical graphs into a folder. The form is as follows:
```bash
PYG_Data
   └── Dataset
          ├── pyg_data_1.pkl
          ├── pyg_data_2.pkl
                    :
          └── pyg_data_n.pkl
```

## Training
First, setting the data splits and hyperparameters in the file ***train.py***. Then, experiments can be run using the following command-line:
```bash
cd train
python train.py
```
The trained model will be saved in the folder ***SavedModels***. 

## Saved models
We provide a saved model, which performing as:
| Dataset | Macro AUC |
| ----- |:--------:|
| Internel validation set | 0.845 |
| Externel validation set | 0.815 |

## Inference
Using the following command-line for model inference and result statistics:
```bash
cd inference
python inference_<experiments>.py
```

## Citation
- If you found our work useful in your research, please consider citing our work at:
```
TBD
```
