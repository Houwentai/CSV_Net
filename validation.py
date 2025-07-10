from sklearn.model_selection import train_test_split
import os
import random
import torch.optim as optim
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from model import GCNModel
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # cuDNN's auto-tuner    


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    seed = 13
    setup_seed(seed)

  
    test_pp_graph_path = "./PP_LargestComponent_Graph_DIRECTORY_test/"
    test_pp_graph_names = []
    for i in range(len(test_wsi_names)):
        test_pp_graph_names.append(os.path.join(test_pp_graph_path,test_wsi_names[i]+".pt"))

    # 测试过程
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNModel(num_node_features = 1024, num_classes = 1, num_clinical_features = 7, graph_hidden_dim= 512 , attention_hidden_dim = 16).to(device)
    criterion = nn.BCELoss()
    test_ids = np.arange(len(test_pp_graph_names))

    model = GCNModel(num_node_features = 1024, num_classes = 1, num_clinical_features = 7, graph_hidden_dim= 512 , attention_hidden_dim = 16).to(device)
    model.load_state_dict(torch.load('./saved_model/checkpoint.pth')) 
    with torch.no_grad():
        model.eval()
        test_label = []
        test_prob = []
        test_patients = []
        running_loss = 0.0
        for index in test_ids:
            label = torch.tensor(test_labels[index], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            graph_data = torch.load(test_pp_graph_names[index]).to(device)
            clinic_feature = torch.tensor(test_clinic_features[index], dtype=torch.float32).unsqueeze(0).to(device)

            output,_ = model(graph_data, clinic_feature)
            output.squeeze(0)
            loss = criterion(output, label)
            running_loss += loss.item() 

            test_label.append(label.item())
            test_prob.append(output.item())
            
    test_auc = roc_auc_score(test_label, test_prob)
    print(f'test_auc: {test_auc:.4f}')

