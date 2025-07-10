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

  
    pp_graph_path = "./PP_LargestComponent_Graph_DIRECTORY"
    pp_graph_names = []
    for i in range(len(wsi_names)):
        pp_graph_names.append(os.path.join(pp_graph_path,wsi_names[i]+".pt"))  
    patient_ids = np.arange(len(pp_graph_names))
    train_index, val_index = train_test_split(patient_ids, test_size=0.4, stratify=labels, random_state=0)
    
    # 训练过程
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNModel(num_node_features = 1024, num_classes = 1, num_clinical_features = 7, graph_hidden_dim= 512 , attention_hidden_dim = 16).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    num_epochs = 20
    
    train_loss_his = []
    val_loss_his = []
    loss_his = []
    
    check_point = 0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_label = []
        train_prob = []
        train_split_patients = []
        running_loss = 0.0
        random.shuffle(train_index)
        
        for index in train_index:
            label = torch.tensor(labels[index], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            graph_data = torch.load(pp_graph_names[index]).to(device)
            clinic_feature = torch.tensor(clinic_features[index], dtype=torch.float32).unsqueeze(0).to(device)
            output,_ = model(graph_data, clinic_feature)
            output.squeeze(0)
            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_his.append(loss.item())
            running_loss += loss.item() 
            train_label.append(label.item())
            train_prob.append(output.item())
            
        train_auc = roc_auc_score(train_label, train_prob)
        train_loss = running_loss / len(train_index)
        train_loss_his.append(train_loss)
    torch.save(model.state_dict(), './saved_model/checkpoint.pth')
