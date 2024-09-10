import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import matplotlib.pyplot as plt

# defined LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# loading data
new_data = pd.read_csv('/media/zzh/data/AMP/Transformerregression/data/last5_65_ecoli_mean_representations.csv')
X_new = new_data.iloc[:, 1:-1].values  # make sure data do not habve type column

#  Load the standardizer and perform standardization
with open('/media/zzh/data/AMP/LSTMregrssion/LSTM_model/ecoliscaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
X_new = scaler.transform(X_new)

# exchange to tensor
X_new = torch.tensor(X_new, dtype=torch.float32)

# Create data loade
new_dataset = TensorDataset(X_new.unsqueeze(1))
new_loader = DataLoader(new_dataset, batch_size=64, shuffle=False)

# Load best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size=X_new.size(1), hidden_size=128, num_layers=2, output_size=1, dropout_rate=0.7).to(device)
model.load_state_dict(torch.load('/media/zzh/data/AMP/Transformerregression/tools/ecoli_best_model_checkpoint.pth'))
model.eval()

# Make predictions
new_predictions = []
with torch.no_grad():
    for X_batch in new_loader:
        X_batch = X_batch[0].to(device)
        outputs = model(X_batch)
        new_predictions.extend(outputs.squeeze().cpu().numpy())

# Output prediction results
new_predictions = pd.DataFrame(new_predictions, columns=['Predicted Values'])
# print(new_predictions)
df_info  = pd.DataFrame()
df_info['Sequence'] = new_data.iloc[:,0]
df_merged = pd.concat([df_info,new_predictions],axis=1)

# # If you need to save the prediction results
df_merged.to_csv('/media/zzh/data/AMP/LSTMregrssion/results/updatedecolipredictions.csv', index=False)

# # Plot the predicted values
# plt.figure(figsize=(20, 7))

# # Display the first 50 samples
# n_samples_to_plot = 50
# plt.plot(new_predictions[:n_samples_to_plot], label='Predicted Values', alpha=0.7)

# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.legend()
# plt.title('Predicted Values on New Data')
# plt.show()