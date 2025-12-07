import sys, os
sys.path.insert(0, 'model_training')
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

print("\n" + "="*70)
print("TRAINING NEURAL NETWORK")
print("="*70)

spark = SparkSession.builder.appName("Train-NN").config("spark.driver.memory", "4g").getOrCreate()
df = spark.read.parquet('data/processed/ml_ready_data.parquet')
malicious_df = df.filter(col('is_malicious') == 1)
print(f"Malicious samples: {malicious_df.count():,}")

pandas_df = malicious_df.select('features', 'malware_family').toPandas()
X = np.array([row.toArray() for row in pandas_df['features']])
y_labels = pandas_df['malware_family'].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)
num_classes = len(label_encoder.classes_)
input_dim = X.shape[1]
print(f"Shape: {X.shape}, Classes: {num_classes}, Labels: {label_encoder.classes_}")

class MalwareClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, num_classes)
        )
    def forward(self, x):
        return self.network(x)

model = MalwareClassifier(input_dim, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)
train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=256, shuffle=True)

print("Training Neural Network (100 epochs)...")
model.train()
for epoch in range(100):
    total_loss, correct, total = 0, 0, 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/100, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

os.makedirs('models/neural_net_multiclass', exist_ok=True)
torch.save({'model_state_dict': model.state_dict(), 'input_dim': input_dim, 'num_classes': num_classes, 'label_encoder': label_encoder}, 'models/neural_net_multiclass/nn_model.pth')
spark.stop()
print("\n✅ NEURAL NETWORK TRAINING COMPLETE!")
print(f"Model saved to: models/neural_net_multiclass/nn_model.pth")
