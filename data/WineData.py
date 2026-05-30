import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class WineDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def getWineDataLoaders():
    data = load_wine()  #Bunch object
    
    X_raw = data.data   #2d numpy array
    y_raw = data.target #1d numpy array

    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=47)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = WineDataset(X_train_scaled, y_train)
    test_dataset = WineDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader





if __name__ == "__main__":
    train_loader, test_loader = getWineDataLoaders()
    
    for features, labels in train_loader:
        print(f"Wine - Features batch shape: {features.shape} [Batch, Features]")
        print(f"Wine - Labels batch shape:   {labels.shape} [Batch]")
        
        print("\n--- Sanity Checks ---")
        print(f"1. First 5 Labels in Batch: {labels[:5].tolist()}")
        
        print("\n2. First row of scaled features (Should be mostly between -3 and 3):")
        print([round(val, 2) for val in features[0].tolist()])
        
        break