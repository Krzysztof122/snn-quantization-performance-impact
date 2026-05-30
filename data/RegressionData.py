import torch
from torch.utils.data import TensorDataset, DataLoader


def getRegressionDataLoader():
    X = torch.linspace(-10, 10, 300).view(-1, 1) #view is like reshape, but for tensors
    y_true = torch.sin(X) + 1.45*torch.sin(3.4*X) - 2.65*torch.sin(X / 1.4) + 0.02*X**2
    noise = torch.randn(X.size()) * 0.3
    y = y_true + noise

    dataset = TensorDataset(X, y)
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return loader



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    loader = getRegressionDataLoader()
    
    for X_batch, y_batch in loader:
        print(f"Regression - X batch shape: {X_batch.shape} [Batch, Features]")
        print(f"Regression - y batch shape: {y_batch.shape} [Batch, 1]")
        break
    
    all_X, all_y = loader.dataset.tensors
    
    X_sorted, _ = torch.sort(all_X, dim=0) 
    y_true = torch.sin(X_sorted) + 1.45*torch.sin(3.4*X_sorted) - 2.65*torch.sin(X_sorted / 1.4) + 0.02*X_sorted**2
    
    plt.figure(figsize=(10, 5))
    plt.scatter(all_X.numpy(), all_y.numpy(), s=15, alpha=0.6, label='Noisy DataLoader Points')
    plt.plot(X_sorted.numpy(), y_true.numpy(), color='red', linewidth=2, label='True Mathematical Function')
    plt.title("Regression Dataset (Stress Test)")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()