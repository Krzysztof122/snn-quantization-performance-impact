import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def getDigitsDataLoaders(batch_size=32):
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: torch.flatten(x)) 
    ])

    train_dataset = datasets.MNIST(
        root="./data/data",
        train=True,
        download=True,
        transform=transform_pipeline
    )

    test_dataset = datasets.MNIST(
        root="./data/data",
        train=False,
        download=True,
        transform=transform_pipeline
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Digits DataLoaders are ready!")
    return train_loader, test_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_loader, test_loader = getDigitsDataLoaders()

    for images, labels in train_loader:
        print(f"Digits - Image batch shape: {images.shape} [Batch, Flattened Pixels]")
        print(f"Digits - Label batch shape: {labels.shape} [Batch]")
        
        fig, axes = plt.subplots(1, 5, figsize=(12, 3))
        for i in range(5):
            img = images[i].view(28, 28).numpy()
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Label: {labels[i].item()}")
            axes[i].axis('off')
            
        plt.suptitle("MNIST Digits (Torchvision)")
        plt.show()
        break