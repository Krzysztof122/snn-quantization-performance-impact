from torchvision import datasets, transforms
from torch.utils.data import DataLoader

    
def getCifarDataLoaders(batch_size=32):

    transform_sequence = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #(-1, 1)
    ])

    train_dataset = datasets.CIFAR10(
        root="./data/data",
        train=True,
        download=True,
        transform=transform_sequence
    )

    test_dataset = datasets.CIFAR10(
        root="./data/data",
        train=False,
        download=True,
        transform=transform_sequence
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    train_loader, test_loader = getCifarDataLoaders()
    cifar_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for images, labels in train_loader:
        print(f"CIFAR-10 - Image batch shape: {images.shape} [Batch, Channels, Height, Width]")
        print(f"CIFAR-10 - Label batch shape: {labels.shape} [Batch]")
        
        fig, axes = plt.subplots(1, 5, figsize=(12, 3))
        for i in range(5):
            img = images[i].numpy() / 2 + 0.5  
            img = np.transpose(img, (1, 2, 0))
            axes[i].imshow(img)
            axes[i].set_title(cifar_classes[labels[i].item()])
            axes[i].axis('off')
            
        plt.suptitle("CIFAR-10 Normalized Tensors")
        plt.show()
        break
