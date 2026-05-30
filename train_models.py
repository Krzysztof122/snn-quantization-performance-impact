import os
import torch
from torch import nn, optim

from data.CifarData import getCifarDataLoaders
from data.DigitsData import getDigitsDataLoaders
from data.RegressionData import getRegressionDataLoader
from data.WineData import getWineDataLoaders

from models.CNN import CNN
from models.MLPbig import MLPbig
from models.MLPsmall import MLPsmall
from models.Regressor import Regressor


def train_classifier(model, train_loader, model_name, epochs=5):
    print(f"\nStarting Training for: {model_name}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


def train_regressor(model, train_loader, model_name, epochs=150):
    print(f"\nStarting Training for: {model_name}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 25 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


def train_models():
    # 1. Load Data
    print("Loading datasets...")
    cnn_train_loader, cnn_test_loader = getCifarDataLoaders()
    wine_train_loader, wine_test_loader = getWineDataLoaders()
    digits_train_loader, digits_test_loader = getDigitsDataLoaders()
    regressor_loader = getRegressionDataLoader()

    # 2. Instantiate Models
    print("Initializing models...")
    cnn = CNN()
    small_mlp = MLPsmall()
    big_mlp = MLPbig()
    regressor = Regressor()
    
    train_classifier(small_mlp, wine_train_loader, "Wine MLP (Small)", epochs=10)
    train_classifier(big_mlp, digits_train_loader, "Digits MLP (Big)", epochs=5)
    train_classifier(cnn, cnn_train_loader, "CIFAR-10 CNN", epochs=5)
    train_regressor(regressor, regressor_loader, "Trigonometric Regressor", epochs=200)
    
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(small_mlp.state_dict(), f"{save_dir}/wine_mlp.pth")
    torch.save(big_mlp.state_dict(), f"{save_dir}/digits_mlp.pth")
    torch.save(cnn.state_dict(), f"{save_dir}/cifar_cnn.pth")
    torch.save(regressor.state_dict(), f"{save_dir}/trig_regressor.pth")
    

if __name__ == "__main__":
    train_models()