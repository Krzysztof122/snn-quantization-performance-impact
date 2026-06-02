import os
import torch
from torch import nn, optim
import torch.ao.quantization

import copy
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig

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


def quantize_dynamic_model(model, type=torch.qint8, output_path=None):
    # 1. Przełączenie modelu w tryb ewaluacji (bardzo ważne dla warstw typu Dropout czy BatchNorm)
    model.eval()

    print("Rozpoczęcie kwantyzacji dynamicznej obiektu modelu...")
    
    # 2. Wykonanie kwantyzacji dynamicznej
    # Konwertuje warstwy Linear i rekurencyjne na format INT8 (qint8)
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        qconfig_spec={torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
        dtype=type
    )
    
    print("Kwantyzacja zakończona sukcesem!")

    # 3. Opcjonalny zapis na dysk
    if output_path:
        torch.save(quantized_model.state_dict(), output_path)
        print(f"Skwantyzowane wagi zostały zapisane w: {output_path}")
        
    return quantized_model

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
    
    models = [(small_mlp, wine_train_loader, "Wine MLP (Small)"), (cnn, cnn_train_loader, "CIFAR-10 CNN"), (regressor, regressor_loader, "Trigonometric Regressor"), (big_mlp, digits_train_loader, "Digits MLP (Big)")]
    quant_types = [torch.qint8, torch.float16]
    
    for m in models:
        train_classifier(m[0], m[1], m[2], epochs = 5)
        torch.save(m[0].state_dict(), f"saved_models/{m[2]}.pth")
        for q in quant_types:
            quantize_dynamic_model(m[0], q, f"saved_models/{m[2]}_quantized.pth")   

    

train_models()