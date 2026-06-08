import os
import torch
from torch import nn, optim
import torch.ao.quantization
from torch.ao.quantization import quantize_dynamic
#na linuxie mozna odkomentowac
#from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight, int4_weight_only

from data.CifarData import getCifarDataLoaders
from data.DigitsData import getDigitsDataLoaders
from data.RegressionData import getRegressionDataLoader
from data.WineData import getWineDataLoaders

from models.CNN import CNN
from models.MLPbig import MLPbig
from models.MLPsmall import MLPsmall
from models.Regressor import Regressor

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import matplotlib.pyplot as plt



def train_model(model, train_loader, model_name, is_classifier=True, quantization=None, is_dynamic=True, quantization_level="f16", epochs=10, save_dir="./saved_models"):
    print(f"\n========== Training of model {model_name} started ==========")
    
    os.makedirs(save_dir, exist_ok=True)
    
    model.train()
    criterion = nn.CrossEntropyLoss() if is_classifier else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) if is_classifier else optim.Adam(model.parameters(), lr=0.01)

    if quantization == "qat":
        print("QAT preparation")
        torch.backends.quantized.engine = 'fbgemm' #dla linux: 'qnnpack'
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm') #dla linux: 'qnnpack'
        torch.ao.quantization.prepare_qat(model, inplace=True)

    #training
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

    if quantization == "qat":
        print("conversion of QAT model to INT8...")
        model.eval()
        torch.ao.quantization.convert(model, inplace=True)
        file_suffix = "qat_int8"
    # PTQ
    else:
        model.eval() 
        if quantization == "ptq":
            if quantization_level == "f16":
                print("conversion to Float16")
                model.half()
                file_suffix = "ptq_f16"
                
            elif quantization_level == "i8":
                if is_dynamic:
                    print("PTQ INT8 dynamic")
                    #wersja 1
                    #quantize_(model, int8_dynamic_activation_int8_weight())   #mi na windowsie nie dziala :(
                    #file_suffix = "ptq_dynamic_i8"
                    #wersja 2
                    model = quantize_dynamic(
                        model,
                        qconfig_spec={nn.Linear, nn.LSTM, nn.GRU},
                        dtype=torch.qint8
                    )
                    file_suffix = "ptq_dynamic_i8"
                else:
                    print("PTQ INT8 static")
                    torch.backends.quantized.engine = 'fbgemm' #dla linux: 'qnnpack'
                    model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm') #dla linux: 'qnnpack'
                    torch.ao.quantization.prepare(model, inplace=True)
                    
                    print("calibration for static PTQ")
                    with torch.no_grad():
                        for i, (X_batch, _) in enumerate(train_loader):
                            model(X_batch)
                            if i > 50: break 
                            
                    torch.ao.quantization.convert(model, inplace=True)
                    file_suffix = "ptq_static_i8"
                    
            elif quantization_level == "i4":
                #na linuxie mozna odkomentowac
                #print("PTQ INT4 Weight-Only")
                #quantize_(model, int4_weight_only())
                #file_suffix = "ptq_i4"
                pass
        else:
            file_suffix = "baseline_fp32"

    save_path = os.path.join(save_dir, f"{model_name}_{file_suffix}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"model saved in: {save_path}")

    return model

def test_model(model, test_loader, model_name, is_classifier=True, quantization=None, quantization_level="f16"):
    print(f"\n========== Testing of model {model_name} started ==========")
    
    # WYMAGANE: Ustawienie backendu dla warstw INT8, jeśli testujemy na Windows/Intel
    if quantization == "qat" or (quantization == "ptq" and quantization_level == "i8"):
        torch.backends.quantized.engine = 'fbgemm' # dla Linuxa: 'qnnpack'

    model.eval()
    
    criterion = nn.CrossEntropyLoss() if is_classifier else nn.MSELoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Listy do przechowywania wszystkich etykiet i przewidywań (potrzebne do scikit-learn)
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            if quantization == "ptq" and quantization_level == "f16":
                X_batch = X_batch.half()
                
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
            
            if is_classifier:
                _, predicted = torch.max(predictions.data, 1)
                
                # Konwersja na CPU i numpy, a następnie dodanie do list zbiorczych
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    
    if is_classifier:
        
        # Obliczanie metryk za pomocą scikit-learn
        # zero_division=0 zapobiega błędom/ostrzeżeniom, gdy jakaś klasa nie została ani razu przewidziana
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision (macro): {precision:.4f}")
        print(f"Test Recall (macro): {recall:.4f}")
        print(f"Test F1 Score (macro): {f1:.4f}")
        
        # Zwracamy poszerzony zestaw metryk dla klasyfikatora
        return avg_loss, accuracy, precision, recall, f1
    else:
        return avg_loss



def train_models():

    # 1. Load Data
    print("Loading datasets...")
    cnn_train_loader, cnn_test_loader = getCifarDataLoaders()
    wine_train_loader, wine_test_loader = getWineDataLoaders()
    digits_train_loader, digits_test_loader = getDigitsDataLoaders()
    regressor_loader = getRegressionDataLoader()
        
    models_setup = [
        (MLPsmall, wine_train_loader, "Wine_MLP_(Small)", True), 
        (CNN, cnn_train_loader, "CIFAR-10_CNN", True), 
        (Regressor, regressor_loader, "Trigonometric_Regressor", False), 
        (MLPbig, digits_train_loader, "Digits_MLP_(Big)", True)
    ]

    #(quantization, is_dynamic, level)
    quant_variants = [
        (None, None, None),     # no quantization
        ("ptq", None, "f16"),   # PTQ to f16
        ("ptq", True, "i8"),    # PTQ to INT8 with dynamic quantization
        #poniższe typy kwantyzacji na razie mi się nie udało ogarnąć żeby działało testowanie. Potem może naprawię ale jakaś gruba sprawa
        #("ptq", False, "i8"),   # PTQ to INT8 with static quantization 
        #("qat", None, None),    # QAT only to INT8
        #na linuxie mozna odkomentowac
        #("ptq", None, "i4"),    # PTQ to INT4 is weights-only
    ]

    for setup in models_setup:
        ModelClass, loader, name, is_cls = setup
        
        # We will track labels (X-axis) and the metric we want to plot (Y-axis)
        plot_labels = []
        plot_metrics = [[], [], [], []]
        
        for variant in quant_variants:
            qtz, dyn, lvl = variant
            fresh_model = ModelClass()
            
            if ModelClass == Regressor:
                epochs = 150
            else:
                epochs = 10 
                
            train_model(
                model=fresh_model, 
                train_loader=loader, 
                model_name=name, 
                is_classifier=is_cls, 
                quantization=qtz, 
                is_dynamic=dyn, 
                quantization_level=lvl, 
                epochs=epochs
            )
            
            # Formulate a clean readable name for the plot
            variant_name = f"{qtz or 'baseline'}_{lvl or 'fp32'}"
            plot_labels.append(variant_name)
            
            # --- FIX: Safe unpacking depending on model type ---
            if is_cls:
                avg_loss, accuracy, precision, recall, f1 = test_model(
                    model=fresh_model, 
                    test_loader=loader, # Note: you might want to use test_loader here instead of train loader!
                    model_name=name, 
                    is_classifier=is_cls,
                    quantization=qtz,
                    quantization_level=lvl
                )
                plot_metrics[0].append(accuracy)
                plot_metrics[1].append(precision)
                plot_metrics[2].append(recall)
                plot_metrics[3].append(accuracy)
            else:
                avg_loss = test_model(
                    model=fresh_model, 
                    test_loader=loader, 
                    model_name=name, 
                    is_classifier=is_cls,
                    quantization=qtz,
                    quantization_level=lvl
                )
                plot_metrics[0].append(avg_loss) # Tracking loss for Regressor instead

        # --- FIX: Plotting logic shifted outside the variant loop ---
        
        
        metrics = [('Accuracy', plot_metrics[0]), ('Precision', plot_metrics[1]), ('Recall', plot_metrics[2]), ('F1 Score', plot_metrics[3])]
        
        if is_cls:
            for metric in metrics:
                plt.figure(figsize=(8, 4))
                plt.bar(plot_labels, metric[1], color='skyblue', edgecolor='black')
                plt.ylabel(metric[0])
                plt.title(f'{metric[0]} w zależności od stopnia kwantyzacji - {name}')
                plt.ylim(0, 1.05)
                plt.savefig(f'{name}_{metric[0]}.png')
                plt.xlabel('Typ kwantyzacji')
                plt.tight_layout()
                #plt.show()
        else:
            plt.figure(figsize=(8, 4))
            plt.bar(plot_labels, metrics[0][1], color='skyblue', edgecolor='black')
            plt.ylabel('Loss function')
            plt.title(f'Loss function w zależności od stopnia kwantyzacji - {name}')
            plt.ylim(0, 2)
            plt.savefig(f'{name}_loss.png')
            plt.xlabel('Typ kwantyzacji')
            plt.tight_layout()
            #plt.show()
        




train_models()