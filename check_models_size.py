import os
import torch

import matplotlib.pyplot as plt

def check_models_size(sciezka_folderu: str = "./saved_models"):
    """
    Przeszukuje wskazany folder i wypisuje rozmiar oraz liczbę parametrów
    dla każdego znalezionego pliku .pth.
    """
    # Sprawdzenie, czy folder w ogóle istnieje
    if not os.path.exists(sciezka_folderu):
        print(f"Błąd: Folder '{sciezka_folderu}' nie istnieje.")
        return

    # Pobranie listy plików z rozszerzeniem .pth
    pliki_pth = [f for f in os.listdir(sciezka_folderu) if f.endswith('.pth')]

    if not pliki_pth:
        print(f"W folderze '{sciezka_folderu}' nie znaleziono żadnych plików .pth.")
        return

    print(f"Znaleziono {len(pliki_pth)} plików .pth w folderze '{sciezka_folderu}':")
    print("=" * 60)
    
    sizes = []

    for plik in pliki_pth:
        sciezka_pliku = os.path.join(sciezka_folderu, plik)
        print(f"\nModel: {plik}")
        
        # 1. Rozmiar na dysku
        rozmiar_bajty = os.path.getsize(sciezka_pliku)
        rozmiar_mb = rozmiar_bajty / (1024 * 1024)
        print(f"   Rozmiar na dysku: {rozmiar_mb:.4f} MB")
        sizes.append(rozmiar_mb)
            
    print("\n" + "=" * 60 + "\nSkonczone!")
    plt.bar(pliki_pth, sizes, color='salmon', edgecolor='black')
    plt.xticks(rotation=90)
    plt.ylabel('Rozmiar w MB')
    plt.title('Rozmiary modelu w różnych stopniach kwantyzacji')
    plt.savefig('sizes.png')
    plt.show()


check_models_size("./saved_models")