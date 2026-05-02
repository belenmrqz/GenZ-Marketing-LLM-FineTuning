from datasets import load_dataset
import os

print("PREPARACIÓN DEL DATASET\n" + "-"*40)

# 1. Load the local JSONL file
print("En busqueda del archivo 'genz_dataset.jsonl'...")
try:
    dataset = load_dataset("json", data_files="genz_dataset.jsonl", split="train")
    print(f"¡Dataset cargado! Tenemos {len(dataset)} ejemplos")
except Exception as e:
    print(f"Hubo un error al cargar el archivo: {e}")
    exit()

# 2. Split the dataset (Train / Test)
# Usaremos un 80% para entrenar y un 20% para el examen final
print("\nDividiendo los datos en Train y Test...")
dataset_split = dataset.train_test_split(test_size=0.2)

print(f"Entrenamiento (Train): {len(dataset_split['train'])} ejemplos.")
print(f"Pruebas (Test): {len(dataset_split['test'])} ejemplos.")

# 3. Push to Hugging Face Hub
hf_username = "Belenmrqz" 
dataset_name = f"{hf_username}/genz-marketing-prompts"

print(f"\nSubiendo el dataset a la nube de Hugging Face: {dataset_name}...")

# Subir dataset a internet
dataset_split.push_to_hub(dataset_name)

print("\n¡Misión cumplida! el dataset ya es público en Hugging Face.")
print("-" * 40)