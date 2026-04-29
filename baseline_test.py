# PUNTO DE CONTROL
 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Tokenizer (interprete / traducctor)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Model (cerebro / calculos)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map = "auto",         # Usa la mejor forma de procesar
    torch_dtype = torch.float16  # Aligera el modelo a 16 bits sin perder mucha inteligencia
)

# Pipeline (Une tokenizer y model para generar texto)
pip = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Preparar el mensaje 
messages = [
    {"role": "system", "content": "You are a boring and formal marketing assistant."},
    {"role": "user", "content": "Write a short marketing post for: A simple white cotton t-shirt."}
]

# Aplicar la plantilla del modelo - Chat Template
formatted_prompt = pip.tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

print(" ")
print("--- ASÍ VE EL MODELO NUESTRO MENSAJE ---")
print(formatted_prompt)
print(" ")
print(" ")

# Generar una respuesta a la petición
print("Generando una respuesta...")

outputs = pip(
    formatted_prompt, 
    max_new_tokens=150,  # Límite de palabras nuevas a generar
    do_sample=True,      # Le damos permiso para no elegir siempre la palabra más obvia
    temperature=0.7      # nivel de creatividad (0.1=predecible)
)

# Limpiando el resultado
# El modelo nos devuelve todo el bloque de texto (nuestro prompt + su respuesta).
# Lo partimos justo donde empieza su parte (<|assistant|>) y nos quedamos con el final ([-1]).
response = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()

print("Respuesta baseline (aburrida):")
print(response)
print("-" * 50)