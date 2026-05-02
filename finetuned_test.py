import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. RUTAS DE LOS MODELOS
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./genz-adapter" 

print("Cargando el modelo base")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("Conectando el adaptador Gen-Z")
# Se fusiona el modelo base con tu entrenamiento
model = PeftModel.from_pretrained(base_model, adapter_path)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 2. EL PROMPT DE PRUEBA
# Usamos exactamente la misma estructura que en nuestro dataset de entrenamiento
messages = [
    {"role": "system", "content": "You are a Gen-Z marketing assistant. Write engaging, trendy, and emoji-filled social media posts."},
    {"role": "user", "content": "Write a short marketing post for: Comfortable pink sneakers for walking."}
]

# Convertimos los mensajes al formato que el modelo entiende
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Pensando la respuesta...")

# 3. GENERACIÓN DE TEXTO
outputs = model.generate(
    **inputs,
    max_new_tokens=100, # Límite de palabras generadas
    temperature=0.7,    # Nivel de creatividad 0.7 
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# 4. TRADUCCIÓN Y RESULTADO
# Decodificamos los números devueltos por el modelo a texto legible
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "="*50)
print("RESULTADO DEL MODELO:")
print("="*50)
# Recortamos el prompt inicial para mostrar solo la respuesta del asistente
final_answer = response.split("assistant\n")[-1]
print(final_answer.strip())