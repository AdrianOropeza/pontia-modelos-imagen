---
license: creativeml-openrail-m
base_model: CompVis/stable-diffusion-v1-4
tags:
  - stable-diffusion
  - lora
  - text-to-image
  - diffusers
datasets:
  - gigant/oldbookillustrations
---

# LoRA: ilustraciones de libros antiguos sobre Stable Diffusion v1-4

Adaptador LoRA entrenado sobre `CompVis/stable-diffusion-v1-4` con el dataset
[`gigant/oldbookillustrations`](https://huggingface.co/datasets/gigant/oldbookillustrations),
compuesto por ilustraciones grabadas de libros publicados entre 1800 y 1900.

## Configuración de entrenamiento

| Parámetro | Valor |
|---|---|
| Rango (r) | 8 |
| Alpha | 8 |
| Módulos | to_q, to_k, to_v, to_out.0 |
| Muestras | 830 |
| Épocas | 5 |
| Batch size | 3 |
| Learning rate | 1e-4 |
| Resolución | 512x512 |
| Precisión | fp32 |
| Parámetros entrenables | 1.594.368 de 861.115.332 (0,19%) |

Las muestras se filtraron por caption en inglés de al menos 15 caracteres, usando las columnas
`1600px` e `info_alt` del dataset. Las imágenes se redimensionaron por el lado corto y se recortaron
al centro para preservar la relación de aspecto original.

## Uso

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")
pipe.load_lora_weights("TU-USUARIO/sd-v1-4-oldbook-illustrations-lora")

image = pipe("a sailing ship at sea", num_inference_steps=50, guidance_scale=7.5).images[0]
```

## Intensidad del adaptador

Con el factor de escala nominal (`alpha / r` = 1.0) el adaptador produce un cambio de estilo visible
pero parcial: la composición gana definición lineal mientras conserva la paleta del modelo base. La
estética de grabado del dataset emerge por completo escalando el adaptador por encima de 1, sin
degradación de la imagen:

```python
pipe.load_lora_weights("TU-USUARIO/sd-v1-4-oldbook-illustrations-lora")
image = pipe(
    "a sailing ship at sea",
    num_inference_steps=50,
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": 2.5},
).images[0]
```

La transición no es proporcional: entre 0 y 1,5 los cambios son graduales y se mantiene la paleta
del modelo base, mientras que a partir de 2,0 el estilo aprendido se impone y la imagen pasa a ser
un grabado en blanco y negro. Un entrenamiento equivalente con `lora_alpha=32` habría producido este
comportamiento a escala 1.0.

## Licencia

El modelo base se distribuye bajo CreativeML Open RAIL-M. El dataset de entrenamiento está bajo
licencia CC BY-NC 4.0, por lo que este adaptador se publica para uso no comercial.
