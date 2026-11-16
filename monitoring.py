"""
monitoring.py
Sistema de monitorización del modelo ViT de predicción de siniestros.
Integra métricas, logs y trazas con MLflow.
"""

import os
import time
import logging
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.pytorch
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    v_gpu_disponible = True
except ImportError:
    v_gpu_disponible = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("/content/monitoring.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True
)
v_logger = logging.getLogger(__name__)

v_slo = {
    "accuracy_f1_min":        0.88,
    "confianza_promedio_min": 0.85,
    "latencia_max_ms":        2000,
    "uso_hardware_max":       0.85,
}


class DatasetValidacion(torch.utils.data.Dataset):
    def __init__(self, v_directorio_raiz, v_image_processor, v_clases_disponibles):
        self.v_image_processor = v_image_processor
        self.v_clases_disponibles = v_clases_disponibles
        self.v_muestras = []
        import xml.etree.ElementTree as ET

        if os.path.exists(v_directorio_raiz):
            for v_root, v_dirs, v_files_in_dir in os.walk(v_directorio_raiz):
                for v_file in v_files_in_dir:
                    if v_file.endswith(('.jpg', '.jpeg', '.png')):
                        v_img_path = os.path.join(v_root, v_file)
                        v_xml_path = os.path.splitext(v_img_path)[0] + ".xml"
                        v_clase_idx = 0

                        if os.path.exists(v_xml_path):
                            try:
                                v_tree = ET.parse(v_xml_path)
                                v_obj = v_tree.getroot().find('object')
                                if v_obj is not None:
                                    v_nombre = v_obj.find('name').text.strip()
                                    if v_nombre in self.v_clases_disponibles:
                                        v_clase_idx = self.v_clases_disponibles.index(v_nombre)
                            except Exception:
                                pass
                        self.v_muestras.append((v_img_path, v_clase_idx))

    def __len__(self):
        return len(self.v_muestras)

    def __getitem__(self, v_idx):
        v_img_path, v_label = self.v_muestras[v_idx]
        v_image = Image.open(v_img_path).convert("RGB")
        v_inputs = self.v_image_processor(images=v_image, return_tensors="pt")
        return v_inputs["pixel_values"].squeeze(0), torch.tensor(v_label, dtype=torch.long)


def obtener_uso_hardware():
    v_uso_cpu = psutil.cpu_percent(interval=1) / 100.0
    v_uso_gpu = 0.0
    if v_gpu_disponible and torch.cuda.is_available():
        try:
            v_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            v_info = pynvml.nvmlDeviceGetMemoryInfo(v_handle)
            v_uso_gpu = v_info.used / v_info.total
        except Exception:
            pass
    v_dispositivo_actual = "GPU" if torch.cuda.is_available() else "CPU"
    v_uso_actual = v_uso_gpu if torch.cuda.is_available() else v_uso_cpu
    return v_uso_actual, v_dispositivo_actual


def evaluar_modelo(v_model, v_dataloader, v_device, v_clase_a_evaluar_idx=0):
    v_model.eval()
    v_todas_predicciones = []
    v_todas_etiquetas = []
    v_todas_confianzas = []

    v_inicio = time.time()

    with torch.no_grad():
        for v_pixel_values, v_labels in v_dataloader:
            v_pixel_values = v_pixel_values.to(v_device)
            v_outputs = v_model(pixel_values=v_pixel_values)

            v_probs = torch.nn.functional.softmax(v_outputs.logits, dim=-1)
            v_confianzas, v_preds = torch.max(v_probs, dim=-1)

            v_todas_predicciones.extend(v_preds.cpu().numpy())
            v_todas_etiquetas.extend(v_labels.numpy())
            v_todas_confianzas.extend(v_confianzas.cpu().numpy())

    v_tiempo_total_seg = time.time() - v_inicio

    v_uso_hardware, v_disp = obtener_uso_hardware()

    v_acc = accuracy_score(v_todas_etiquetas, v_todas_predicciones)
    v_f1 = f1_score(v_todas_etiquetas, v_todas_predicciones, average='weighted', zero_division=0)
    v_confianza_prom = np.mean(v_todas_confianzas)

    v_num_imagenes = len(v_todas_etiquetas)
    v_latencia_ms_por_imagen = (v_tiempo_total_seg / v_num_imagenes) * 1000

    v_metricas = {
        "accuracy": round(v_acc, 4),
        "f1_score": round(v_f1, 4),
        "confianza_promedio": round(v_confianza_prom, 4),
        "latencia_ms": round(v_latencia_ms_por_imagen, 2),
        "uso_hardware": round(v_uso_hardware, 4),
        "dispositivo": v_disp,
        "n_muestras": v_num_imagenes
    }
    return v_metricas


def verificar_slos(v_metricas):
    v_violaciones = []

    if v_metricas["accuracy"] < v_slo["accuracy_f1_min"] or v_metricas["f1_score"] < v_slo["accuracy_f1_min"]:
        v_violaciones.append({
            "evento": "Accuracy / F1-Score menor a 88%",
            "accion": "Reentrenar modelo ViT con nuevos casos de siniestros validados",
            "prioridad": "Alta"
        })

    if v_metricas["confianza_promedio"] < v_slo["confianza_promedio_min"]:
        v_violaciones.append({
            "evento": "Confianza Promedio menor a 85%",
            "accion": "Enviar expediente a revisión manual por un ajustador y revisar calidad de imágenes.",
            "prioridad": "Media"
        })

    if v_metricas["latencia_ms"] > v_slo["latencia_max_ms"]:
        v_violaciones.append({
            "evento": "Latencia mayor a 2000 ms",
            "accion": "Reiniciar servicio contenedor_siniestros.",
            "prioridad": "Alta"
        })

    if v_metricas["uso_hardware"] > v_slo["uso_hardware_max"]:
        v_violaciones.append({
            "evento": f"Uso de {v_metricas['dispositivo']} mayor al 85%",
            "accion": "Escalar recursos del servidor.",
            "prioridad": "Media"
        })

    return v_violaciones


def run_monitoreo(v_ruta_modelo="./modelo_vit_siniestros",
                  v_ruta_datos="/content/imagenes_entrenamiento"):
    v_logger.info(f"Iniciando monitoreo del modelo en: {v_ruta_modelo}")

    try:
        v_processor = AutoImageProcessor.from_pretrained(v_ruta_modelo, local_files_only=True)
        v_model = AutoModelForImageClassification.from_pretrained(v_ruta_modelo, local_files_only=True)
    except Exception as v_e:
        v_logger.error(f"Error cargando el modelo: {v_e}")
        return

    v_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v_model.to(v_device)

    v_clases_disponibles = [v_model.config.id2label[i] for i in range(len(v_model.config.id2label))]

    v_dataset = DatasetValidacion(v_ruta_datos, v_processor, v_clases_disponibles)
    if len(v_dataset) == 0:
        v_logger.error("No se encontraron imágenes para monitorear.")
        return

    v_dataloader = DataLoader(v_dataset, batch_size=8, shuffle=False)

    v_logger.info("Evaluando modelo...")
    v_metricas = evaluar_modelo(v_model, v_dataloader, v_device)
    v_violaciones = verificar_slos(v_metricas)

    mlflow.set_experiment("monitoreo-siniestros-vit")

    with mlflow.start_run(run_name="monitoreo_produccion"):
        mlflow.set_tags({
            "modelo": "ViT-Siniestros",
            "dispositivo": str(v_device),
            "timestamp": datetime.now().isoformat()
        })

        v_metricas_mlflow = {k: v for k, v in v_metricas.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(v_metricas_mlflow)
        mlflow.log_metric("slo_violaciones", len(v_violaciones))

        if v_violaciones:
            mlflow.set_tag("status", "ALERTA")
            for v_v in v_violaciones:
                v_logger.error(f"🚨 ALERTA {v_v['prioridad']}: {v_v['evento']}")
                v_logger.error(f"   Acción requerida: {v_v['accion']}")
        else:
            mlflow.set_tag("status", "OK")
            v_logger.info("✅ Todos los SLOs cumplidos")

        print(f"\n{'='*60}")
        print(f"  Monitoreo de Modelo ViT - Siniestros")
        print(f"{'='*60}")
        for v_k, v_v in v_metricas.items():
            print(f"  {v_k:<25}: {v_v}")

        if v_violaciones:
            print(f"\n  🚨 ALERTAS DETECTADAS:")
            for v_v in v_violaciones:
                print(f"    - [{v_v['prioridad']}] {v_v['evento']}")
                print(f"      -> {v_v['accion']}")
        else:
            print(f"\n  ✅ Sistema operando normalmente.")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    run_monitoreo(
        v_ruta_modelo="/content/modelo_vit_siniestros",
        v_ruta_datos="/content/imagenes_entrenamiento"
    )
