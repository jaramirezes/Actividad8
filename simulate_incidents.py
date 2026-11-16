"""
simulate_incidents.py
Simulación de incidentes críticos para el modelo ViT de siniestros.
Ejecuta runbooks que SÍ reentrenan el modelo.
"""

import os
import mlflow
import mlflow.pytorch
import numpy as np
import time
import logging
import torch
from datetime import datetime
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, f1_score

import runbooks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("/content/incidents.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True
)
v_logger = logging.getLogger(__name__)

v_slo_accuracy_min  = 0.88
v_slo_f1_min        = 0.88
v_slo_confianza_min = 0.85
v_slo_latencia_ms   = 2000

v_ruta_modelo = "/content/modelo_vit_siniestros"
v_ruta_datos  = "/content/imagenes_entrenamiento"


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


def cargar_y_preparar():
    v_processor = AutoImageProcessor.from_pretrained(v_ruta_modelo, local_files_only=True)
    v_model = AutoModelForImageClassification.from_pretrained(v_ruta_modelo, local_files_only=True)
    v_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v_model.to(v_device)
    v_clases_disponibles = [v_model.config.id2label[i] for i in range(len(v_model.config.id2label))]
    v_dataset = DatasetValidacion(v_ruta_datos, v_processor, v_clases_disponibles)
    v_dataloader = DataLoader(v_dataset, batch_size=8, shuffle=False)
    return v_model, v_dataloader, v_device


def evaluar(v_model, v_dataloader, v_device, v_ruido_std=0.0):
    v_model.eval()
    v_todas_predicciones = []
    v_todas_etiquetas = []
    v_todas_confianzas = []

    v_t0 = time.time()
    with torch.no_grad():
        for v_pixel_values, v_labels in v_dataloader:
            if v_ruido_std > 0:
                v_pixel_values = v_pixel_values + torch.randn_like(v_pixel_values) * v_ruido_std
                v_pixel_values = torch.clamp(v_pixel_values, 0.0, 1.0)
            v_pixel_values = v_pixel_values.to(v_device)
            v_outputs = v_model(pixel_values=v_pixel_values)
            v_probs = torch.nn.functional.softmax(v_outputs.logits, dim=-1)
            v_confianzas, v_preds = torch.max(v_probs, dim=-1)
            v_todas_predicciones.extend(v_preds.cpu().numpy())
            v_todas_etiquetas.extend(v_labels.numpy())
            v_todas_confianzas.extend(v_confianzas.cpu().numpy())

    v_lat = (time.time() - v_t0) * 1000 / max(len(v_todas_etiquetas), 1)

    return {
        "accuracy":           round(accuracy_score(v_todas_etiquetas, v_todas_predicciones), 4),
        "f1_score":           round(f1_score(v_todas_etiquetas, v_todas_predicciones, average='weighted', zero_division=0), 4),
        "confianza_promedio": round(float(np.mean(v_todas_confianzas)), 4),
        "latencia_ms":        round(v_lat, 2),
        "n_muestras":         len(v_todas_etiquetas),
    }


# ── Incidente 1: Degradación del modelo ────────────────────────────────────
def incidente_degradacion():
    print("\n" + "=" * 60)
    print("  INCIDENTE 1: Degradación del modelo por data drift")
    print("=" * 60)

    mlflow.set_experiment("monitoreo-siniestros-vit")
    v_model, v_dataloader, v_device = cargar_y_preparar()

    with mlflow.start_run(run_name="incidente_degradacion_modelo"):
        mlflow.set_tags({
            "tipo_incidente": "degradacion_modelo",
            "causa":          "data_drift",
            "severidad":      "ALTA",
            "timestamp":      datetime.now().isoformat()
        })

        v_m_antes = evaluar(v_model, v_dataloader, v_device, v_ruido_std=0.0)
        v_logger.info(f"Métricas ANTES del incidente: {v_m_antes}")
        mlflow.log_metrics({f"antes_{k}": v for k, v in v_m_antes.items() if isinstance(v, (int, float))})

        v_m_durante = evaluar(v_model, v_dataloader, v_device, v_ruido_std=0.5)
        v_logger.warning(f"Métricas DURANTE incidente: {v_m_durante}")
        mlflow.log_metrics({f"durante_{k}": v for k, v in v_m_durante.items() if isinstance(v, (int, float))})

        v_resultado = None
        if (v_m_durante["accuracy"] < v_slo_accuracy_min
                or v_m_durante["f1_score"] < v_slo_f1_min
                or v_m_durante["confianza_promedio"] < v_slo_confianza_min):
            v_logger.error(
                f"SLO VIOLATION: Accuracy {v_m_durante['accuracy']} < {v_slo_accuracy_min} "
                f"o F1 {v_m_durante['f1_score']} < {v_slo_f1_min} "
                f"o Confianza {v_m_durante['confianza_promedio']} < {v_slo_confianza_min}"
            )
            mlflow.set_tag("slo_status", "VIOLATION")

            print("\n  → Ejecutando Runbook: Reentrenamiento REAL del modelo")
            v_resultado = runbooks.runbook_reentrenamiento(v_model, v_dataloader, v_device)
            mlflow.log_metrics({f"despues_{k}": v for k, v in v_resultado.items() if isinstance(v, (int, float))})
            mlflow.set_tag("accion", "reentrenamiento")
            mlflow.set_tag("resolucion", "OK" if v_resultado["accuracy"] >= v_slo_accuracy_min else "PENDIENTE")
        else:
            mlflow.set_tag("slo_status", "OK")
            v_resultado = v_m_durante

        print(f"\n  Antes   → Accuracy: {v_m_antes['accuracy']} | F1: {v_m_antes['f1_score']} | Confianza: {v_m_antes['confianza_promedio']}")
        print(f"  Durante → Accuracy: {v_m_durante['accuracy']} | F1: {v_m_durante['f1_score']} | Confianza: {v_m_durante['confianza_promedio']}")
        print(f"  Después → Accuracy: {v_resultado['accuracy']} | F1: {v_resultado['f1_score']} | Confianza: {v_resultado['confianza_promedio']}")

        # Retornamos el resultado para que el incidente 3 sepa si el modelo mejoró o no
        return v_resultado


# ── Incidente 2: Alta latencia ─────────────────────────────────────────────
def incidente_latencia():
    print("\n" + "=" * 60)
    print("  INCIDENTE 2: Incremento de latencia en predicciones")
    print("=" * 60)

    mlflow.set_experiment("monitoreo-siniestros-vit")
    v_model, v_dataloader, v_device = cargar_y_preparar()

    with mlflow.start_run(run_name="incidente_alta_latencia"):
        mlflow.set_tags({
            "tipo_incidente": "alta_latencia",
            "causa":          "carga_excesiva",
            "severidad":      "ALTA",
            "timestamp":      datetime.now().isoformat()
        })

        v_logger.warning("Simulando alta latencia en el servicio...")
        time.sleep(2.5)
        v_metricas = evaluar(v_model, v_dataloader, v_device)
        v_metricas["latencia_ms"] = round(v_metricas["latencia_ms"] + 2500, 2)

        mlflow.log_metrics({k: v for k, v in v_metricas.items() if isinstance(v, (int, float))})

        v_resultado = {"status": "NO_ACTION"}
        if v_metricas["latencia_ms"] > v_slo_latencia_ms:
            v_logger.error(f"SLO VIOLATION: Latencia {v_metricas['latencia_ms']}ms > {v_slo_latencia_ms}ms")
            mlflow.set_tag("slo_status", "VIOLATION")

            print("\n  → Ejecutando Runbook: Escalamiento del servicio")
            v_resultado = runbooks.runbook_escalamiento()
            mlflow.set_tag("accion", "escalamiento")
            mlflow.set_tag("resolucion", v_resultado["status"])
        else:
            mlflow.set_tag("slo_status", "OK")

        print(f"\n  Latencia detectada: {v_metricas['latencia_ms']:.1f}ms")
        print(f"  SLO máximo:         {v_slo_latencia_ms}ms")
        print(f"  Accuracy:           {v_metricas['accuracy']}")
        print(f"  Resolución:         {v_resultado['status']}")


# ── Incidente 3: Fallo Crítico (Rollback) ──────────────────────────────────
def incidente_rollback(v_metricas_reentrenamiento=None):
    print("\n" + "=" * 60)
    print("  INCIDENTE 3: Evaluación de Fallo Crítico")
    print("=" * 60)

    # Condición: Solo hacemos rollback si el reentrenamiento falló en alcanzar el SLO
    if v_metricas_reentrenamiento is not None and v_metricas_reentrenamiento["accuracy"] >= v_slo_accuracy_min:
        print("\n  ✅ El reentrenamiento del Incidente 1 fue exitoso.")
        print(f"     Accuracy actual: {v_metricas_reentrenamiento['accuracy']} (SLO: {v_slo_accuracy_min})")
        print("  No se requiere Rollback. El modelo sigue en producción.")
        return

    # Si no hay métricas o el modelo falló, disparamos el Rollback
    mlflow.set_experiment("monitoreo-siniestros-vit")

    with mlflow.start_run(run_name="incidente_rollback_critico"):
        mlflow.set_tags({
            "tipo_incidente": "fallo_critico",
            "causa":          "degradacion_severa",
            "severidad":      "CRÍTICA",
            "timestamp":      datetime.now().isoformat()
        })

        v_logger.warning("Fallo crítico detectado o reentrenamiento insuficiente. Iniciando Rollback...")

        print("\n  → Ejecutando Runbook: Rollback a versión estable")
        v_resultado = runbooks.runbook_rollback("v1.0")

        mlflow.set_tag("accion", "rollback")
        mlflow.set_tag("resolucion", v_resultado["status"])

        print(f"\n  Estado del servicio: {v_resultado['status']}")
        print(f"  Versión restaurada:  {v_resultado['version_activa']}")


def _volcar_logs():
    for v_nombre in list(logging.Logger.manager.loggerDict.keys()) + [None]:
        v_lg = logging.getLogger(v_nombre) if v_nombre else logging.root
        for v_h in list(v_lg.handlers):
            if isinstance(v_h, logging.FileHandler):
                try:
                    v_h.flush()
                    v_h.close()
                    v_lg.removeHandler(v_h)
                except Exception:
                    pass


if __name__ == "__main__":
    try:
        # 1. Corre incidente de degradación y guardamos el resultado
        v_metricas_inc1 = incidente_degradacion()

        # 2. Corre incidente de latencia
        incidente_latencia()

        # 3. Corre incidente de rollback (le pasamos las métricas del Inc 1)
        incidente_rollback(v_metricas_inc1)

        v_logger.info("✅ Simulación de incidentes completada.")
        print("\n✅ Simulación de incidentes completada.")
        print("   Ejecuta: mlflow ui  para ver los resultados")
    finally:
        _volcar_logs()
