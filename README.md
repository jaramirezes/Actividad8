# 🛡️ Monitoreo MLOps y Gobernanza Operativa: Clasificación Automatizada de Daños Vehiculares (ViT)

Este repositorio contiene la implementación del sistema de monitoreo continuo, observabilidad, simulación de incidentes y gobernanza operativa para un modelo de **Vision Transformer (ViT)** orientado a la clasificación automatizada de daños vehiculares en siniestros.

---

## 📁 Estructura del Repositorio

El proyecto sigue una arquitectura modular que separa la documentación ejecutiva del código fuente y los artefactos de telemetría:

```text
Actividad8/
│
├── docs/                               # Documentación y entregables de la actividad
│   ├── Actividad 8.docx                # Documento principal de entrega
│   ├── Evidencias.pptx                 # Presentación con capturas y logs
│   └── Informe Técnico.docx            # Informe técnico detallado 
│
├── fuentes/                            # Código fuente y scripts de observabilidad
│   ├── Actividad8.ipynb                # Notebook integral de desarrollo y pruebas
│   ├── monitoring.py                   # Motor de monitoreo, métricas y validación de SLOs
│   ├── runbooks.py                     # Playbooks automatizados de respuesta a incidentes
│   └── simulate_incidents.py           # Simulador de degradación (Data Drift, latencia, fallos)
│
├── dashboard_siniestros_real.html      # Dashboard interactivo de monitoreo operacional
├── incidents.log                       # Registro histórico de alertas e incidentes disparados
├── mlflow.db                           # Base de datos SQLite para trazabilidad en MLflow
├── monitoring.log                      # Logs detallados de ejecución del sistema de monitoreo
└── requirements.txt                    # Dependencias y librerías necesarias
```

---

## 🚀 Componentes Principales

### 1. Monitoreo y Observabilidad (`fuentes/monitoring.py`)
Evalúa en tiempo real el comportamiento del modelo ViT en producción mediante:
* **Métricas de Desempeño:** Seguimiento de precisión, confianza de predicción y tasas de error.
* **Detección de Anomalías:** Identificación temprana de *Data Drift* (cambios en la distribución de imágenes de entrada) y *Concept Drift*.
* **Cumplimiento de SLOs:** Validación continua de latencia de inferencia y disponibilidad del servicio.

### 2. Simulación de Incidentes (`fuentes/simulate_incidents.py`)
Permite estresar el pipeline MLOps inyectando escenarios controlados de fallo (imágenes corruptas, degradación de confianza, picos de latencia) para validar la resiliencia del sistema.

### 3. Gobernanza y Runbooks Automatizados (`fuentes/runbooks.py`)
Define protocolos estandarizados de mitigación ante alertas críticas:
* **Enrutamiento a revisión humana (*Human-in-the-Loop*)** cuando la confianza del ViT cae por debajo del umbral operativo.
* **Disparo de pipelines de reentrenamiento** ante detección sostenida de *drift*.
* **Rollback automático** a versiones estables previas registradas en `mlflow.db`.

### 4. Visualización (`dashboard_siniestros_real.html`)
Interfaz web autocontenida para la inspección visual del estado de salud del modelo, historial de incidentes y métricas clave de negocio.

---

## ⚙️ Instalación y Ejecución

1. **Clonar el repositorio e ingresar al directorio:**
   ```bash
   git clone https://github.com/jaramirezes/Actividad8.git
   cd Actividad8
   ```

2. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar el ciclo de monitoreo o simulación:**
   ```bash
   Ejecutar el notebook Actividad8.IPYNB en Google Colab paso a paso para obtener los resultados de manera visual. 
   python fuentes/Actividad8.IPYNB
   ```

   # Simular incidentes operativos
   python fuentes/simulate_incidents.py

   # Ejecutar validación de monitoreo y runbooks
   python fuentes/monitoring.py
   ```

4. **Visualizar el dashboard:**
   Abre el archivo `dashboard_siniestros_real.html` directamente en cualquier navegador web moderno.

---

## 📄 Entregables Académicos (`/docs`)
Para una revisión profunda de la arquitectura, métricas de evaluación, análisis de causa raíz y evidencias de ejecución, consulta los documentos alojados en la carpeta `docs/`:
* `Actividad 8.docx`
* `Informe Técnico.docx`
* `Evidencias.pptx`