#!/bin/bash
# Init script dos clusters do Databricks: instala o Tesseract OCR com o idioma português,
# usado pelo PyMuPDF para extrair texto de PDFs escaneados (notebook 03_trusted/trusted_poli_consolidated_pdf).
# É enviado ao workspace pelo Terraform (infra/databricks.tf).
set -euo pipefail

apt-get update -qq
apt-get install -y -qq tesseract-ocr tesseract-ocr-por
