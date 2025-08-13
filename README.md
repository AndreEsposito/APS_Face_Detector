<!-- Hero Section -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/OpenCV-4%2B-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-OK-FF6F00" alt="MediaPipe">
  <img src="https://img.shields.io/badge/YOLOv8-optional-00A67E" alt="YOLOv8">
  <img src="https://img.shields.io/badge/License-MIT-000000" alt="MIT">
</p>

<h1 align="center">🎯 Face Detector Project</h1>
<p align="center">
  <b>Detecção unificada de rostos, piscadas e contagem de dedos em tempo real</b><br>
  <sub>Feito com OpenCV, MediaPipe e suporte opcional a YOLOv8 e Face Recognition</sub>
</p>

<!-- GIF de Demonstração -->
<p align="center">
  <img src="https://media.giphy.com/media/26AHONQ79FdWZhAI0/giphy.gif" width="80%" alt="GIF animado de detecção">
</p>

<!-- Links Rápidos -->
<p align="center">
  <a href="#-instalação">Instalação</a> •
  <a href="#-como-usar">Como usar</a> •
  <a href="#-configurações-no-código">Configurações</a> •
  <a href="#-estrutura-do-projeto">Estrutura</a> •
  <a href="#-contribuição">Contribuição</a>
</p>

---

## ✨ Recursos Principais

- 🎥 **Webcam em tempo real** com janela única: <i>Detecção Unificada</i>  
- 🖐️ **Mãos + contagem de dedos** (MediaPipe Hands)  
- 👁️ **Piscadas** via Face Mesh + EAR (Eye Aspect Ratio)  
- ⚡ **FPS ao vivo** (com suavização)  
- 💾 **Salvar captura** (tecla <kbd>S</kbd>)  
- 🛑 **Sair rápido** (tecla <kbd>Q</kbd>)  
- 🧠 **YOLOv8** e **face_recognition** prontos para ativar (comentados no código)  

---

## 🧰 Tech Stack

<table>
<tr>
<td><b>Core</b></td>
<td>Python 3.8+, OpenCV, NumPy</td>
</tr>
<tr>
<td><b>ML/Vision</b></td>
<td>MediaPipe (Hands/Face Mesh), PyTorch (YOLO), Ultralytics YOLO (opcional), face_recognition (opcional)</td>
</tr>
<tr>
<td><b>SO</b></td>
<td>Windows, Linux, macOS</td>
</tr>
</table>

---

## 📦 Instalação

```bash
# 1) Clonar o repositório
git clone https://github.com/ReaperKoji/face_detector_project.git
cd face_detector_project

# 2) (Opcional) Criar ambiente virtual
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 3) Instalar dependências
pip install -r requirements.txt

# 4) 🚀 Como Usar
python main.py
```

---

## 🖥️ Como Usar

Durante a execução:  
➡️ Pressione **S** para salvar captura  
➡️ Pressione **Q** para sair  

---

## ⚙️ Configurações no Código

No `main.py`, você pode ativar/desativar módulos:

```python
USE_YOLO = False  # Habilita YOLOv8
USE_FACE_RECOGNITION = False  # Habilita reconhecimento facial
```

---

## 📂 Estrutura do Projeto

```
face_detector_project/
│── assets/              # GIFs, imagens e mídias
│── requirements.txt     # Dependências
│── main.py              # Código principal
│── utils.py             # Funções auxiliares
│── README.md            # Documentação
```

---

## 🤝 Contribuição

Sinta-se livre para abrir issues e enviar pull requests.  
Toda contribuição é bem-vinda! 💙

<p align="center">
  <img src="https://media.giphy.com/media/3o7abBphHJngINCHio/giphy.gif" width="150" alt="Obrigado!">
</p>
