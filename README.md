# APS Face Detector (OpenCV + Facemark LBF • Python 3.13)

Detecção de rosto via **OpenCV (Haar Cascade)** e **landmarks faciais (68 pontos)** com **Facemark LBF**.
Calcula **EAR (Eye Aspect Ratio)** para indicar piscadas em tempo real.
Compatível com **Python 3.13** no Windows.

> **Hotkeys**: **S** salva um frame em `./captures/` • **Q** encerra.

---

## Sumário

* [Pré-requisitos](#pré-requisitos)
* [Instalação (Windows/PowerShell)](#instalação-windowspowershell)
* [Modelo de Landmarks (lbfmodel.yaml)](#modelo-de-landmarks-lbfmodelyaml)
* [Como executar](#como-executar)
* [Opções de linha de comando](#opções-de-linha-de-comando)
* [Estrutura do projeto](#estrutura-do-projeto)
* [Solução de problemas (FAQ)](#solução-de-problemas-faq)
* [Alterações principais](#alterações-principais)
* [Roadmap](#roadmap)

---

## Pré-requisitos

* **Python 3.13** (Windows 10/11)
* Webcam habilitada e permissão de câmera no sistema

---

## Instalação (Windows/PowerShell)

```powershell
# 1) Clonar o repositório
git clone https://github.com/AndreEsposito/APS_Face_Detector.git
cd APS_Face_Detector

# 2) Criar e ativar o ambiente virtual
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Atualizar ferramentas básicas
pip install --upgrade pip wheel setuptools

# 4) Instalar dependências principais
pip install opencv-contrib-python numpy
```

> **Nota:** se houver conflito com `opencv-python`, remova-o e mantenha apenas o **contrib**:

```powershell
pip uninstall -y opencv-python
pip install opencv-contrib-python
```

### (Opcional) YOLO/Ultralytics

Não é necessário para rodar. Se quiser testar depois:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

---

## Modelo de Landmarks (`lbfmodel.yaml`)

O **Facemark LBF** precisa do arquivo de **modelo pré-treinado** (68 pontos) em formato **YAML**.

1. Baixe `lbfmodel.yaml` (clique em **Raw** e salve o arquivo).
2. Crie a pasta `models/` na raiz do projeto.
3. Salve o arquivo em: `models/lbfmodel.yaml`.

Estrutura esperada:

```
APS_Face_Detector/
 ├─ main.py
 ├─ models/
 │   └─ lbfmodel.yaml
 └─ ...
```

> Sem esse arquivo o programa não consegue localizar os 68 pontos da face (olhos, boca, etc.) e, portanto, não calcula o EAR.

---

## Como executar

Na raiz do projeto (venv ativo):

```powershell
python .\main.py --camera-index 0 --model-path models\lbfmodel.yaml
```

Exemplos úteis:

```powershell
# usar outra câmera
python .\main.py --camera-index 1 --model-path models\lbfmodel.yaml

# ajustar resolução e limiar de piscada
python .\main.py --width 1280 --height 720 --blink-thresh 0.20 --model-path models\lbfmodel.yaml
```

---

## Opções de linha de comando

```text
--camera-index   Índice da webcam (default: 0)
--model-path     Caminho do lbfmodel.yaml (default: models/lbfmodel.yaml)
--blink-thresh   Limiar EAR para indicar piscada (default: 0.22)
--save-dir       Pasta de saída para capturas (default: captures/)
--width          Largura desejada do frame (0 = manter padrão)
--height         Altura desejada do frame (0 = manter padrão)
--detector       Detector de face (apenas 'haar' no momento)
```

---

## Estrutura do projeto

```
APS_Face_Detector/
├─ main.py                 # Pipeline principal (Haar + Facemark LBF + EAR)
├─ captures/               # (criada em runtime) frames salvos com 'S'
├─ models/
│  └─ lbfmodel.yaml        # modelo de landmarks (68 pts) - necessário
├─ requirements.txt        # (opcional) mínimo sugerido abaixo
└─ ...
```

**requirements.txt (mínimo sugerido):**

```txt
numpy>=1.26
opencv-contrib-python>=4.10
```

*(YOLO/Torch são opcionais e não entram no mínimo.)*

---

## Solução de problemas (FAQ)

**1) `cv2.error ... in function 'fit' ... faces is not a numpy array`**

* Atualize o `main.py`. O projeto já converte `faces_rects` para `NumPy (N,4) int32` e chama `facemark.fit` com imagem **em escala de cinza**.

**2) `module 'cv2' has no attribute 'face'`**

* Falta o módulo contrib. Instale:

  ```powershell
  pip uninstall -y opencv-python
  pip install opencv-contrib-python
  ```

**3) `Não foi possível abrir a webcam`**

* Tente `--camera-index 1` ou `2`.
* Feche apps que usam a câmera (Teams/Zoom/OBS).
* Verifique permissões de câmera no Windows.
* Iluminação fraca atrapalha a detecção.

**4) `Modelo LBF não encontrado`**

* Baixe o `lbfmodel.yaml` e coloque em `.\models\lbfmodel.yaml`.
* Ajuste `--model-path` se estiver em outra pasta.

**5) Baixo FPS / detecção instável**

* Use `--width 640 --height 480`.
* Aumente `minSize` no detector (alterar no código, se necessário).
* Garanta iluminação frontal e rosto a \~50–70cm da câmera.

---

## Alterações principais

* **Removido**: MediaPipe e face\_recognition (dlib).
* **Adicionado**: OpenCV **Facemark LBF** para landmarks 68-pts.
* **Mantido**: Haar Cascade para detecção de face.
* **Compatibilidade**: Python **3.13** no Windows.
* **CLI**: flags para câmera, modelo, EAR, resolução e pasta de saída.

---

## Roadmap

* Detector DNN (SSD/ResNet-10) como alternativa mais robusta ao Haar.
* Opção `--detector yolo` (Ultralytics) quando desejado.
* Métricas (blinks/min, latência, FPS médio) e logs.
* **Auto-download** do `lbfmodel.yaml` no primeiro run (opt-in).
* Scripts `scripts/setup.ps1` e `scripts/run.ps1` para onboarding rápido.

---
