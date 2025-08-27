# APS Face Access (OpenCV + Facemark LBF + LBPH + RBAC) — Python 3.13

Sistema de **identificação/autenticação biométrica facial** com **liveness (piscada/EAR)** e **controle de acesso por níveis (RBAC 1/2/3)**, compatível com **Python 3.13** (Windows).
Projeto **modularizado**, com CLI clara e **semáforo de captura** no ENROLL para guiar a aquisição de amostras.

> Hotkeys da janela: **S** salva um frame em `./captures/` • **Q** encerra.

---

## Sumário

* [Arquitetura & Pastas](#arquitetura--pastas)
* [Tecnologias Utilizadas](#tecnologias-utilizadas)
* [Pré-requisitos](#pré-requisitos)
* [Instalação](#instalação)
* [Como Rodar (Passo a Passo)](#como-rodar-passo-a-passo)
* [Comandos & Parâmetros](#comandos--parâmetros)
* [Funcionalidades](#funcionalidades)
* [Dicas de Qualidade & Calibração](#dicas-de-qualidade--calibração)
* [Logs & Relatórios](#logs--relatórios)
* [Solução de Problemas (FAQ)](#solução-de-problemas-faq)
* [Roadmap (Evoluções Sugeridas)](#roadmap-evoluções-sugeridas)
* [Licença](#licença)

---

## Arquitetura & Pastas

```
APS_Face_Detector/
├─ core/
│  ├─ config.py            # thresholds, tamanhos, RBAC
│  ├─ paths.py             # caminhos e criação de pastas
│  ├─ utils.py             # logging, overlays e salvar frame
│  ├─ detector.py          # Haar + detectMultiScale
│  ├─ landmarks.py         # Facemark LBF + EAR + crop/normalize
│  ├─ liveness.py          # janela de blinks
│  ├─ recognizer_lbph.py   # treino/predição LBPH
│  ├─ rbac.py              # overlay do nível e recursos
│  └─ storage.py           # SQLite + logs CSV
├─ models/
│  ├─ lbfmodel.yaml        # (baixe e coloque aqui)
│  └─ lbph.yml             # (gerado após ENROLL)
├─ data/                   # (criado em runtime)
│  ├─ samples/<user_id>/*.jpg
│  └─ db.sqlite
├─ reports/                # (criado em runtime)
│  └─ access_log.csv
├─ captures/               # (criado em runtime quando capturado um frame)
├─ main.py                 # CLI: enroll/auth
└─ requirements.txt
```

---

## Tecnologias Utilizadas

* **Python 3.13**
* **OpenCV (opencv-contrib-python)**

  * Haar Cascade (detecção)
  * **Facemark LBF** (68 landmarks) para EAR/pose
  * **LBPH** (reconhecimento)
* **NumPy**
* **SQLite** (persistência de usuários/nível)
* **CSV** (log de tentativas)

---

## Pré-requisitos

* Windows com webcam funcionando
* Python 3.13
* Evite instalar o projeto em caminhos com acentos/símbolos (ex.: `°`, `ç`), para não quebrar carregamento de modelos do OpenCV.

> **Importante:** o arquivo `models/lbfmodel.yaml` **não vai no repositório**. Você deve baixá-lo e colocá-lo em `models/`.

---

## Instalação

```powershell
# 1) Ambiente virtual
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1 
# caso o powershell bloqueie esse comando acima, roda esse comando abaixo:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force 

# 2) Instale dependências
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
# ou, manualmente:
# pip uninstall -y opencv-python
# pip install opencv-contrib-python>=4.10 numpy>=1.26
```

> Se você tinha `opencv-python` instalado, **desinstale** e use **opencv-contrib-python** (necessário para `cv2.face` e Facemark LBF).

---

## Como Rodar (Passo a Passo)

### 1) Obtendo o **`models/lbfmodel.yaml`** (landmarks 68 pts)

O OpenCV **referencia publicamente** um modelo LBF pré-treinado para 68 pontos faciais; a própria documentação aponta para esse arquivo hospedado no GitHub. ([docs.opencv.org][1])

```bash
mkdir -p models
curl -L "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml" -o models/lbfmodel.yaml
```

> O link acima é o modelo LBF do repositório do GSOC 2017 (usado pela doc/tutoriais de OpenCV). Se preferir, confira o repositório: kurnianggoro/GSOC2017. ([GitHub][2])

[1]: https://github.com/kurnianggoro/GSOC2017/blob/master/data/lbfmodel.yaml "GSOC2017"
[2]: https://github.com/kurnianggoro/GSOC2017?utm_source=chatgpt.com "kurnianggoro/GSOC2017"

### 2) Coloque o modelo de landmarks

* Salve **`models/lbfmodel.yaml`** (68 pontos) em `./models/`.

### 3) Cadastro (ENROLL)

Captura amostras do usuário, já **recortadas/normalizadas** (grayscale, 200x200) e treina o **LBPH**.

**Comando básico:**

```powershell
python .\main.py enroll --user "Seu Nome" --nivel 2 --samples 20 --camera-index 0 --model-path models\lbfmodel.yaml
```

**Recomendado (captura mais “qualificada”):**

```powershell
# 0.8s entre capturas + exige blink + exige diversidade de pose
python .\main.py enroll --user "Seu Nome" --nivel 2 --samples 20 --camera-index 0 --model-path models\lbfmodel.yaml --capture-interval 0.8 --capture-on-blink --pose-diversity
```

* **Semáforo no ENROLL:** painel no canto superior direito

  * **Verde** = pronto para capturar (face ok, intervalo ok, blink/pose se ativados)
  * **Vermelho** = aguarde cumprir os critérios
  * Linhas mostram o status individual: **Face**, **Interval**, **Blink**, **Pose**

> Ao terminar, o modelo LBPH é salvo em `models/lbph.yml`.

### 4) Autenticação (AUTH)

Realiza **detecção → landmarks/EAR → liveness → LBPH**. Se aprovado, mostra **overlay RBAC** com os recursos do nível do usuário.

```powershell
python .\main.py auth --camera-index 0 --model-path models\lbfmodel.yaml
```

---

## Comandos & Parâmetros

### Subcomandos

* `enroll` — cadastro e treino
* `auth` — autenticação com liveness + RBAC

### ENROLL – parâmetros principais

| Parâmetro            | Descrição                                        | Padrão                 |
| -------------------- | ------------------------------------------------ | ---------------------- |
| `--user`             | Nome do usuário a cadastrar                      | **obrigatório**        |
| `--nivel`            | Nível de acesso (1/2/3)                          | **obrigatório**        |
| `--samples`          | Nº de amostras a capturar                        | 20                     |
| `--camera-index`     | Índice da webcam                                 | 0                      |
| `--model-path`       | Caminho do `lbfmodel.yaml`                       | `models/lbfmodel.yaml` |
| `--capture-interval` | **Tempo mínimo** entre capturas (s)              | 0.7                    |
| `--capture-on-blink` | Captura **somente após blink**                   | desativado             |
| `--pose-diversity`   | Exige **variação de pose** (yaw/pitch)           | desativado             |
| `--yaw-thresh`       | Variação mínima de yaw (se `--pose-diversity`)   | 0.15                   |
| `--pitch-thresh`     | Variação mínima de pitch (se `--pose-diversity`) | 0.12                   |

### AUTH – parâmetros principais

| Parâmetro        | Descrição                                        | Padrão                       |
| ---------------- | ------------------------------------------------ | ---------------------------- |
| `--camera-index` | Índice da webcam                                 | 0                            |
| `--model-path`   | Caminho do `lbfmodel.yaml`                       | `models/lbfmodel.yaml`       |
| `--blink-thresh` | Limiar EAR (liveness)                            | definido em `core/config.py` |
| `--lbph-thresh`  | Limiar de decisão do LBPH (menor = mais estrito) | 70.0                         |

---

## Funcionalidades

* **Detecção facial (Haar)**
* **Landmarks (Facemark LBF, 68 pts)**

  * **EAR** (Eye Aspect Ratio) para liveness por **piscada**
  * Estimativa simples de **yaw/pitch** para diversidade de pose no ENROLL
* **Semáforo de captura (ENROLL)** para orientar o usuário
* **Cadastro (ENROLL) inteligente**

  * Intervalo mínimo entre capturas
  * Gate por **blink** (opcional)
  * Gate por **diversidade de pose** (opcional)
* **Reconhecimento (AUTH)** via **LBPH** (cv2.face)

  * Decisão por **distância** + **liveness OK**
* **RBAC (1/2/3)** com overlay didático dos recursos liberados
* **Persistência**

  * Usuários/nível: **SQLite** (`data/db.sqlite`)
  * Amostras: `data/samples/<user_id>/*.jpg`
  * Modelo LBPH: `models/lbph.yml`
* **Logs CSV** das tentativas (`reports/access_log.csv`)
* **Compatível com Windows/Python 3.13**
* **Offline** (não depende de internet após baixar o `lbfmodel.yaml`)

---

## Dicas de Qualidade & Calibração

* **Varie a pose** no ENROLL (frente/esquerda/direita/cima/baixo) e a iluminação (sem exageros).
* Use `--capture-interval 0.8 --capture-on-blink --pose-diversity` para um dataset mais diverso.
* **Calibre o limiar LBPH**: comece com `--lbph-thresh 70`.

  * Se houver falsos positivos, reduza (ex.: 60–65).
  * Se houver muitos falsos negativos, aumente (ex.: 75–80).
* Garanta que **apenas 1 face** esteja visível no AUTH, próximo à câmera.

---

## Logs & Relatórios

* `reports/access_log.csv` registra:
  `timestamp, user_pred_id, user_pred_nome, distancia, liveness_ok, nivel_concedido, obs`
* Use esse CSV para montar **métricas** (acurácia, FAR/FRR simples) e gráficos no relatório da APS.

---

## Solução de Problemas (FAQ)

**“cv2.face indisponível / Facemark não existe”**
→ Desinstale `opencv-python` e instale `opencv-contrib-python`.

```powershell
pip uninstall -y opencv-python
pip install --upgrade opencv-contrib-python
```

**“Can't open haarcascade\_frontalface\_default.xml”**
→ Caminhos com acentos/símbolos podem quebrar o carregamento nativo.

* Mova o projeto para um caminho ASCII simples (ex.: `C:\projetos\APS\FaceDetector`).
* Recrie o venv e reinstale as deps.

**“lbfmodel.yaml não encontrado ou corrompido”**
→ Coloque o arquivo correto em `models/lbfmodel.yaml` (tamanho típico: vários MB).

**“Modelo LBPH vazio”**
→ Rode o **ENROLL** primeiro (capturando amostras), depois **AUTH**.

**“Múltiplas faces detectadas”**
→ Aproxime apenas um usuário.

**“Blink não detecta / liveness falha”**
→ Ajuste `--blink-thresh` (AUTH) ou melhore a iluminação.
→ No ENROLL, use `--capture-on-blink` para educar o usuário a piscar.

---

## Roadmap (Evoluções Sugeridas)

* **Cotas por pose** (frente/esq/dir/cima/baixo) no ENROLL para balancear samples.
* **Filtro de qualidade** (blur/iluminação) para descartar amostras ruins.
* **Relatório PDF** automatizado a partir do CSV (gráficos e sumários).
* **API (FastAPI)** com endpoints `/auth/face` e recursos por nível.
* Detector **DNN** (SSD/ResNet, RetinaFace) e embeddings mais robustos (troca de LBPH).

---

## Licença

Defina uma licença para o repositório (ex.: **MIT**).
O arquivo `lbfmodel.yaml` é de terceiros (OpenCV/contrib). **Verifique a licença** do modelo antes de redistribuir.

---

**Pronto!** Com este projeto você cobre: **aquisição por vídeo**, **biometria facial** com **liveness**, **autenticação** e **controle de acesso por níveis**, além de **logs** para análise — exatamente o que a APS pede.
