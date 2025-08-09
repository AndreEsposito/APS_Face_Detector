cat > README.md << EOF
# Face Detector Project 

Detector avançado de rosto, mãos e piscadas com reconhecimento facial, desenvolvido usando MediaPipe, YOLOv8, e face_recognition.

## Tecnologias usadas
- Python 3.12
- OpenCV, MediaPipe
- YOLOv8 (via ultralytics)
- face_recognition (com dlib)
- Torch (CPU/GPU)

## Instalação

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Como Usar

\`\`\`bash
python main.py
\`\`\`

- Pressione **\`q\`** para sair.
- Pressione **\`s\`** para salvar o frame atual (arquivo \`.jpg\`).
- Piscar, fazer gesto com a mão e ver reconhecimento facial em tempo real.

## Estrutura do Projeto

\`\`\`
face_detector_project/
├── main.py
├── faces/
│   ├── meu_rosto1.jpg
│   └── amigo.jpg
├── requirements.txt
├── .gitignore
└── README.md
\`\`\`

## License
MIT
EOF
