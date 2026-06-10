# caries-cnn-panoramic

Código utilizado nos experimentos do trabalho de conclusão de curso:

> LIMA, José Nilton Silva. **Comparação de arquiteturas de redes neurais convolucionais para classificação de cáries em radiografias panorâmicas**. 2026. Trabalho de Conclusão de Curso (Tecnologia em Análise e Desenvolvimento de Sistemas) — Instituto Federal do Piauí, Campus Piripiri, 2026.

---

## Estrutura do repositório

```
caries-cnn-panoramic/
├── 01_dataset_preparation_and_exploration.ipynb   # Montagem do dataset, exploração e experimentos iniciais
├── 02_final_experiments_cross_validation.ipynb    # Experimentos finais que originaram os resultados do artigo
└── README.md
```

---

## Datasets

### InReDD

O **InReDD** (Intraoral and Radiographic Dental Dataset) foi construído a partir de atendimentos clínicos realizados na Faculdade de Odontologia de Ribeirão Preto (FORP-USP). Reúne 924 radiografias panorâmicas de pacientes entre 14 e 81 anos, com anotações produzidas por três radiologistas dentomaxilofaciais com protocolo de revisão independente.

**Acesso restrito.** Para utilizar o InReDD é necessário:

1. Criar uma conta credenciada na plataforma **PhysioNet**: [https://physionet.org](https://physionet.org)
2. Concluir o treinamento de proteção de dados humanos exigido pela plataforma
3. Enviar uma solicitação formal de acesso aos autores diretamente pela página oficial do dataset

**Página oficial do dataset:**
[https://doi.org/10.13026/r5nt-we67](https://doi.org/10.13026/r5nt-we67)

Referência:
> UEHARA MARTINS, C. et al. InReDD-Dataset-PAN924. PhysioNet, nov. 2025. Version 1.0.0. Disponível em: https://doi.org/10.13026/r5nt-we67

---

### DENTEX

O **DENTEX** (Dental Enumeration and Diagnosis on Panoramic X-Rays) é um benchmark público para detecção de múltiplas patologias em radiografias panorâmicas, desenvolvido com imagens de hospitais na Turquia. Disponível sob licença **CC BY-SA 4.0**.

Principais características:

- Pacientes com idade mínima de 12 anos
- Anotações por dente: quadrante, posição FDI e condição patológica
- Condições disponíveis: cárie, cárie profunda, lesão periapical e dente impactado
- Partições oficiais: treinamento (705 imagens), validação (50) e teste (250)

Neste trabalho foram utilizadas apenas as instâncias rotuladas como **cárie** (`Caries`) e **cárie profunda** (`Deep Caries`), abrangendo as três partições.

**Repositório oficial:**
[https://github.com/ibrahimethemhamamci/DENTEX](https://github.com/ibrahimethemhamamci/DENTEX)

Referência:
> HAMAMCI, I. E. et al. Dentex: An abnormal tooth detection with dental enumeration and diagnosis benchmark for panoramic x-rays. arXiv preprint arXiv:2305.19112, 2023.

---

## Como executar

Os notebooks foram desenvolvidos no **Google Colaboratory** com acesso ao Google Drive. Antes de executar, coloque os datasets nas seguintes pastas do seu Drive:

```
MyDrive/
├── INREDD/
│   ├── images/
│   └── annotations/
│       ├── teeth_fdi_labels.json
│       └── mouth_and_teeth_labels.json
└── DENTEX/
    ├── training_data/
    ├── validation_data/
    └── test_data/
```

---

### `01_dataset_preparation_and_exploration.ipynb`

Este notebook cobre a construção do dataset e os experimentos iniciais com hiperparâmetros e data augmentation. Execute as células na ordem indicada pelos títulos.

| Célula | O que faz |
|--------|-----------|
| **Célula 1** — Montar Drive e Configurar Caminhos | Monta o Google Drive e define todas as variáveis de caminho para InReDD e DENTEX. Execute primeiro. |
| **Célula 2** — Instalar Dependências | Instala `pandas`, `scikit-learn`, `pillow` e `psutil`. Verifica TensorFlow e disponibilidade de GPU. |
| **Célula 3** — Importações e Funções Utilitárias | Importa todas as bibliotecas e define constantes globais (`TARGET_SIZE=299x299`, `EXPANSION_FACTOR=3.0`, `RANDOM_SEED=42`). |
| **Célula 4** — Pré-processamento InReDD | Carrega `mouth_and_teeth_labels.json`, extrai anotações de dentes cariados (C + Dc) e saudáveis (H), e recorta cada dente individualmente com expansão de 300%. |
| **Célula 5** — Pré-processamento DENTEX | Processa os JSONs das três partições do DENTEX, filtrando apenas cárie e cárie profunda (`category_id_3` ∈ {1, 3}). |
| **Célula 6** — Combinar e Balancear | Reúne os positivos de ambas as fontes e amostra aleatoriamente até 3.652 exemplares por classe, totalizando 7.304 imagens. |
| **Célula 7** — Criar Splits 5-Fold | Gera os 5 folds estratificados com `StratifiedKFold(n_splits=5)` e salva cada fold como JSON. Inclui célula de verificação com 6 checagens de integridade. |
| **Célula 8** — Construção dos Modelos | Define a função `build_model()` para Inception-v3, InceptionResNet-v2, Xception e EfficientNetV2-S com pesos ImageNet e fine-tuning completo. |
| **Célula 9** — Pipeline com Data Augmentation | Configura `ImageDataGenerator` com `width_shift_range=0.25`, `height_shift_range=0.25`, `zoom_range=0.15`, `horizontal_flip`, `vertical_flip` e `brightness_range=[0.8, 1.2]`. |
| **Célula 10** — Treinamento (5-Fold) | Executa o treinamento de cada arquitetura nos 5 folds. As sub-células rodam individualmente: Inception-v3, InceptionResNet-v2, Xception e EfficientNetV2-S. |
| **Células 11–13** — Profiling e Resumo | Consolidam os resultados de desempenho e tempo/memória por arquitetura e exibem tabela final com média e desvio padrão por métrica. |

---

### `02_final_experiments_cross_validation.ipynb`

Este notebook contém o protocolo final que gerou os resultados reportados no artigo. O dataset já deve estar pré-processado pelo notebook anterior antes de executar.

| Célula | O que faz |
|--------|-----------|
| **Célula 0** — Backup do Dataset | Cria cópia integral de `DENTAL_PREPROCESSED/` antes de qualquer operação. Idempotente: não sobrescreve backup existente. Execute primeiro. |
| **Célula 1** — Configurar Caminhos | Define caminhos para o dataset pré-processado, separando `splits/train_val` (1/3) e `splits/inference` (2/3). Não configura caminhos do InReDD/DENTEX originais. |
| **Célula 2** — Verificação de CUDA e GPU | Verifica driver NVIDIA, versão do TensorFlow e memória disponível na GPU. Útil para confirmar que o ambiente está corretamente configurado antes do treinamento. |
| **Célula 2b** — Instalar Dependências | Instala dependências necessárias e confirma versões. |
| **Célula 3** — Importações e Constantes | Importa bibliotecas e fixa sementes de aleatoriedade (`RANDOM_SEED=42`) para garantir reprodutibilidade. |
| **Célula 4** — Carregar Manifesto | Carrega `task4_manifest.json` gerado pelo notebook anterior. Verifica existência de todas as imagens no disco antes de prosseguir. |
| **Célula 5** — Divisão 1/3 Treino + 2/3 Inferência | Divide o manifesto em subconjunto de treinamento/validação (2.434 imagens) e subconjunto de inferência (4.870 imagens). Verifica balanceamento de classes nas duas partições. |
| **Célula 6** — Criar Splits 5-Fold | Aplica `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` exclusivamente sobre as 2.434 imagens e salva os folds em JSON. |
| **Célula 6b** — Verificação de Integridade | Executa 6 checagens: existência dos JSONs, estrutura, proporções 80/20, equilíbrio de classes, existência dos arquivos no disco e ausência de duplicatas entre splits. |
| **Célula 7** — Construção dos Modelos | Define `build_model()` com as quatro arquiteturas. Fine-tuning completo sobre pesos ImageNet. |
| **Célula 8** — Pipeline SEM Data Augmentation | Configura `ImageDataGenerator` apenas com `rescale=1./255`. Esta é a diferença central em relação ao notebook anterior. |
| **Célula 9** — Treinamento (5-Fold) | Executa o treinamento final com learning rates ajustados (`lr=1e-5` para Inception-v3, InceptionResNet-v2 e EfficientNetV2-S; `lr=1e-6` para Xception). Early stopping com paciência de 6 épocas e `ReduceLROnPlateau`. |
| **Células de treinamento individuais** | Cada arquitetura tem sua própria célula de execução: `train_task4('inceptionv3')`, `train_task4('inceptionresnetv2')`, `train_task4('xception')`, `train_task4('efficientnetv2s')`. Execute uma por vez para monitorar o uso de VRAM. |
| **Célula 10** — Profiling Consolidado | Exibe tabelas de desempenho, tempo de treinamento e uso de memória por arquitetura. |
| **Célula 11** — Resumo Final | Imprime tabela com precisão, revocação, acurácia, especificidade e F1-score por fold e médias finais para cada arquitetura. |

---

## Requisitos

- Google Colab Pro (recomendado para GPU T4 com High RAM)
- TensorFlow 2.20.0
- CUDA 12.5.1 / cuDNN 9
- Python 3.10+
- Bibliotecas: `pandas`, `scikit-learn`, `pillow`, `psutil`

---

## Citação

Se este código for útil para seu trabalho, cite:

> LIMA, José Nilton Silva. **caries-cnn-panoramic**: código dos experimentos de comparação de arquiteturas CNN para classificação de cáries em radiografias panorâmicas. GitHub, 2026. Disponível em: [https://github.com/j-nilton/caries-cnn-panoramic](https://github.com/j-nilton/caries-cnn-panoramic). Acesso em: [data de acesso].

---

## Licença

MIT License
