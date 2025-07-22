# Classificador de Áudio com Deep Learning

Este projeto implementa um sistema completo de classificação de áudio usando mel-espectrogramas e espectrogramas lineares (STFT) com CNNs e Transfer Learning.

## 📋 Características

- **Processamento de Áudio**: Conversão de arquivos de áudio em mel-espectrogramas e espectrogramas STFT
- **Transfer Learning**: Utiliza ResNet18 pré-treinada para classificação de imagens de espectrogramas
- **Ensemble Learning**: Combina predições de ambos os modelos
- **Interface Web**: Interface Streamlit para classificação em tempo real
- **Métricas Detalhadas**: Avaliação completa com relatórios de classificação

## 🚀 Instalação

1. Clone ou baixe os arquivos do projeto
2. Instale as dependências:

```bash
pip install -r requirements.txt
```
3. Preferencialmente faça a instalação em ambiente virtual para não haver conflitos entre as bibliotecas.

## 📁 Estrutura do Projeto

```
├── audio_classifier.py      # Script principal de treinamento
├── real_time_interface.py   # Interface web Streamlit
├── requirements.txt         # Dependências do projeto
├── README.md                # Documentação
└── dataset_png/             # Diretório para dados de treinamento
    ├── mel_spectrograms/    # Imagens de mel-espectrogramas
    └── stft_spectrograms/   # Imagens de espectrogramas STFT
```

## 🎯 Como Usar

### 1. Preparação dos Dados

Organize seus dados de áudio da seguinte forma:

```
dataset_png/
├── mel_spectrograms/
│   ├── classe_x/
│   │   ├── classe_x_musica_a.png
│   │   └── classe_x_musica_b.png
│   └── classe_y/
│       ├── classe_y_musica_a.png
│       └── classe_y_musica_b.png
|       .
|       .
|       .
└── stft_spectrograms/
    ├── classe_x/
    │   ├── classe_x_musica_a.png
    │   └── classe_x_musica_b.png
    └── classe_y/
        ├── classe_y_musica_a.png
        └── classe_y_musica_b.png
        .
        .
        .
```

### 2. Treinamento dos Modelos

Execute o script principal para treinar os modelos:

```bash
python audio_classifier.py
```

Este script irá:

- Treinar modelos separados para mel-espectrogramas e STFT
- Avaliar a performance de cada modelo
- Criar e avaliar o modelo ensemble
- Salvar os modelos treinados

### 3. Interface Web

Após o treinamento, execute a interface web:

```bash
streamlit run real_time_interface.py
```

A interface permite:
- Upload de arquivos de áudio (WAV, MP3, FLAC, M4A)
- Visualização dos espectrogramas gerados
- Classificação usando os três modelos (Mel, STFT, Ensemble)
- Comparação de confiança entre os métodos


### Transfer Learning

- Utiliza ResNet18 pré-treinada do PyTorch
- Substitui a camada final para o número de classes do problema
- Treina apenas a camada de classificação final

### Ensemble Model

```python
# Combina predições de ambos os modelos
ensemble_model = EnsembleModel(mel_model, stft_model, num_classes)
```

## 📊 Avaliação

O sistema fornece:
- **Acurácia** para cada modelo individual
- **Relatório de classificação** detalhado
- **Comparação de confiança** entre métodos
- **Visualização** dos espectrogramas

## 🎧 Tipos de Espectrogramas

### Mel-espectrogramas
- Representação baseada na escala mel (percepção auditiva humana)
- Melhor para características perceptuais do áudio
- Comumente usado em reconhecimento de fala

### Espectrogramas STFT
- Transformada de Fourier de Tempo Curto
- Representação linear da frequência
- Preserva mais detalhes espectrais

## ⚙️ Configurações

### Parâmetros de Áudio
- **Taxa de amostragem**: 22050 Hz
- **N_mels**: 128 (mel-espectrograma)
- **N_fft**: 2048 (STFT)
- **Hop_length**: 512

### Parâmetros de Treinamento
- **Batch size**: 4
- **Learning rate**: 0.001
- **Optimizer**: Adam
- **Épocas**: 20 (configurável)

## 📝 Licença

Este projeto é fornecido como exemplo educacional. Adapte conforme necessário para seus casos de uso específicos.

