# 🎵 Classificador de Áudio com Deep Learning

Este projeto implementa um sistema completo de classificação de áudio usando mel-espectrogramas e espectrogramas lineares (STFT) com CNNs e Transfer Learning.

## 📋 Características

- **Processamento de Áudio**: Conversão de arquivos de áudio em mel-espectrogramas e espectrogramas STFT
- **Transfer Learning**: Utiliza ResNet18 pré-treinada para classificação de imagens de espectrogramas
- **Ensemble Learning**: Combina predições de ambos os modelos para melhor performance
- **Interface Web**: Interface Streamlit para classificação em tempo real
- **Métricas Detalhadas**: Avaliação completa com relatórios de classificação

## 🚀 Instalação

1. Clone ou baixe os arquivos do projeto
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## 📁 Estrutura do Projeto

```
├── audio_classifier.py      # Script principal de treinamento
├── real_time_interface.py   # Interface web Streamlit
├── requirements.txt         # Dependências do projeto
├── README.md               # Documentação
└── data/                   # Diretório para dados de treinamento
    ├── mel_spectrograms/   # Imagens de mel-espectrogramas
    └── stft_spectrograms/  # Imagens de espectrogramas STFT
```

## 🎯 Como Usar

### 1. Preparação dos Dados

Organize seus dados de áudio da seguinte forma:

```
data/
├── mel_spectrograms/
│   ├── classe_a/
│   │   ├── audio1_mel.png
│   │   └── audio2_mel.png
│   └── classe_b/
│       ├── audio3_mel.png
│       └── audio4_mel.png
└── stft_spectrograms/
    ├── classe_a/
    │   ├── audio1_stft.png
    │   └── audio2_stft.png
    └── classe_b/
        ├── audio3_stft.png
        └── audio4_stft.png
```

### 2. Treinamento dos Modelos

Execute o script principal para treinar os modelos:

```bash
python audio_classifier.py
```

Este script irá:
- Gerar dados dummy para demonstração (substitua pelos seus dados reais)
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

## 🔧 Funcionalidades Principais

### Processamento de Áudio

```python
# Gerar mel-espectrograma
mel_spec = create_mel_spectrogram(audio_path)

# Gerar espectrograma STFT
stft_spec = create_stft_spectrogram(audio_path)

# Salvar como imagem
save_spectrogram_as_image(mel_spec, 'output.png')
```

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
- **Épocas**: 25 (configurável)

## 🔄 Fluxo de Trabalho

1. **Carregamento**: Áudio → Librosa
2. **Conversão**: Áudio → Espectrogramas (Mel + STFT)
3. **Visualização**: Espectrogramas → Imagens PNG
4. **Treinamento**: Imagens → CNN (ResNet18)
5. **Ensemble**: Combinação dos modelos
6. **Inferência**: Novo áudio → Classificação

## 🚨 Notas Importantes

- **Dados Dummy**: O script atual gera dados fictícios para demonstração
- **Dados Reais**: Substitua pela sua estrutura de dados real
- **Classes**: Adapte o número de classes conforme seu problema
- **Hardware**: GPU recomendada para treinamento mais rápido

## 🔧 Personalização

### Adicionar Novas Classes
1. Modifique a estrutura de diretórios
2. Ajuste `num_classes` no código
3. Atualize `class_names` na interface

### Modificar Arquitetura
- Substitua ResNet18 por outros modelos (ResNet50, EfficientNet, etc.)
- Ajuste a camada de ensemble para combinações mais complexas

### Otimizar Performance
- Aumente o número de épocas
- Ajuste learning rate
- Implemente data augmentation
- Use técnicas de regularização

## 📈 Resultados Esperados

O sistema fornece três tipos de predição:
- **Mel-espectrograma**: Focado em características perceptuais
- **STFT**: Focado em características espectrais detalhadas
- **Ensemble**: Combinação otimizada de ambos

## 🤝 Contribuições

Para melhorar o sistema:
1. Implemente data augmentation
2. Adicione mais arquiteturas de rede
3. Otimize hiperparâmetros
4. Adicione suporte a mais formatos de áudio

## 📝 Licença

Este projeto é fornecido como exemplo educacional. Adapte conforme necessário para seus casos de uso específicos.

