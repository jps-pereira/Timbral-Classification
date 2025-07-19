import streamlit as st
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import tempfile
import os
import torchvision.models as models
from audio_classifier import (
    create_mel_spectrogram, 
    create_stft_spectrogram, 
    save_spectrogram_as_image, 
    load_resnet_model, 
    EnsembleModel,
    get_transforms,
    SpectrogramDataset # Importar SpectrogramDataset para obter as classes
)

# Configuração da página
st.set_page_config(
    page_title="Classificador de Áudio",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Classificador de Áudio com Deep Learning")
st.markdown("### Upload de áudio para classificação usando Mel-espectrogramas e STFT")

# Função para carregar modelos
@st.cache_resource
def load_models():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Obter o número de classes do dataset de espectrogramas
    # Assumimos que a estrutura de diretórios de classes é a mesma para mel e stft
    try:
        # O caminho deve ser relativo ao local de execução do script Streamlit
        # que geralmente é o diretório raiz do projeto.
        # O dataset_png está dentro do repositório clonado.
        dataset_root_dir = os.path.join(os.path.dirname(__file__), 'dataset_png', 'mel_spectrograms')
        dummy_dataset = SpectrogramDataset(root_dir=dataset_root_dir)
        num_classes = len(dummy_dataset.classes)
        class_names = dummy_dataset.classes
        st.write(f"Classes encontradas: {class_names}")
    except Exception as e:
        st.error(f"Erro ao carregar as classes do dataset: {e}. Certifique-se de que o dataset foi gerado corretamente.")
        return None, None, None, None
    
    # Carregar modelos individuais
    # Usando weights=models.ResNet18_Weights.DEFAULT para resolver o warning de depreciação
    mel_model = load_resnet_model(num_classes, weights=models.ResNet18_Weights.DEFAULT)
    stft_model = load_resnet_model(num_classes, weights=models.ResNet18_Weights.DEFAULT)
    
    # Tentar carregar os pesos salvos
    try:
        # Os modelos .pth devem estar no diretório raiz do projeto
        mel_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'mel_resnet_model.pth'), map_location=device))
        stft_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'stft_resnet_model.pth'), map_location=device))
        st.success("Modelos carregados com sucesso!")
    except FileNotFoundError:
        st.warning("Modelos não encontrados (mel_resnet_model.pth ou stft_resnet_model.pth). Execute o treinamento primeiro.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Erro ao carregar os pesos dos modelos: {e}")
        return None, None, None, None
    
    # Criar modelo ensemble
    ensemble_model = EnsembleModel(mel_model, stft_model, num_classes)
    
    mel_model.eval()
    stft_model.eval()
    ensemble_model.eval()
    
    return mel_model, stft_model, ensemble_model, class_names

# Função para processar áudio e fazer predição
def predict_audio(audio_file, mel_model, stft_model, ensemble_model):
    # Salvar arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Carregar áudio com librosa
        y, sr = librosa.load(tmp_path, sr=22050) # Usar SR do audio_classifier

        # Gerar espectrogramas (passando y e sr, não o caminho do arquivo)
        mel_spec = create_mel_spectrogram(y, sr=sr)
        stft_spec = create_stft_spectrogram(y, sr=sr)
        
        # Salvar espectrogramas como imagens temporárias
        mel_img_path = tempfile.mktemp(suffix="_mel.png")
        stft_img_path = tempfile.mktemp(suffix="_stft.png")
        
        save_spectrogram_as_image(mel_spec, mel_img_path)
        save_spectrogram_as_image(stft_spec, stft_img_path)
        
        # Carregar e transformar imagens
        transform = get_transforms()
        
        mel_image = Image.open(mel_img_path).convert("RGB")
        stft_image = Image.open(stft_img_path).convert("RGB")
        
        mel_tensor = transform(mel_image).unsqueeze(0)
        stft_tensor = transform(stft_image).unsqueeze(0)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mel_tensor = mel_tensor.to(device)
        stft_tensor = stft_tensor.to(device)
        
        # Fazer predições
        with torch.no_grad():
            mel_output = mel_model(mel_tensor)
            stft_output = stft_model(stft_tensor)
            ensemble_output = ensemble_model(mel_tensor, stft_tensor)
            
            mel_prob = torch.softmax(mel_output, dim=1)
            stft_prob = torch.softmax(stft_output, dim=1)
            ensemble_prob = torch.softmax(ensemble_output, dim=1)
            
            mel_pred = torch.argmax(mel_prob, dim=1).item()
            stft_pred = torch.argmax(stft_prob, dim=1).item()
            ensemble_pred = torch.argmax(ensemble_prob, dim=1).item()
        
        # Limpar arquivos temporários
        os.unlink(tmp_path)
        os.unlink(mel_img_path)
        os.unlink(stft_img_path)
        
        return {
            'mel_prediction': mel_pred,
            'stft_prediction': stft_pred,
            'ensemble_prediction': ensemble_pred,
            'mel_confidence': mel_prob.max().item(),
            'stft_confidence': stft_prob.max().item(),
            'ensemble_confidence': ensemble_prob.max().item(),
            'mel_spectrogram': mel_spec,
            'stft_spectrogram': stft_spec
        }
        
    except Exception as e:
        # Certifique-se de que o arquivo temporário é removido mesmo em caso de erro
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e

# Interface principal
def main():
    # Carregar modelos
    mel_model, stft_model, ensemble_model, class_names = load_models()
    
    if mel_model is None:
        st.error("Não foi possível carregar os modelos. Verifique os logs acima.")
        return
    
    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "Escolha um arquivo de áudio",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Formatos suportados: WAV, MP3, FLAC, M4A"
    )
    
    if uploaded_file is not None:
        # Mostrar informações do arquivo
        st.write(f"**Arquivo:** {uploaded_file.name}")
        st.write(f"**Tamanho:** {uploaded_file.size} bytes")
        
        # Reproduzir áudio
        st.audio(uploaded_file)
        
        # Botão para classificar
        if st.button("🔍 Classificar Áudio", type="primary"):
            with st.spinner("Processando áudio e gerando predições..."):
                try:
                    results = predict_audio(uploaded_file, mel_model, stft_model, ensemble_model)
                    
                    # Mostrar resultados
                    st.success("Classificação concluída!")
                    
                    # Criar colunas para os resultados
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("🎧 Mel-espectrograma")
                        st.write(f"**Predição:** {class_names[results['mel_prediction']]}")
                        st.write(f"**Confiança:** {results['mel_confidence']:.2%}")
                        
                        # Mostrar mel-espectrograma
                        fig, ax = plt.subplots(figsize=(8, 4))
                        librosa.display.specshow(results['mel_spectrogram'], x_axis='time', y_axis='mel', ax=ax)
                        ax.set_title('Mel-espectrograma')
                        plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("📊 STFT-espectrograma")
                        st.write(f"**Predição:** {class_names[results['stft_prediction']]}")
                        st.write(f"**Confiança:** {results['stft_confidence']:.2%}")
                        
                        # Mostrar STFT-espectrograma
                        fig, ax = plt.subplots(figsize=(8, 4))
                        librosa.display.specshow(results['stft_spectrogram'], x_axis='time', y_axis='log', ax=ax)
                        ax.set_title('STFT-espectrograma')
                        plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
                        st.pyplot(fig)
                    
                    with col3:
                        st.subheader("🤝 Ensemble")
                        st.write(f"**Predição:** {class_names[results['ensemble_prediction']]}")
                        st.write(f"**Confiança:** {results['ensemble_confidence']:.2%}")
                        
                        # Gráfico de comparação
                        fig, ax = plt.subplots(figsize=(8, 4))
                        methods = ['Mel', 'STFT', 'Ensemble']
                        confidences = [
                            results['mel_confidence'],
                            results['stft_confidence'],
                            results['ensemble_confidence']
                        ]
                        bars = ax.bar(methods, confidences, color=['skyblue', 'lightcoral', 'lightgreen'])
                        ax.set_ylabel('Confiança')
                        ax.set_title('Comparação de Confiança')
                        ax.set_ylim(0, 1)
                        
                        # Adicionar valores nas barras
                        for bar, conf in zip(bars, confidences):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{conf:.2%}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Erro ao processar o áudio: {str(e)}")

if __name__ == "__main__":
    main()


