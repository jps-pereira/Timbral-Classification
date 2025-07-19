import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
import traceback

# Configurações
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

# Função para criar mel-espectrograma
def create_mel_spectrogram(y, sr=SR):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Função para criar espectrograma STFT
def create_stft_spectrogram(y, sr=SR):
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    stft_magnitude = np.abs(stft)
    stft_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
    return stft_db

# Função para salvar espectrograma como imagem
def save_spectrogram_as_image(spectrogram, save_path):
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

# Função para obter transformações
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Função para carregar modelo ResNet
def load_resnet_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

# Classe do modelo ensemble
class EnsembleModel(torch.nn.Module):
    def __init__(self, mel_model, stft_model, num_classes):
        super(EnsembleModel, self).__init__()
        self.mel_model = mel_model
        self.stft_model = stft_model
        self.fc = torch.nn.Linear(num_classes * 2, num_classes)
    
    def forward(self, mel_input, stft_input):
        mel_output = self.mel_model(mel_input)
        stft_output = self.stft_model(stft_input)
        combined = torch.cat((mel_output, stft_output), dim=1)
        return self.fc(combined)

# Função para carregar modelos
def load_models():
    try:
        # Detectar classes automaticamente
        mel_spectrograms_path = "dataset_png/mel_spectrograms"
        if not os.path.exists(mel_spectrograms_path):
            st.error(f"❌ Diretório de mel-espectrogramas não encontrado: {mel_spectrograms_path}")
            return None, None, None, None
        
        class_names = sorted([d for d in os.listdir(mel_spectrograms_path) 
                             if os.path.isdir(os.path.join(mel_spectrograms_path, d))])
        
        if not class_names:
            st.error("❌ Nenhuma classe encontrada no dataset")
            return None, None, None, None
        
        num_classes = len(class_names)
        st.info(f"📊 Classes detectadas: {class_names} (Total: {num_classes})")
        
        # Carregar modelos
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        mel_model = load_resnet_model(num_classes)
        stft_model = load_resnet_model(num_classes)
        
        # Caminhos dos modelos
        mel_model_path = "mel_resnet_model.pth"
        stft_model_path = "stft_resnet_model.pth"
        
        if not os.path.exists(mel_model_path):
            st.error(f"❌ Modelo mel não encontrado: {mel_model_path}")
            st.info("💡 Execute o treinamento primeiro com: python audio_classifier.py")
            return None, None, None, None
            
        if not os.path.exists(stft_model_path):
            st.error(f"❌ Modelo STFT não encontrado: {stft_model_path}")
            st.info("💡 Execute o treinamento primeiro com: python audio_classifier.py")
            return None, None, None, None
        
        mel_model.load_state_dict(torch.load(mel_model_path, map_location=device))
        stft_model.load_state_dict(torch.load(stft_model_path, map_location=device))
        
        # Criar modelo ensemble
        ensemble_model = EnsembleModel(mel_model, stft_model, num_classes)
        
        mel_model.eval()
        stft_model.eval()
        ensemble_model.eval()
        
        return mel_model, stft_model, ensemble_model, class_names
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelos: {e}")
        return None, None, None, None

# Função para processar áudio e fazer predição com segmentação
def predict_audio(audio_file, mel_model, stft_model, ensemble_model):
    # Salvar arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Carregar áudio com librosa
        y, sr = librosa.load(tmp_path, sr=SR)
        
        # Gerar espectrogramas do áudio completo para visualização
        mel_spec_full = create_mel_spectrogram(y, sr=sr)
        stft_spec_full = create_stft_spectrogram(y, sr=sr)
        
        # SEGMENTAÇÃO DO ÁUDIO EM 3 SEGUNDOS PARA INFERÊNCIA
        segment_duration = 3  # segundos
        segment_samples = segment_duration * sr
        
        # Dividir áudio em segmentos de 3 segundos
        segments = []
        for start in range(0, len(y), segment_samples):
            end = min(start + segment_samples, len(y))
            segment = y[start:end]
            
            # Pular segmentos muito pequenos (menos de 1 segundo)
            if len(segment) >= sr:
                segments.append(segment)
        
        if not segments:
            st.error("❌ Áudio muito curto para segmentação")
            return None
        
        st.info(f"🔄 Processando {len(segments)} segmentos de 3 segundos...")
        
        # Processar cada segmento
        mel_predictions = []
        stft_predictions = []
        ensemble_predictions = []
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        transform = get_transforms()
        
        for i, segment in enumerate(segments):
            # Gerar espectrogramas para o segmento
            mel_spec_segment = create_mel_spectrogram(segment, sr=sr)
            stft_spec_segment = create_stft_spectrogram(segment, sr=sr)
            
            # Salvar espectrogramas como imagens temporárias
            mel_img_path = tempfile.mktemp(suffix=f"_mel_seg{i}.png")
            stft_img_path = tempfile.mktemp(suffix=f"_stft_seg{i}.png")
            
            save_spectrogram_as_image(mel_spec_segment, mel_img_path)
            save_spectrogram_as_image(stft_spec_segment, stft_img_path)
            
            # Carregar e transformar imagens
            mel_image = Image.open(mel_img_path).convert("RGB")
            stft_image = Image.open(stft_img_path).convert("RGB")
            
            mel_tensor = transform(mel_image).unsqueeze(0).to(device)
            stft_tensor = transform(stft_image).unsqueeze(0).to(device)
            
            # Fazer predições para o segmento
            with torch.no_grad():
                mel_output = mel_model(mel_tensor)
                stft_output = stft_model(stft_tensor)
                ensemble_output = ensemble_model(mel_tensor, stft_tensor)
                
                mel_prob = torch.softmax(mel_output, dim=1)
                stft_prob = torch.softmax(stft_output, dim=1)
                ensemble_prob = torch.softmax(ensemble_output, dim=1)
                
                mel_predictions.append(mel_prob.cpu().numpy())
                stft_predictions.append(stft_prob.cpu().numpy())
                ensemble_predictions.append(ensemble_prob.cpu().numpy())
            
            # Limpar arquivos temporários do segmento
            os.unlink(mel_img_path)
            os.unlink(stft_img_path)
        
        # Agregar predições de todos os segmentos (média das probabilidades)
        mel_avg_prob = np.mean(mel_predictions, axis=0)
        stft_avg_prob = np.mean(stft_predictions, axis=0)
        ensemble_avg_prob = np.mean(ensemble_predictions, axis=0)
        
        # Obter predições finais
        mel_pred = np.argmax(mel_avg_prob)
        stft_pred = np.argmax(stft_avg_prob)
        ensemble_pred = np.argmax(ensemble_avg_prob)
        
        # Limpar arquivo temporário
        os.unlink(tmp_path)
        
        return {
            'mel_prediction': mel_pred,
            'stft_prediction': stft_pred,
            'ensemble_prediction': ensemble_pred,
            'mel_confidence': mel_avg_prob.max(),
            'stft_confidence': stft_avg_prob.max(),
            'ensemble_confidence': ensemble_avg_prob.max(),
            'mel_spectrogram': mel_spec_full,  # Espectrograma completo para visualização
            'stft_spectrogram': stft_spec_full,  # Espectrograma completo para visualização
            'num_segments': len(segments)
        }
        
    except Exception as e:
        # Certifique-se de que o arquivo temporário é removido mesmo em caso de erro
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        st.error(f"❌ Erro ao processar áudio: {str(e)}")
        return None

# Interface principal
def main():
    st.title("Classificação de Instrumentos Musicais")
    st.write("Upload de um arquivo de áudio para classificação usando mel-espectrogramas e espectrogramas STFT")
    
    # Carregar modelos
    mel_model, stft_model, ensemble_model, class_names = load_models()
    
    if mel_model is None:
        st.stop()
    
    # Upload de arquivo
    uploaded_file = st.file_uploader("Escolha um arquivo de áudio", type=['wav', 'mp3', 'flac'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Classificar Áudio"):
            with st.spinner("Processando áudio..."):
                result = predict_audio(uploaded_file, mel_model, stft_model, ensemble_model)
                
                if result is not None:
                    # Mostrar resultados
                    st.success(f"✅ Processamento concluído! ({result['num_segments']} segmentos de 3s analisados)")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("🎧 Mel-Espectrograma")
                        st.write(f"**Predição:** {class_names[result['mel_prediction']]}")
                        st.write(f"**Confiança:** {result['mel_confidence']:.2%}")
                    
                    with col2:
                        st.subheader("📊 STFT")
                        st.write(f"**Predição:** {class_names[result['stft_prediction']]}")
                        st.write(f"**Confiança:** {result['stft_confidence']:.2%}")
                    
                    with col3:
                        st.subheader("🤝 Ensemble")
                        st.write(f"**Predição:** {class_names[result['ensemble_prediction']]}")
                        st.write(f"**Confiança:** {result['ensemble_confidence']:.2%}")
                    
                    # Visualizar espectrogramas (do áudio completo)
                    st.subheader("📈 Visualização dos Espectrogramas")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Mel-Espectrograma**")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        librosa.display.specshow(result['mel_spectrogram'], sr=SR, hop_length=HOP_LENGTH, 
                                               x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
                        ax.set_title('Mel-Espectrograma')
                        st.pyplot(fig)
                    
                    with col2:
                        st.write("**Espectrograma STFT**")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        librosa.display.specshow(result['stft_spectrogram'], sr=SR, hop_length=HOP_LENGTH,
                                               x_axis='time', y_axis='hz', ax=ax, cmap='viridis')
                        ax.set_title('Espectrograma STFT')
                        st.pyplot(fig)

if __name__ == "__main__":
    main()

