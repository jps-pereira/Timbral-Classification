import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Parâmetros para geração de espectrogramas (consistentes com o modelo)
SR = 22050  # Taxa de amostragem
N_FFT = 2048  # Tamanho da janela FFT
HOP_LENGTH = 512  # Tamanho do passo (overlap)
N_MELS = 128  # Número de bandas de mel para mel-espectrograma
SEGMENT_DURATION = 3  # Duração de cada segmento de áudio em segundos

# Função para carregar áudio e gerar mel-espectrograma
def create_mel_spectrogram(y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

# Função para carregar áudio e gerar espectrograma linear (STFT)
def create_stft_spectrogram(y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return stft_db

# Função para salvar espectrogramas como imagens
def save_spectrogram_as_image(spectrogram, filename, cmap='magma'):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, x_axis='time', y_axis='mel' if 'mel' in filename else 'log', cmap=cmap)
    plt.colorbar(format='%+2.0f dB')
    #plt.title(filename.split('/')[-1].replace('_', ' ').replace('-', ' ').replace(' .png', ''))
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def process_audio_file(audio_path, output_base_dir):
    y, sr = librosa.load(audio_path, sr=SR)
    
    # Calcular o número de segmentos
    total_duration = librosa.get_duration(y=y, sr=sr)
    num_segments = int(np.floor(total_duration / SEGMENT_DURATION))
    
    # Extrair nome da classe e do arquivo de áudio
    class_name = os.path.basename(os.path.dirname(audio_path))
    audio_filename_base = os.path.splitext(os.path.basename(audio_path))[0]

    print(f"Processando {audio_path} ({num_segments} segmentos de {SEGMENT_DURATION}s)")

    for i in range(num_segments):
        start_sample = int(i * SEGMENT_DURATION * sr)
        end_sample = int((i + 1) * SEGMENT_DURATION * sr)
        segment = y[start_sample:end_sample]

        if len(segment) < SEGMENT_DURATION * sr: # Garante que o segmento tem o tamanho correto
            continue

        # Gerar e salvar Mel-espectrograma
        mel_spec = create_mel_spectrogram(segment, sr=sr)
        mel_output_dir = os.path.join(output_base_dir, 'mel_spectrograms', class_name)
        os.makedirs(mel_output_dir, exist_ok=True)
        mel_output_path = os.path.join(mel_output_dir, f'{audio_filename_base}_segment_{i:03d}_mel.png')
        save_spectrogram_as_image(mel_spec, mel_output_path, cmap='magma')

        # Gerar e salvar STFT-espectrograma
        stft_spec = create_stft_spectrogram(segment, sr=sr)
        stft_output_dir = os.path.join(output_base_dir, 'stft_spectrograms', class_name)
        os.makedirs(stft_output_dir, exist_ok=True)
        stft_output_path = os.path.join(stft_output_dir, f'{audio_filename_base}_segment_{i:03d}_stft.png')
        save_spectrogram_as_image(stft_spec, stft_output_path, cmap='magma')

    print(f"  -> {num_segments} segmentos processados para {audio_path}")

if __name__ == '__main__':
    input_audio_base_dir = './dataset_wav'
    output_dataset_base_dir = './dataset_png'

    # Estrutura de diretórios de áudio fornecida pelo usuário
    audio_structure = {
        'classe_b': [
            'classe_b_musica_a_tom_Bb.wav',
            'classe_b_musica_a_tom_D.wav',
            'classe_b_musica_b_tom_Ab.wav',
            'classe_b_musica_b_tom_F.wav'
        ],
        'classe_g': [
            'classe_g_musica_a_tom_C.wav',
            'classe_g_musica_a_tom_Eb.wav',
            'classe_g_musica_b_tom_D.wav',
            'classe_g_musica_b_tom_G.wav'
        ],
        'classe_s': [
            'classe_s_musica_a_tom_E.wav',
            'classe_s_musica_a_tom_Gb.wav',
            'classe_s_musica_b_tom_A.wav',
            'classe_s_musica_b_tom_B.wav'
        ],
        'classe_t': [
            'classe_t_musica_a_tom_C.wav',
            'classe_t_musica_a_tom_Gb.wav',
            'classe_t_musica_b_tom_D.wav',
            'classe_t_musica_b_tom_Db.wav'
        ]
    }

    # Criar diretório de saída base
    os.makedirs(output_dataset_base_dir, exist_ok=True)

    print(f"Iniciando a geração do dataset de espectrogramas em {output_dataset_base_dir}")

    for class_name, audio_files in audio_structure.items():
        class_audio_dir = os.path.join(input_audio_base_dir, class_name)
        for audio_file in audio_files:
            full_audio_path = os.path.join(class_audio_dir, audio_file)
            if os.path.exists(full_audio_path):
                process_audio_file(full_audio_path, output_dataset_base_dir)
            else:
                print(f"AVISO: Arquivo não encontrado: {full_audio_path}. Pulando...")

    print("\nGeração do dataset de espectrogramas concluída!")
    print("Verifique a pasta 'dataset_spectrograms' para os resultados.")


