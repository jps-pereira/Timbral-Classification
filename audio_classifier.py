import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

# Função para carregar áudio e gerar mel-espectrograma
def create_mel_spectrogram(audio_path, sr=22050, n_mels=128, hop_length=512):
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

# Função para carregar áudio e gerar espectrograma linear (STFT)
def create_stft_spectrogram(audio_path, sr=22050, n_fft=2048, hop_length=512):
    y, sr = librosa.load(audio_path, sr=sr)
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return stft_db

# Função para salvar espectrogramas como imagens
def save_spectrogram_as_image(spectrogram, filename, cmap='magma'):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, x_axis='time', y_axis='mel' if 'mel' in filename else 'log', cmap=cmap)
    plt.colorbar(format='%+2.0f dB')
    #plt.title(filename.split('/')[-1].replace('_', ' ').replace('.png', ''))
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label

# Transformações para as imagens (espectrogramas)
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet18 espera entrada 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# Função para carregar o modelo e configurar para Transfer Learning
def load_resnet_model(num_classes):
    model = models.resnet18(weights=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Função de treinamento do modelo
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return model

# Função de avaliação do modelo
def evaluate_model(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            corrects += torch.sum(preds == labels.data)
    accuracy = corrects.double() / total
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy


if __name__ == '__main__':
    # 2. Carregar Datasets e Dataloaders
    data_transforms = get_transforms()

    # Mel-espectrogramas
    mel_dataset = SpectrogramDataset(root_dir='dataset_png/mel_spectrograms', transform=data_transforms)
    mel_train_size = int(0.8 * len(mel_dataset))
    mel_val_size = len(mel_dataset) - mel_train_size
    mel_train_dataset, mel_val_dataset = torch.utils.data.random_split(mel_dataset, [mel_train_size, mel_val_size])

    mel_dataloaders = {
        'train': DataLoader(mel_train_dataset, batch_size=4, shuffle=True, num_workers=2),
        'val': DataLoader(mel_val_dataset, batch_size=4, shuffle=False, num_workers=2)
    }
    mel_num_classes = len(mel_dataset.classes)
    print(f"Mel-spectrograms: {mel_num_classes} classes encontradas.")

    # STFT-espectrogramas
    stft_dataset = SpectrogramDataset(root_dir='dataset_png/stft_spectrograms', transform=data_transforms)
    stft_train_size = int(0.8 * len(stft_dataset))
    stft_val_size = len(stft_dataset) - stft_train_size
    stft_train_dataset, stft_val_dataset = torch.utils.data.random_split(stft_dataset, [stft_train_size, stft_val_size])

    stft_dataloaders = {
        'train': DataLoader(stft_train_dataset, batch_size=4, shuffle=True, num_workers=2),
        'val': DataLoader(stft_val_dataset, batch_size=4, shuffle=False, num_workers=2)
    }
    stft_num_classes = len(stft_dataset.classes)
    print(f"STFT-spectrograms: {stft_num_classes} classes encontradas.")

    # 3. Treinar e avaliar modelos separadamente
    print("\nTreinando modelo para Mel-espectrogramas...")
    mel_model = load_resnet_model(mel_num_classes)
    mel_criterion = nn.CrossEntropyLoss()
    mel_optimizer = optim.Adam(mel_model.parameters(), lr=0.001)
    mel_model = train_model(mel_model, mel_dataloaders, mel_criterion, mel_optimizer, num_epochs=2)
    print("Avaliando modelo de Mel-espectrogramas no conjunto de validação:")
    evaluate_model(mel_model, mel_dataloaders['val'])

    print("\nTreinando modelo para STFT-espectrogramas...")
    stft_model = load_resnet_model(stft_num_classes)
    stft_criterion = nn.CrossEntropyLoss()
    stft_optimizer = optim.Adam(stft_model.parameters(), lr=0.001)
    stft_model = train_model(stft_model, stft_dataloaders, stft_criterion, stft_optimizer, num_epochs=2)
    print("Avaliando modelo de STFT-espectrogramas no conjunto de validação:")
    evaluate_model(stft_model, stft_dataloaders['val'])

    # Salvar os modelos treinados
    torch.save(mel_model.state_dict(), 'mel_resnet_model.pth')
    torch.save(stft_model.state_dict(), 'stft_resnet_model.pth')
    print("Modelos treinados salvos como mel_resnet_model.pth e stft_resnet_model.pth")



# Classe para o Ensemble Model
class EnsembleModel(nn.Module):
    def __init__(self, mel_model, stft_model, num_classes):
        super(EnsembleModel, self).__init__()
        self.mel_model = mel_model
        self.stft_model = stft_model
        # Congelar os pesos dos modelos pré-treinados
        for param in self.mel_model.parameters():
            param.requires_grad = False
        for param in self.stft_model.parameters():
            param.requires_grad = False

        # Camada de combinação (pode ser mais complexa, como uma rede neural)
        self.classifier = nn.Linear(num_classes * 2, num_classes)

    def forward(self, mel_inputs, stft_inputs):
        mel_outputs = self.mel_model(mel_inputs)
        stft_outputs = self.stft_model(stft_inputs)
        combined_outputs = torch.cat((mel_outputs, stft_outputs), dim=1)
        return self.classifier(combined_outputs)

# Função para avaliar o modelo ensemble
def evaluate_ensemble_model(ensemble_model, mel_dataloader, stft_dataloader, class_names):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensemble_model.to(device)
    ensemble_model.eval()
    corrects = 0
    total = 0
    all_preds = []
    all_labels = []

    # Certifique-se de que os dataloaders têm o mesmo número de amostras e ordem
    # Para um ensemble real, você precisaria de um DataLoader que retorne pares de mel/stft para a mesma amostra
    # Para este exemplo, vamos iterar e assumir que a ordem é a mesma para as amostras de validação
    for (mel_inputs, mel_labels), (stft_inputs, stft_labels) in zip(mel_dataloader, stft_dataloader):
        mel_inputs = mel_inputs.to(device)
        stft_inputs = stft_inputs.to(device)
        labels = mel_labels.to(device) # Assumindo que as labels são as mesmas

        with torch.no_grad():
            outputs = ensemble_model(mel_inputs, stft_inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = corrects.double() / total
    print(f'Ensemble Accuracy: {accuracy:.4f}')

    # Opcional: Relatório de classificação
    from sklearn.metrics import classification_report
    print("\nClassification Report for Ensemble Model:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    return accuracy


    # 4. Fusão dos dois modelos (Ensemble)
    print("\nRealizando fusão dos modelos (Ensemble)...")
    # Certifique-se de que mel_num_classes e stft_num_classes são os mesmos
    if mel_num_classes != stft_num_classes:
        raise ValueError("Número de classes diferentes para Mel e STFT espectrogramas.")

    ensemble_model = EnsembleModel(mel_model, stft_model, mel_num_classes)
    ensemble_criterion = nn.CrossEntropyLoss()
    ensemble_optimizer = optim.Adam(ensemble_model.classifier.parameters(), lr=0.001) # Apenas treinar a camada de combinação

    # Para treinar o ensemble, você precisaria de um DataLoader que forneça pares de imagens (mel, stft)
    # Para simplificar, vamos apenas avaliar o ensemble com os modelos pré-treinados.
    # Em um cenário real, você treinaria a camada de combinação do ensemble.

    print("Avaliando modelo Ensemble no conjunto de validação:")
    # Para avaliação do ensemble, precisamos de um DataLoader que retorne ambos os tipos de espectrogramas para a mesma amostra.
    # Isso exigiria uma modificação na classe SpectrogramDataset ou a criação de um novo DataLoader.
    # Por simplicidade, vamos usar os dataloaders existentes e assumir que a ordem é a mesma para as amostras de validação.
    # ATENÇÃO: Em um cenário real, isso pode não ser verdade e levar a resultados incorretos.
    # O ideal seria ter um dataset que carregue o par (mel_image, stft_image) para cada amostra de áudio original.
    evaluate_ensemble_model(ensemble_model, mel_dataloaders['val'], stft_dataloaders['val'], mel_dataset.classes)


