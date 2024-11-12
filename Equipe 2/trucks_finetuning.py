import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
# from keras.applications.resnet import ResNet50
# from keras.applications.resnet import preprocess_input
# from keras.applications.vgg19 import VGG19,preprocess_input
# from keras.applications.xception import Xception, preprocess_input
from keras import Model, layers
from keras.callbacks import Callback
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import average_precision_score
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, Dropout

dir_zip = '/content/drive/MyDrive/Classroom/Tópicos Especiais em IA - UFRB/'
work_dir = '/content/sample_data'
num_classes = 3
epochs = 100

def cosine_annealing_schedule(epoch, lr_initial=1e-5, epochs=epochs):
# Função de agendamento de Learning Rate com decaimento cosseno~^
    return lr_initial * (np.cos(np.pi * epoch / epochs) + 1) / 2

lr_scheduler = LearningRateScheduler(cosine_annealing_schedule)

class AUCPRCallback(Callback):
    def __init__(self, validation_generator):
        super().__init__()
        self.validation_generator = validation_generator

    def on_epoch_end(self, epoch, logs=None):
        # Obtém o próximo lote de validação
        x_val, y_val = self.validation_generator.next()

        # Prevê as saídas para os dados de validação
        y_pred = self.model.predict(x_val)

        # O que queremos aqui é uma matriz de tamanho (n_amostras, n_classes)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)  # Se for 1D, transformamos em 2D (1 classe)

        # Converte os rótulos para uma representação binária para cada classe
        y_val = y_val.astype(int)  # Garante que y_val seja inteiro

        # Cria a matriz binária com as classes (uma matriz de tamanho [n_amostras, n_classes])
        y_val_binary = np.zeros((y_val.size, int(np.max(y_val)) + 1))

        # Preenchendo a matriz binária com as classes
        y_val_binary[np.arange(y_val.size), y_val] = 1

        # Calculando o AUC-PRC para cada classe
        try:
            auc_prc = average_precision_score(y_val_binary, y_pred, average='macro')
            print(f'\nEpoch {epoch + 1}: AUC-PRC = {auc_prc:.4f}')

            logs['auc_prc'] = auc_prc

        except ValueError as e:
            print(f'\nEpoch {epoch + 1}: Error calculating AUC-PRC - {str(e)}')

class SWA(Callback):
    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch 
        self.swa_weights = None  # Inicialize a variável de pesos para a média

    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print(f'Stochastic weight averaging selecionado para os últimos {self.nb_epoch - self.swa_epoch} épocas.')

    def on_epoch_end(self, epoch, logs=None):
        # Quando atingirmos a época de SWA, armazenamos os pesos atuais
        if epoch >= self.swa_epoch:
            if self.swa_weights is None:
                # Armazena os pesos do modelo após a primeira SWA epoch
                self.swa_weights = self.model.get_weights()
            else:
                # Média ponderada dos pesos após as épocas SWA
                for i in range(len(self.swa_weights)):
                    # Pondera a média dos pesos
                    self.swa_weights[i] = (self.swa_weights[i] * (epoch - self.swa_epoch) + self.model.get_weights()[i]) / (epoch - self.swa_epoch + 1)

        # Se estiver na última época, salvar os pesos do SWA
        if epoch == self.nb_epoch - 1:
            self.model.set_weights(self.swa_weights)
            print(f'Pesos finais configurados para a média ponderada de SWA.')
            self.model.save_weights(self.filepath)  # Salva os pesos SWA
            print(f'Pesos SWA salvos no arquivo {self.filepath}')

# Caminho para salvar os pesos do SWA
swa_filepath = 'C:\\Users\\pavel\\OneDrive\\Área de Trabalho\\Clusterização\\Modelos - Tópicos Especiais em IA\\pesos_swa.h5'

# Instanciando o SWA
swa = SWA(swa_epoch=80, filepath=swa_filepath)

dir='C:\\Users\\pavel\\OneDrive\\Área de Trabalho\\Clusterização\\Trucks dataset others\\cropped'


def build_finetune_model(base_model, dropout, num_classes):

    x = base_model.output
    
    x = AveragePooling2D((5, 5), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dropout(dropout)(x)
    predictions = Dense(200, activation='tanh', name='finalfc')(x)
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    rotation_range=20.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=[0.9, 1.25],
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    validation_split=0.2,
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    dir,
    # batch_size=32,
    class_mode='sparse',
    subset="training",
    shuffle=True,
    target_size=(299,299))

X_batch, y_batch = next(train_generator)
print(X_batch.shape, y_batch.shape)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    preprocessing_function=preprocess_input)

validation_generator = validation_datagen.flow_from_directory(
    dir,
    shuffle=True,
    class_mode='sparse',
    subset="validation",
    target_size=(299,299))

conv_base = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(299,299,3) 
    )

aucpr_callback = AUCPRCallback(validation_generator=validation_generator)

finetune_model = build_finetune_model(conv_base, 
                                      dropout=0.2, 
                                      num_classes=196)

# checkpoint = tf.train.Checkpoint(model=finetune_model)
# checkpoint.restore('C:\\Users\\pavel\\OneDrive\\Área de Trabalho\\Pesos efficientnetb0\\Nova Pasta\\pesos.h5').assert_consumed()
# finetune_model.load_weights('C:\\Users\\pavel\\OneDrive\\Área de Trabalho\\Pesos efficientnetb0\\Nova Pasta\\pesos_150_epochs.h5')

x = finetune_model.layers[-2].output
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(512, activation='tanh')(x)
# x = layers.Dense(256, activation='tanh')(x)
# x = layers.Dense(128, activation='tanh')(x)
# x = layers.Dense(64, activation='tanh')(x)
# x = layers.Dropout(0.5)(x)
# x = layers.Dense(32, activation='tanh')(x)
x = layers.Dropout(0.2)(x)
final_layer=Dense(3, activation='softmax', name='finalfc')(x)

finetune_model_final = Model(finetune_model.input, final_layer)

finetune_model_final.load_weights(
    'C:\\Users\\pavel\\OneDrive\\Área de Trabalho\\Pesos efficientnetb0\\Nova Pasta\\pesos_150_epochs.h5',
    by_name=True,
    skip_mismatch=True
)

i=0
for layer in finetune_model.layers:
    layer.trainable = False
    if i > 100:
       break


optimizer = keras.optimizers.Adam()
# finetune_model_final.summary()
finetune_model_final.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

history = finetune_model_final.fit(train_generator,
                            epochs=epochs,
                            validation_data=validation_generator,
                            callbacks=[lr_scheduler, swa, aucpr_callback]
)


import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))
plt.plot(history.history['loss'], color='blue')
plt.plot(history.history['val_loss'], color='red')
plt.title('Model Loss', fontsize=20)
plt.ylabel('Loss')
plt.xlabel('Epochs')

plt.legend(['Treinamento', 'Validação'], loc='upper right', fontsize=15)
plt.show()

plt.figure(figsize=(20, 5))
plt.plot(history.history['accuracy'], color='blue')
plt.plot(history.history['val_accuracy'], color='red')
plt.title('Model Accuracy', fontsize=20)
plt.ylabel('Accuracy')
plt.xlabel('Epochs')

plt.legend(['Treinamento', 'Validação'], loc='upper left', fontsize=15)
plt.show()

import pandas as pd

# Armazena as métricas AUC-PRC de treinamento e validação em um dicionário
metrics = {
    'Epochs': range(1, len(history.history['auc_prc']) + 1),
    'AUC-PRC (Train)': history.history['auc_prc'],
}

# Cria um DataFrame do Pandas com as métricas de AUC-PRC
df_metrics = pd.DataFrame(metrics)

maximo_aucprc = df_metrics['AUC-PRC (Train)'].max()
media_aucprc = df_metrics['AUC-PRC (Train)'].mean()

print(f"\nValor máximo do AUC-PRC durante o treino: {round(maximo_aucprc, 3)}")
print(f"\nValor médio do AUC-PRC durante o treino: {round(media_aucprc, 3)}\n")
