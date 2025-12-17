import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

#Dataset path 
dataset_dir = "waste_dataset"  #change this to your dataset folder

#Parameters
img_size = (224, 224)  #MobileNetV2 default
batch_size = 32
epochs = 15

#Data Augmentation & Split 
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

#Load MobileNetV2 Pretrained Base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*img_size, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze base layers

#Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Train ===
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

#Unfreeze some layers for fine-tuning 
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False  #train only the last 40 layers

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

#Save Model 
model.save("waste_classifier_mobilenetv2.h5")

#Plot Training Graphs
plt.figure()
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Model Accuracy")
plt.show()

plt.figure()
plt.plot(history.history['loss'] + history_fine.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Model Loss")
plt.show()

print("✅ Model training completed and saved as 'waste_classifier_mobilenetv2.h5'")
