import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

#  Enable dynamic GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
print("✅ Running on:", "GPU" if gpus else "CPU")

#  Dataset configuration
train_dir = r'C:\Users\ncfad\American-Sign-Language-Recognition-System\ASL Dataset\asl_alphabet_train'
test_dir = r'C:\Users\ncfad\American-Sign-Language-Recognition-System\ASL Dataset\asl_alphabet_test'
img_size = (64, 64)
batch_size = 32

#  Data generators (train & validation)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

#  CNN Model Architecture (30 classes)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(30, activation='softmax')  # 30 gestures
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  Callbacks for early stopping & best model saving
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_asl_model.h5', save_best_only=True)
]

#  Train the model
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=callbacks
)

#  Save final model
model.save("asl_model.h5")
print("✅ Training completed and model saved as 'asl_model.h5'")

#  Evaluate final model accuracy
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

#  Save training history for Notebook visualization
np.savez("training_history.npz", 
         accuracy=history.history['accuracy'],
         val_accuracy=history.history['val_accuracy'],
         loss=history.history['loss'],
         val_loss=history.history['val_loss'])

#  Plot accuracy curve for visual analysis
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()
