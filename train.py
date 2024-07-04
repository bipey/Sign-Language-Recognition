import os
import tensorflow as tf
from tensorflow import keras
from keras import models, layers

os.environ['PYTHONIOENCODING'] = 'utf-8'
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Directories for training and validation data
train_dir = 'data/train'
val_dir = 'data/val'

# Create ImageDataGenerators
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

batch_size = 64

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

class_names = list(train_generator.class_indices.keys())
print(class_names)

# Define the model
model = models.Sequential()
# Convolutional layers
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.4))

model.add(layers.Flatten())
# Fully connected layers
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.3))
# Output layer
model.add(layers.Dense(len(class_names), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint('model.keras', save_best_only=True, monitor='val_loss', mode='min'),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
]

# Create tf.data.Dataset objects
train_ds = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 64, 64, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(class_names)), dtype=tf.float32)
    )
)

val_ds = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 64, 64, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(class_names)), dtype=tf.float32)
    )
)

# Prefetch data for efficiency
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Train the model
history = model.fit(
    train_ds,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,
    validation_data=val_ds,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=callbacks
)

# Save model and weights
model_json = model.to_json()
with open("model.json", 'w') as json_file:
    json_file.write(model_json)
model.save_weights("model.weights.h5")

# Evaluate the model
test_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

test_ds = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 64, 64, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(class_names)), dtype=tf.float32)
    )
).prefetch(buffer_size=tf.data.AUTOTUNE)
test_loss, test_acc = model.evaluate(test_ds, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)