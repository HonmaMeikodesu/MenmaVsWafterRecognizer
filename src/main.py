from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
import keras

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(250, 250, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(216, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
        ".\\training_set\\",
        target_size=(250, 250),
        batch_size=100,
        class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = validation_datagen.flow_from_directory(
        '.\\test_set\\',
        target_size=(250, 250), 
        batch_size=25,
        class_mode='binary')

history = model.fit(
      train_generator,
      validation_data = validation_generator,  
      epochs=20,
      steps_per_epoch=4,
      validation_steps=3,
      verbose=1)
