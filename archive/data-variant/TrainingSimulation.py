from utilities import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
path = 'data'
data = importData(path)

balanceData(data, display=False)

imagesPath, steering = loadData(path,data)

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)
print('Training Samples:', len(xTrain))
print('Validation Samples:', len(xVal))

model = createModel()
model.summary()

history = model.fit(batchGenerator(xTrain, yTrain, 100, 1),
                    steps_per_epoch=300,
                    epochs=10,
                    validation_data=batchGenerator(xVal, yVal, 100, 0),
                    validation_steps=200,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min',),
                                tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min'),
                                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001, mode='min')])
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epochs')
plt.show()