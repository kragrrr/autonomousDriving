from utilities import *
from sklearn.model_selection import train_test_split

path = 'data'
data = importData(path)

balanceData(data, display=True)

imagesPath, steering = loadData(path,data)

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)
print('Training Samples:', len(xTrain))
print('Validation Samples:', len(xVal))

# save the training and validation samples in a csv file with corresponding steering angles
data = {'Images': imagesPath, 'Actual_Angle': steering}
df = pd.DataFrame(data)
df.to_csv('data/data.csv', index=False)

model = createModel()
model.summary()

history = model.fit(batchGenerator(xTrain, yTrain, 100, 1), steps_per_epoch=300, epochs=10, validation_data=batchGenerator(xVal, yVal, 100, 0), validation_steps=200)
model.save('models/model3.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()