from utils import*
from sklearn.model_selection import train_test_split

#step1
path='data'
data=importdata(path)

#step 2 visuailization and distrbution of data
data=balance_data(data,display=True)

#step 3 putting images on a list and the steering on a list
images_path,steerings=load_data(path,data)
#print(images_path[0],steerings[0])


# step 4 splitting the data
x_train,x_test,y_train,y_test=train_test_split(images_path,steerings,test_size=0.2,random_state=5)
print('Total Training Images: ',len(x_train))
print('Total Validation Images: ',len(x_test))


model=Model()
model.summary()



history =model.fit(batch_size(x_train,y_train,100,1),steps_per_epoch=1000,epochs=7,
          validation_data=batch_size(x_test,y_test,100,0),validation_steps=200)


model.save('model.h5')
print('model saved')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()