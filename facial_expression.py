import os
import numpy as np
from skimage import io
from skimage.color import grey2rgb
from skimage.transform import resize
with open("CohonExp7.txt") as fobj:
    lines = fobj.readlines()

label=[]

for i in range(0,1632,1):
#    print(lines[i][-2:-1])
    a=lines[i][-2:-1]
    label.append(a)
    
label=np.array(label)
label=label.astype('int32')


PATH = os.getcwd()
data_path = PATH +'/s'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        img_path = data_path + '/'+ dataset + '/'+ img
        img =io.imread(img_path,as_grey=True)
        img=resize(img,(100,100))
        img=grey2rgb(img)
        img=img.astype('float')
        img_data_list.append(img)
img_data=np.array(img_data_list)
img_data.shape

np.save('cohn_kanade.npy',img_data)

img_data=np.load('cohn_kanade.npy')


no_of_samples,height,width,depth=img_data.shape


print(no_of_samples,height,width,depth)

classes=np.unique(label)
num_of_classes=classes.shape[0]

print(num_of_classes)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(img_data,label,test_size=0.20)

print(x_train.shape)
print(x_test.shape)

x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')
x_train /= np.max(x_train)
x_test /= np.max(x_test) 

from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,Dense, Dropout,Flatten

from keras.utils import to_categorical

y_train = to_categorical(y_train, num_of_classes)
y_test = to_categorical(y_test, num_of_classes) 

inp = Input(shape=(height, width, depth))
x=Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='glorot_normal')(inp)
x=MaxPooling2D((2,2),strides=(2,2))(x)
x=Dropout(.2)(x)
x=Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='glorot_normal')(x)
x=MaxPooling2D((2,2),strides=(2,2))(x)
x=Dropout(.2)(x)
x=Flatten()(x)
x=Dense(64,activation='relu')(x)
x=Dropout(0.2)(x)
x=Dense(32,activation='relu')(x)
x=Dropout(0.2)(x)
out=Dense(7,activation='softmax')(x)

model=Model(inputs=inp,outputs=out)


model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # reporting the accuracy

from keras.callbacks import ModelCheckpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


model.fit(x_train, y_train,batch_size=64 , epochs=25, validation_data=(x_test,y_test),callbacks=callbacks_list) 
#model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

model.load_weights("weights.best.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
scores = model.evaluate(x_train, y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))





from keras.models import model_from_json

model_json=model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("weights.best.hdf5")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(x_test,y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))




#########################################################################################################################
#Tuning Hyper Parameters also possible


#as an example will tune the number of epochs ans batch size
from keras.wrappers.scikit_learn import KerasClassifier



def create_model():
    x=Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='glorot_normal')(inp)
    x=MaxPooling2D((2,2),strides=(2,2))(x)
    x=Dropout(.2)(x)
    x=Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='glorot_normal')(x)
    x=MaxPooling2D((2,2),strides=(2,2))(x)
    x=Dropout(.2)(x)
    x=Flatten()(x)
    x=Dense(64,activation='relu')(x)
    x=Dropout(0.2)(x)
    x=Dense(32,activation='relu')(x)
    x=Dropout(0.2)(x)
    out=Dense(7,activation='softmax')(x)
    
    model=Model(inputs=inp,outputs=out)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model=KerasClassifier(build_fn=create_model)
epochs=[20,30,40,50]
batch_size=[32,64,128]
param_grid=dict(epochs=epochs,batch_size=batch_size)

from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(estimator=model,param_grid,scoring='accuracy')

grid.fit(img_data,label)

print(grid.best_score_,grid.best_params_)



#thus also optimizer, learning rate,momentum,no of hidden neurons can be tuned

    


















