from tensorflow.keras.preproccessing.text import Tokenizer

from tensorflow.keras.preproccessing.sequence import pad_sequence

from keras.applications.xception import Xception


from keras.models import load_model


from pickle import load
import numpy as np

from PIL import Image

import argparse
from tensorflow.keras.layers import Input ,Dense,LSTM,Embedding , Dropout ,add
from tensorflow.keras.models import saved_model ,Model


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image',required=True, help='Image')
args = vars(ap.parse_args())
img_path = args['Image']



def extract_features(filename, model):
    try:
        image = Image.open(filename)


    except:

        print("error")

    
    image = image.resize((299,299))

    image = np.array(image)

    if image.shape[2] == 4 :
        image = image[..., 3]
    

    image = np.expand_dims(image, axis=0)
    image = image /127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature



def word_for_id(integer ,tokenizer):
    for word , index in tokenizer.word_index.items():
        if index == integer:
            return word
     
    return None

#generate next word
def generate_desc(model,tokenizer,photo,max_length):
    in_text = 'start'
    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequence([sequence], max_length=max_length)


        pred = model.predict([photo,sequence] ,verbose=0)
        pred = np.argmax(pred)


        word = word_for_id(pred , tokenizer)

        if word is None:
            break

        in_text += ' ' +word

        if word  == 'end':
            break


    return in_text    




def define_model(vocab_size,max_length):

    ##CNN model from 2048 to 256 nodes
    inputs1 = Input(shape =(2048,) , name='input_1')
    f1 = Dropout(0.5)(inputs1) #first layer    avoid overfiiting
    f2 = Dense(256, activation='relu')(f1)


    ##LSTM SEQUENCE MODEL

    inputs2  = Input(shape =(max_length,) , name='input_2')
    s1 = Embedding(vocab_size , 256, mask_zero=True)(inputs2)
    s2 = Dropout(0.5)(s1)
    s3 = LSTM(256)(s2)


    decoder1 = add([f2,s3])
    decoder2 = Dense(256 ,activation='relu' )(decoder1)
    outputs = Dense(vocab_size,activation='softmax')(decoder2)
    model = Model(inputs =[inputs1 , inputs2] ,outputs=outputs)


    model.complie(loss="categorical_crossentropy", optimizer='adam')
    print(model.summary())

    return model



max_length = 32
tokenizer = load(open('tokenizer.p','rb'))
vocab_size = len(tokenizer.word_index) + 1

model =define_model(vocab_size ,max_length)
model.load_weights("models/model_9.h5")
xception_model = Xception(include_top =False , pooling='avg')



photo = extract_features(img_path ,xception_model)
img = Image.open(img_path)


description  = generate_desc(model,tokenizer , photo, max_length)
print(description)


