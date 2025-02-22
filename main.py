import string  ###caption are string

import numpy as np
import time
import os
from PIL import Image

from tqdm import tqdm
from pickle import dump ,load

import tensorflow as tf

import matplotlib.pyplot as plt
from keras.applications.xception import Xception,preprocess_input
from keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical,get_file
from keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input,Dense,LSTM,Embedding,Dropout

from tqdm.notebook import tqdm_notebook

tqdm_notebook.pandas()


  

def load_doc(filename):
    file = open(filename , "r")
    text = file.read()
    file.close()
    return text


def all_img_caption(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    descriptions = {}
    for caption in captions[:-1]:
        img_name , caption = caption.split('\t')
        if img_name[:-2] not in descriptions:
            descriptions[img_name[:-2]] = [caption]
        else:    
            descriptions[img_name[:-2]].append(caption)    
    return descriptions


#####################data cleaning #########################

def cleaning_text(captions):
    table = str.maketrans('','' ,string.punctuation)
    for img ,caps in captions.items():
         for i ,img_caption in enumerate(caps):
             img_caption.replace("-", " ")
             desc = img_caption.split()
             desc = [word.lower() for word in desc]
             desc = [word.translate(table) for word in desc] ##remove puntucation
             desc = [word for word in desc if(len(word)) > 1] ## remove dashes  eg:cat's   's remove
             desc = [word for word in desc if(word.isalpha())] ###remove all the numbers

             img_caption = ' '.join(desc)
             captions[img][i] = img_caption 

    return captions




def text_vocabulary(descriptions):
    vocab =set()

    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab

####save all descriptions

def save_description(descriptions,filename):
    lines = list()

    for key,desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' +desc)
        data = "\n".join(lines)
        file = open(filename, 'w')
        file.write(data)
        file.close()

    
    
dataset_images = "Flicker8k_Dataset"

dataset_text = "Flickr8k_text"
filename = os.path.join(dataset_text, "Flickr8k.lemma.token.txt")

descriptions = all_img_caption(filename)

print("length" ,len(descriptions))


clean_descriptions = cleaning_text(descriptions)
vocabulatory = text_vocabulary(clean_descriptions)
print("vocab" , len(vocabulatory))

save_description(clean_descriptions,"description.txt")



##downloading the model

def download_with_Retry(url,filename, max_retries = 3):
    for attempt in range(max_retries):
        try:
            return get_file(filename,url)

        except Exception as e:

            if attempt == max_retries -1:
                raise  e
            print(f"Downloaded attempt failed")
            time.sleep(3)

 
weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_path = download_with_Retry(weights_url, 'xception_weights.h5')
model = Xception(include_top=False, pooling='avg', weights=weights_path)


###extract the features

def extract_features(directory):

    features = {}
    valid_images = ['.jpg','jpeg','.png']
    for img in tqdm(os.listdir(directory)):
        ext = os.path.splitext(img)[1].lower()
        if ext not in valid_images:
            continue
 
        filename = directory + "/" + img
        image = Image.open(filename)
        image = image.resize((299,299))
        image = np.expand_dims(image, axis=0) ##height,width,channel
        image = image/127.5


        feature = model.predict(image)
        features[img] = feature


    return features

features = extract_features(dataset_images)
dump(features,open("features.p" , "wb"))


features = load(open("features.p","rb"))

###loading the data
def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    photo_present =[photo for photo in photos if os.path.exists(os.path.join(dataset_images,photo))]

    return photo_present



def load_clean_descriptions(filename,photos):
    file = load_doc(filename)
    descriptions = {}
    
    for line in file.split("\n"):
        words = line.split()
        if len(words)<1 :
            continue

        image, image_caption = words[0], words[1:]

        if image in photos:
            if image not in descriptions:
                descriptions[image] = []

            desc ='<start>'  + " ".join(image_caption) + '<end>'
            descriptions[image].append(desc)

    return descriptions        


####loading features


def load_features(photos):

    all_features = load(open("features.p", "rb"))
    features = {k:all_features[k] for k in photos}
    print(features)
    return features


filename = dataset_text + "/" + "Flickr_8k.testImages.txt"

train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("description.txt", train_imgs)

train_features =load_features(train_imgs)




#####convert dictionary into a list#####

def dict_to_list(descriptions):

    all_desc = []

    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]


    return all_desc




def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer =Tokenizer()
    tokenizer.fit_on_texts(desc_list)

    return tokenizer



#create tokenizer

tokenizer = create_tokenizer(train_descriptions)

dump(tokenizer ,open("tokenizer.p", "wb"))




vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)



##mqximum length of description    



def max_length(descriptions):

    desc_list = dict_to_list(descriptions)

    return max(len(d.split()) for d in desc_list)



###predicting word by word
max_length = max_length(train_descriptions)
print(max_length)




def data_generator(descriptions, features, tokenizer , max_length):
    def generator():
        while True:
            for key,description_list in descriptions.items():
                feature = features[key][0] #first feature    key for the image
                input_image , input_sequence ,output_word = create_sequence(tokenizer,max_length, descriptions ,features)
                 
                for i in range(len(input_image)):
                     yield {'input_1' :input_image[i], 'input_2':input_sequence[i] } ,output_word[i]
             
          
    output_signature = (
        {
            'input_1': tf.TensorSpec(shape=(2048,), dtype=tf.float32),
            'input_2': tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    )



    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    

    return dataset.batch(32)



def create_sequence(tokenizer, max_length , desc_list ,features):
    x1,x2, y = list() , list() , list()

    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq ,out_seq = seq[:i] ,seq[i]
            in_seq =pad_sequences([in_seq] , maxlen = max_length)[0]
            out_seq = to_categorical([out_seq], num_classes = vocab_size) [0]
            x1.append(features)
            x2.append(in_seq)
            y.append(out_seq)


        return np.array[x1] , np.array[x2] , np.array[y]    
    

dataset = data_generator(train_descriptions , features ,tokenizer ,max_length)


for (a,b) in dataset.take[1]:
    print(a['input_1'].shape. a['input_2'].shape , b.shape)
    break




##define model

def define_model(vocab_size,max_length):

    ##CNN model from 2048 to 256 nodes
    inputs1 = Input(shape =(2048,) , name='input_1')
    f1 = Dropout(0.5)(inputs1) #first layer    avoid overfiiting
    f2 = Dense(256, activation='relu')(f1)


    ##LSTM SEQUENCE MODEL

    inputs2  = Input(shape =(max_length,) , name='input_2')
    s1 = Embedding(vocab_size , 256, mask_zero=True)(input2)
    s2 = Dropout(0.5)(s1)
    s3 = LSTM(256)(s2)


    decoder1 = add([f2,s3])
    decoder2 = Dense(256 ,activation='relu' )(decoder1)
    outputs = Dense(vocab_size,activation='softmax')(decoder2)
    model = Model(inputs =[inputs1 , inputs2] ,outputs=outputs)


    model.complie(loss="categorical_crossentropy", optimizer='adam')
    print(model.summary())

    return model


model = define_model(vocab_size , max_length)

epochs = 10
step_per_epoch = 5

os.makedir('models')

for i in range(epochs):
    dataset = data_generator(train_descriptions , features ,tokenizer ,max_length)
    model.fit(dataset , epochs ,step_per_epoch ,verbose=1)
    model.save("models/model_" +str(i)+ ".h5" )




















 






