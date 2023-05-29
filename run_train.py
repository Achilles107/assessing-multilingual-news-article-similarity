# ## Imports

from tqdm import tqdm
import json
import pandas as pd
from numpy import NaN, array
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense,  Input, concatenate


# Crawl data from json files

directory=pd.read_csv("semeval-2022_task8_train-data_batch.csv")

df_data={
    'pair_id':[],
    'title1':[],
    'text1':[],
    'title2':[],
    'text2':[],
    'Overall':[],
    'Language1':[],
    'Language2':[]
}

for idx, i in tqdm(enumerate(directory['pair_id'])):
    id1, id2 = i.split('_')
    try:
        file1=json.load(open(f"output_dir\output_dir\{str(id1)[-2:]}\{id1}.json"))
        file2=json.load(open(f"output_dir\output_dir\{str(id2)[-2:]}\{id2}.json"))
        df_data['pair_id'].append(i)
        
        if file1['title'].strip() == '':
            df_data['title1'].append(NaN)
        else:
            df_data['title1'].append(file1['title'])

        if file1['text'].strip() == '':
            df_data['text1'].append(NaN)
        else:
            df_data['text1'].append(file1['text'])

        if str(file2['title'].strip()) == '':
            df_data['title2'].append(NaN)
        else:
            df_data['title2'].append(file2['title'])

        if file2['text'].strip() == '':
            df_data['text2'].append(NaN)
        else:
            df_data['text2'].append(file2['text'])
        
        df_data['Overall'].append(directory['Overall'][idx])
        df_data['Language1'].append(directory['url1_lang'][idx])
        df_data['Language2'].append(directory['url2_lang'][idx])
        
    except Exception:
        pass

df=pd.DataFrame(df_data)
df.dropna(inplace=True)
df=df.reset_index()
df=df.drop(['index'], axis=1)
df.to_csv('output.csv')

# Preprocessing data



df=pd.read_csv('output.csv')

stop_words_eng = set(stopwords.words('english'))
stop_words_fr = set(stopwords.words('french'))
stop_words_es = set(stopwords.words('spanish'))
stop_words_tr = set(stopwords.words('turkish'))
stop_words_de = set(stopwords.words('german'))
stop_words_ar = set(stopwords.words('arabic'))
stopwords={
    'en':stop_words_eng,
    'fr':stop_words_fr,
    'es':stop_words_es,
    'tr':stop_words_tr,
    'ar':stop_words_ar,
    'de':stop_words_de
}

for i in tqdm(range(len(df))):
    if df['Language1'][i]=='pl' or df['Language2'][i]=='pl':
        continue
    word_tokens_title1 = df['title1'][i]
    word_tokens_text1 = df['text1'][i]
    word_tokens_title2 = df['title2'][i]
    word_tokens_text2 = df['text2'][i]

    df['text1'][i] = ' '.join([w.lower() for w in word_tokens_title1.split() if not w.lower() in stopwords[df['Language1'][i]]])
    df['text1'][i] +=' '+' '.join([w.lower() for w in word_tokens_text1.split() if not w.lower() in stopwords[df['Language1'][i]]])
    df['text2'][i] = ' '.join([w.lower() for w in word_tokens_title2.split() if not w.lower() in stopwords[df['Language2'][i]]])
    df['text2'][i] +=' '+' '.join([w.lower() for w in word_tokens_text2.split() if not w.lower() in stopwords[df['Language2'][i]]])

df=df.drop(['Unnamed: 0', 'title1', 'title2'], axis=1)

df=df.rename(columns={
    'text1': 'article1',
    'text2': 'article2'
})

df.to_csv('preprocessed.csv')

## Implementing model
train_data = pd.read_csv('preprocessed.csv')
train_data = train_data.dropna()
train_data = train_data.reset_index(drop=True)

X_train, y_train,  = [], []

for idx in range(len(train_data)):
  X_train.append(train_data['article1'][idx])  # Fetching article 1
  X_train.append(train_data['article2'][idx]) # Fetching article 2
  y_train.append(train_data['Overall'][idx])

#Tokenizing
def tokenizer(data_train):
    tf_vec = TfidfVectorizer(min_df =4) # applying Term frequency Inverse Document Frequency
    tf_vec.fit(data_train) # Fitting the train data
    data_train = tf_vec.transform(data_train)
    return data_train

X_train = tokenizer(X_train)

# Extracting Features
def extract_feature(data_train, X_train, y_train):
    X_train_art1 = [] # list of article 1 
    X_train_art2 = [] # list of article 2

    X_train1 = []
    X_train2 = []
    for idx in range(len(data_train)):
        art1 = 2*idx
        art2 = 2*idx+1
        X_train_art1.append(X_train[art1])
        X_train_art2.append(X_train[art2])

    for idx in range(len(data_train)):
        X_train1.append(array(csr_matrix.todense(X_train_art1[idx])))
        X_train2.append(array(csr_matrix.todense(X_train_art2[idx])))

    y_train = array(y_train)

    X_train1 = array(X_train1)
    X_train2 = array(X_train2)
    
    X_train1 = X_train1.reshape(len(train_data),X_train1.shape[2])
    X_train2 = X_train2.reshape(len(train_data),X_train2.shape[2])
    return X_train1, X_train2, y_train

X_train1, X_train2, y_train = extract_feature(train_data, X_train,y_train)

def myMLP(X_train1, X_train2, y_train):
    input_art1 = Input(shape=(X_train1.shape[1], ))
    dense_art1 = Dense(1024, )(input_art1)

    input_art2 = Input(shape=(X_train2.shape[1], ))
    dense_art2 = Dense(1024, )(input_art2)

    merge_inputs = concatenate([dense_art1, dense_art2])

    dense_3  = Dense(256,)(merge_inputs)
    dense_4  = Dense(50,)(dense_3)
    output_layer = Dense(1,)(dense_4)

    model = Model(inputs=[input_art1, input_art2], outputs=output_layer)
    model.compile(loss='mse', optimizer = Adam()) # passing adam optimizer and mean squared error loss function
    print(model.summary())

    model.fit([X_train1, X_train2], y_train, epochs=10, validation_split=0.2) # validation is 20%
    
    model.save("Saved_model") # Saving the model

myMLP(X_train1=X_train1, X_train2=X_train2, y_train=y_train) #Calling the model


