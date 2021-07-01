from flask import Flask, render_template, request
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

appl = Flask(__name__)
tfVectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
loaded_model= pickle.load(open('model.pkl','rb'))
df = pd.read_csv('news_dataset.csv')
dataframe= df.dropna()
content = dataframe['content']
label = dataframe['label']
content_train, content_test,label_train,label_test= train_test_split(content,label,test_size=0.2,random_state=0)

def fake_news_det(content_news):
    tfid_content_train = tfVectorizer.fit_transform(content_train)
    tfid_content_test  = tfVectorizer.transform(content_test)
    input_data = [content_news]
    vectorize_input_data = tfVectorizer.transform(input_data)
    predicate = loaded_model.predict(vectorize_input_data)
    return predicate

@appl.route('/')
def home():
    return render_template('home.html')

@appl.route('/predict',methods= ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template ('home.html', prediction=pred)
    else:
        return render_template ('home.html', prediction="Something Went Wrong")

if __name__ == '__main__':
    appl.run(debug=True)
