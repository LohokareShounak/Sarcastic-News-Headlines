import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import streamlit as st
import numpy
def Load_Model():
    file_path = r"Transformers/Models/Tensorflow"
    Model = TFDistilBertForSequenceClassification.from_pretrained(file_path)
    return Model

def Predict_Page():

    st.title("Headlines these days trying to be funny aren't they?")

    st.write("""
            Lets see if they are or not?
            """)
    
    headline = st.text_input("Headline", value = "")

    if len(headline) > 4:
        Model = Load_Model()

        Tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        Text_Encode = Tokenizer.encode(headline,
                        truncation = True,
                        padding = True,
                        return_tensors = "tf")
        
        Output = Model.predict(Text_Encode)[0]
        print(Output)
        preds = int(tf.round(tf.nn.softmax(Output).numpy()[1]))

        if preds == 1:
            st.markdown("![Bazzinga](https://c.tenor.com/0d_WxNIZ6hEAAAAC/big-bang-theory-sheldon-cooper.gif)")
        
        else:
            st.markdown("![:((](https://c.tenor.com/b5jxHCnZHfkAAAAM/the-big-bang-theory-sheldon.gif)")

