import streamlit as st
#import torch
from transformers import BertTokenizer, AutoModelForSequenceClassification
import numpy as np

def Predict_Page():
    st.title("Headlines these days trying to be funny aren't they?")

    st.write("""
            Lets see if they are or not?
            """)
    
    headline = st.text_input("Headline", value = "")

    if len(headline) > 4:
        Tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")

        Model = AutoModelForSequenceClassification.from_pretrained("BlindMan820/Sarcastic-News-Headlines")  

        Text_Encode = Tokenizer.encode_plus(headline,
                        truncation = True,
                        padding = True,
                        return_tensors = "pt")
        input_ids = Text_Encode['input_ids']
        attention_mask = Text_Encode["attention_mask"]
        Output = Model(input_ids, attention_mask)
        print(Output)
        preds = int(np.argmax(Output.logits.detach()))

        if preds == 1:
            st.markdown("![Bazzinga](https://c.tenor.com/0d_WxNIZ6hEAAAAC/big-bang-theory-sheldon-cooper.gif)")
        
        else:
            st.markdown("![:((](https://c.tenor.com/b5jxHCnZHfkAAAAM/the-big-bang-theory-sheldon.gif)")
