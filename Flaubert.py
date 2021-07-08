from transformers import XLMTokenizer, XLMWithLMHeadModel
import torch

def Ner():
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
    model = XLMWithLMHeadModel.from_pretrained('xlm-mlm-en-2048')

    inputs = tokenizer("The capital of France is <special1>.", return_tensors="pt")
    labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
