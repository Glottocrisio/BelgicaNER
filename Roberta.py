from transformers import RobertaTokenizer, RobertaForTokenClassification
import torch

def Ner():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForTokenClassification.from_pretrained('roberta-base')

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
