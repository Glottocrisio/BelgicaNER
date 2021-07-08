import torch


def Ner():
    camembert = torch.hub.load('pytorch/fairseq', 'camembert-base')
    camembert.eval() 


    #from  transformers import CamembertForTokenClassification
    #camembert = CamembertForTokenClassification.from_pretrained('Jean-Baptiste/camembert-ner-with-dates')
    #camembert.eval()

    # Extract the last layer's features
    line = "J'aime le camembert !"
    tokens = camembert.encode(line)
    last_layer_features = camembert.extract_features(tokens)
    assert last_layer_features.size() == torch.Size([1, 10, 768])

    # Extract all layer's features (layer 0 is the embedding layer)
    all_layers = camembert.extract_features(tokens, return_all_hiddens=True)
    assert len(all_layers) == 13
    assert torch.all(all_layers[-1] == last_layer_features)
    print(all_layers)

def Ner2():
    from transformers import CamembertTokenizer, CamembertForTokenClassification
    import torch

    tokenizerC = CamembertTokenizer.from_pretrained('Jean-Baptiste/camembert-ner-with-dates')
    modelC = CamembertForTokenClassification.from_pretrained('Jean-Baptiste/camembert-ner-with-dates')

    from transformers import Pipeline

    nlp = Pipeline(model =modelC, tokenizer=tokenizerC)
    doc = nlp("Apple est créée le 1er avril 1976 dans le garage de la maison d'enfance de Steve Jobs à Los Altos en Californie par Steve Jobs, Steve Wozniak et Ronald Wayne14, puis constituée sous forme de société le 3 janvier 1977 à l'origine sous le nom d'Apple Computer, mais pour ses 30 ans et pour refléter la diversification de ses produits, le mot « computer » est retiré le 9 janvier 2015.")

    #inputs = tokenizer("J'aime le camembert !", return_tensors="pt")
    #labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1

    #outputs = model(**inputs, labels=labels)
    #loss = outputs.loss
    #logits = outputs.logits
    print(doc)