import spacy
from spacy import displacy
from spacy.lang.fr.examples import sentences
from pprint import pprint
from collections import Counter
from IPython.display import display

def Ner():

    nlp = spacy.load("fr_dep_news_trf") #fr_dep_news_trf  en_core_news_sm
  
    ocr = "llorﬂeullure,IpIcullura,lvleuIlure Conférences d'aujourd'hui dimanche: M.~\RCl.l\'ELLE. - Cercle Horticole. -A 4 h. 112. aux Ecoles des Haies par M. G. Ghslmagne tombola générale extraor-§_*?d'i*I1uire*comportant plus de 60 lots d'oi~U, MIUNT-IGNIES-SUR-SAM.BP.E. - Ger-;qš9 Horticola du Centre. -- A 4 h. 1/2,,pçr M. A. Evrard, au local. Sujet: Cul-¿.1uμ'o et. taille du poírier. - Les engrais_-.scliinúques en culture niaraichèrc.,_,¿.Princlpn.ux avantages a,ccox'u:*›s gratuitement aux membres du cercle: Partici-Hpatüon aux tombolns orgmilsées après haucune des conférences. Participation,aux tcrmbolas très inlportanœs qui ont lieu lors des conférences fédérales. Disegnons å ﬂeurs. ,trîbution djassortiments de graines potagères. Importante réduction sur le prix yﬂdes coupons lors des excursions organisées par .le cercle. neux, violoniste prix du Conservatoire de Bruxelles; Öonsette, violoncelliste, 1 Yrix d`exccllence de l'.-\académie de Chareroi; Mlle Flore Duchène, pianiste. 1 Ê-›rix_ d'excellence de l`Académie de Char*eroμ Première partie: 1. Le Rêve, par l`Orphécm (Paillard) _: 2. Les Forgerons de l`Avenir, id. (Sourilaslç 3. Fantaisie-Ballet, par M. Doueu (De Bériot); -5. Air de Si-gurd, par M. Boven (Reyer); 5. Air de Louise, par Mlle Duchesne (Charpentier);6. Clmnsc-11 Napolitaine, par M. Gonsette (César Casella); 7. Prologue de Paillas-se, par M. Druine (Cavallo). Deuxième partie: 1. Thais (méditatinn), par M. Gonsette (Mass.ene*t.); 2. Gﬁavotte, id. (Poper); 3. Lohengrin, par M. Boven agner): -5. Air des bijoux, par Mlle Duchesne (Fo.ust.); 5. Légende, gar M. Doneux Wienlawski); 5. Chant indou, par M. Druine (Bc-mberg); 7. Trio de Jérusalem, par Mlle Duchesne, MM. Boven et Druine (Verdi). Après le concert, bal ensymnhonie. "

    ocr2= "Apple est créée le 1er avril 1976 dans le garage de la maison d'enfance de Steve Jobs à Los Altos en Californie par Steve Jobs, Steve Wozniak et Ronald Wayne14, puis constituée sous forme de société le 3 janvier 1977 à l'origine sous le nom d'Apple Computer, mais pour ses 30 ans et pour refléter la diversification de ses produits, le mot « computer » est retiré le 9 janvier 2015."

    doc = nlp(ocr)
    print(doc.text)
    for token in doc:
        print(token.text,token.pos_, token.dep_)
    pprint([(X.text, X.label_) for X in doc.ents])
    displacy.render(doc, style='ent')

    labels = [x.label_ for x in doc.ents]
    Counter(labels)

    dict([(str(x), x.label_) for x in doc.ents])

    #displacy.render(str(doc), style='ent', manual = True)
    #displacy.render(nlp(str(doc)), style='ent', jupyter = True, options = {'distance': 120})


    # Download a spacy model for processing French
    #nlp = spacy.load("en_core_news_md")

     #Process a sentence using the spacy model
    #doc = nlp("en ligue avec les vues de lla Saskalclicxvan. Dans un lélégruinme qi|'il a envoyé il llion. Gorilon, lil ilenianile la date ile la niisc à exé-_i-.ulion de ce ilrojvl, car il voudrait *i|ui- ces gens fussent sur la ll.-rrc li-inps pour euseniciiccr dés ce prin temps. Il ajonlg *( u`il éluilie la for- mation il`uu lnirlca-11 non-politique pour diriger l`cx_0ilc des sans-trio vail ui*l›ainS vers la ferme. ll ilit avoir eu des couvi-rszitions avec les autorités nmuicipales ilss villes el les sociétés s`orciip:int des chôimcurs, avec la i›onelusio1i de part eld”aulrc que l'on peut établir 2,000 familles")

    ## Display the entities found by the model, and the type of each.
    #print('{:<12}  {:}\n'.format('Entity', 'Type'))

    ## For each entity found...
    for ent in doc.ents:
    
        #Print the entity text `ent.text` and its label `ent.label_`.
        print('{:<12}  {:}'.format(ent.text, ent.label_))
