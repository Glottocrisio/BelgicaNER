import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipelinefrom 
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
import sparknlp



from sparknlp.training import CoNLL

def Ner():
    spark = sparknlp.start()
    print("Spark NLP version: ", sparknlp.version())
    print("Apache Spark version: ", spark.version)
    training_data = CoNLL().readDataset(spark, './fr.train')
    training_data.show()