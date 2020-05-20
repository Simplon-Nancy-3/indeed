import pandas as pd
def insertDFtoMongo(df,collection):
    from pymongo import MongoClient
    client = MongoClient('localhost', 27017)
    collection = client.IndeedProject[collection]
    collection.insert_many(df.to_dict('records'))

def getMongoData():
    from pymongo import MongoClient
    client = MongoClient('localhost', 27017)
    collection = client.IndeedProject.IndeedProjectCollection
    df =  pd.DataFrame(list(collection.find()))
    return df

