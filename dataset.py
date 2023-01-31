import pymongo
# connect to mongodb
def connectToMongodb():
    try:
        client = pymongo.MongoClient("mongodb://root:Pictalk2023!@mongodb.apps.asidiras.dev/",27017,
                                        serverSelectionTimeoutMS=1000)
        client.server_info() # force connection on a request as the
                            # connect=True parameter of MongoClient seems
                            # to be useless here 
    except pymongo.errors.ServerSelectionTimeoutError as err:
    # do whatever you need
        print(err)
    return client
def readAllDocumentsAndWriteThemToFile(conn):
    db = conn.pictohub
    collection = db.pictograms
    print(collection.count_documents({}))
    cursor = collection.find()
    for document in cursor:
        print(index)
        if (document['keywords']):
            emptyList = []
            for keyword in document['keywords']['fr']:
                emptyList.append(keyword['keyword'])
                if 'synonymes' in keyword:
                    for synonym in keyword['synonymes']:
                        emptyList.append(synonym)
            # remove duplicates in list
            writeListToFile(removeDuplicatesInList(emptyList))
            
def removeDuplicatesInList(listWithDuplicates):
    return list(set(listWithDuplicates))
             
def writeListToFile(listToWrite):
    with open('test.txt', 'a') as f:
        f.write('\n'.join(listToWrite))

           
readAllDocumentsAndWriteThemToFile(connectToMongodb())