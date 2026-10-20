from sickle import Sickle

sickle = Sickle('http://apps.who.int/iris/oai/request')
records = sickle.ListRecords(metadataPrefix='oai_dc', set='WHO')
for i, record in enumerate(records):
    if i >= 5:  # limiting to first 5 records
        break
    print(record.metadata['title'])