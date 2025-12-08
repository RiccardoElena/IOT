from massive import RESTClient

client = RESTClient("***REMOVED***")

aggs = []
for a in client.list_aggs(
    "AAPL",
    1,
    "hour",
    "2023-01-09",
    "2023-02-10",
    adjusted="true",
    sort="asc",
    limit=120,
):
    aggs.append(a)

print(aggs)
