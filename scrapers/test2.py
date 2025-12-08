from massive import RESTClient

client = RESTClient("bRRqs2oIShp7Y8lRF99yYu234yFkYSe0")

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
