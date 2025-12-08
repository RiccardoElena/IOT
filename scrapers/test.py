from massive import RESTClient

client = RESTClient("bRRqs2oIShp7Y8lRF99yYu234yFkYSe0")

tickers = []
for t in client.list_tickers(
  market="stocks",
  active="true",
  order="asc",
  limit="100",
  sort="ticker",
  ):
    tickers.append(t)

print(tickers)