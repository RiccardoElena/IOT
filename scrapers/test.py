from massive import RESTClient

client = RESTClient("***REMOVED***")

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