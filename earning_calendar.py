import csv, datetime as dt, requests, polars as pl

APIKEY = "YOUR_KEY"  # or "demo"
DAYS = 10
CSV_URL = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey={APIKEY}"

today = dt.date.today()
end = today + dt.timedelta(days=DAYS)

rows = []
with requests.Session() as s:
    decoded = s.get(CSV_URL, timeout=30).content.decode("utf-8")
    for r in csv.DictReader(decoded.splitlines()):
        d = r.get("reportDate", "")
        if d and today <= dt.date.fromisoformat(d) <= end:
            rows.append({"date": d, "ticker": r.get("symbol", ""), "name": r.get("name", "")})

df = pl.DataFrame(rows).sort(["date", "ticker"])
print(df)

# optional export
# df.write_csv("earnings_next_10d.csv")
