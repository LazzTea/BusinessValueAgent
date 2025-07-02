import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_ai_model(symbol=None):
    # Set up training set
    
    df = pd.read_csv("nasdaq_screener_large_companies.csv",nrows=500)
    tickers = df["Symbol"].dropna().unique().tolist()
    tickers = [t for t in tickers if t.isalpha() and len(t) <= 5]
    
    
    valid_tickers = []
    for ticker in tickers:
        try:
            yf.Ticker(ticker).fast_info
            valid_tickers.append(ticker)
        except Exception as e:
            print(f"{ticker} failed: {e}")
    
    tickers = valid_tickers
    # tickers = [
    #     "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NVDA", "INTC", "NFLX", "IBM",
    #     "JPM", "BAC", "WMT", "TGT", "UNH", "PFE", "XOM", "CVX", "BA", "DIS",
    #     "ORCL", "CRM", "COST", "NKE", "SBUX"
    # ]
    
    def get_financial_data(ticker):
        try:
            info = yf.Ticker(ticker).info
            pe_ratio = info.get("trailingPE")
            eps = info.get("trailingEps")
            debt_to_equity = info.get("debtToEquity")
            return_on_equity = info.get("returnOnEquity")
            current_ratio = info.get("currentRatio")

            # Check if any are None or not finite (inf, -inf, NaN)
            values = [pe_ratio, eps, debt_to_equity, return_on_equity, current_ratio]
            if any(v is None for v in values):
                return None
            if any(not np.isfinite(v) for v in values):
                return None

            return {
                "ticker": ticker,
                "pe_ratio": pe_ratio,
                "eps": eps,
                "debt_to_equity": debt_to_equity,
                "return_on_equity": return_on_equity,
                "current_ratio": current_ratio,
            }
        except Exception as e:
            print(f"Error getting data for {ticker}: {e}")
            return None


    data = [get_financial_data(t) for t in tickers]
    data = [d for d in data if d is not None]
    df = pd.DataFrame(data)

    # Determine the Future out come of the set
    for ticker in tickers:
        try:
            hist = yf.download(ticker, start="2023-01-01", end="2024-01-01",auto_adjust=True)
            if hist.empty:
                continue
            start_price = hist["Close"].iloc[0]
            end_price = hist["Close"].iloc[-1]
            pct_change = float((end_price.iloc[-1] - start_price.iloc[0]) / start_price.iloc[0])
            # print(f'Printing pct_change {pct_change} print is over')
            df.loc[df["ticker"] == ticker, "pct_change"] = pct_change
        except Exception as e:
            print(f"Error downloading history for {ticker}: {e}")

    # Label as undervalued if price increased more than 20%
    df["label"] = df["pct_change"].apply(lambda x: "undervalued" if x > 0.2 else "overvalued")
    
    print(f"Number of tickers collected: {len(data)}")
    

    # Drop any rows with missing values
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna()
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df.dropna(subset=["pe_ratio", "eps", "debt_to_equity", "return_on_equity", "current_ratio", "pct_change"], inplace=True)

    # Train the model on the given data set
    X = df[["pe_ratio", "eps", "debt_to_equity", "return_on_equity", "current_ratio"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42) # currently no max_depth
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    if(symbol is None):
        return model
    
    ticker = yf.Ticker(symbol)

    # Extract key metrics
    info = ticker.info

    data = {
        "pe_ratio": info.get("trailingPE", None),
        "eps": info.get("trailingEps", None),
        "debt_to_equity": info.get("debtToEquity", None),
        "return_on_equity": info.get("returnOnEquity", None),
        "current_ratio": info.get("currentRatio", None)
    }
    
    data = pd.DataFrame([data])
    data = data.replace([float("inf"), float("-inf")], pd.NA)
    data = data.dropna()
    data.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    data.dropna(subset=["pe_ratio", "eps", "debt_to_equity", "return_on_equity", "current_ratio"], inplace=True)
    
    if(data.empty):
        return False, "Data on business is incomplete"
    
    return data, model.predict(data)