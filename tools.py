from langchain.tools import tool
import yfinance as yf

@tool
def get_financial_data(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    return {
        "pe_ratio": info.get("trailingPE"),
        "eps": info.get("trailingEps"),
        "debt_to_equity": info.get("debtToEquity"),
        "roe": info.get("returnOnEquity"),
        "current_ratio": info.get("currentRatio"),
    }


@tool
def is_undervalued(financial_data: dict) -> str:
    # Assuming you’ve loaded a model previously
    X = [[
        financial_data['pe_ratio'],
        financial_data['eps'],
        financial_data['debt_to_equity'],
        financial_data['roe'],
        financial_data['current_ratio']
    ]]
    prediction = model.predict(X)[0]
    return "undervalued" if prediction == 1 else "not undervalued"