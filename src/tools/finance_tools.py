import json
import yfinance as yf

def finance_metrics_tool(ticker_symbol: str) -> str:
    """
    Accepts a stock ticker symbol.
    Retrieves core financial attributes via yfinance and formats into a compact JSON string.
    """
    print(f"Fetching financial data for ticker: {ticker_symbol}...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        financials = ticker.financials
        
        metrics = {
            "ticker": ticker_symbol,
            "company_name": info.get("longName"),
            "current_price": info.get("currentPrice"),
            "pe_ratio": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
            "market_cap": info.get("marketCap"),
            # Convert Timestamp keys to string format for JSON serialization
            "revenue_history": {str(k): v for k, v in financials.loc['Total Revenue'].to_dict().items()} if 'Total Revenue' in financials.index else {},
            "net_income_history": {str(k): v for k, v in financials.loc['Net Income'].to_dict().items()} if 'Net Income' in financials.index else {}
        }
        
        return json.dumps(metrics, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to retrieve data for {ticker_symbol}: {str(e)}"})

if __name__ == "__main__":
    test_ticker = "NVDA"
    result = finance_metrics_tool(test_ticker)
    print("Financial Tool Output:")
    print(result)