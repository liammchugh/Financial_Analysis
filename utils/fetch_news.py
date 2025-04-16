import os
import requests
from datetime import datetime, timedelta
from utils.llm_prompt import generate_response

from typing import Optional

def fetch_news(service: str, query: str, to_date: datetime, lookback_days: int = 14, limit: int = 6) -> str:
    """
    Fetches recent news for the given ticker from NewsAPI.org.
    
    Args:
        service (str): Service to fetch news from (e.g., "newsapi", "finlight").
        query (str): The query (ticker, keywords) to search for.
        lookback_days (int): How many days back to search for news.
        limit (int): Number of news articles to fetch.
        
    Returns:
        str: A formatted string containing news article headlines and descriptions.
    """
    
    # Build the date range for the query
    from_date = to_date - timedelta(days=lookback_days)
    

    if service == "newsapi":
        newsapi_key = os.getenv("newsapi_key")
        if not newsapi_key:
            raise ValueError("Please set the NEWSAPI_KEY environment variable with your NewsAPI.org key.")

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,  # search by ticker or company name
            "from": from_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": to_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": limit,
            "apiKey": newsapi_key,
        }
        response = requests.get(url, params=params)
        data = response.json()
    elif service == "finlight":
        url = f"https://api.finlight.me/v1/articles?query={query}&from={from_date.strftime('%Y-%m-%dT%H:%M:%SZ')}&to={to_date.strftime('%Y-%m-%dT%H:%M:%SZ')}&pageSize={limit}"
        finlight_api_key = os.getenv("FINLIGHT_API_KEY")
        if not finlight_api_key:
            raise ValueError("Please set the FINLIGHT_API_KEY environment variable with your Finlight API key.")
        
        headers = {
            "accept": "application/json",
            "X-API-KEY": finlight_api_key,
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an exception for HTTP errors
        data = response.json()
    else:
        raise ValueError(f"Unsupported service: {service}")

    # Check for errors from the API
    if data.get("status") != "ok":
        raise ValueError("NewsAPI returned an error: " + data.get("message", "Unknown error"))

    articles = data.get("articles", [])
    if not articles:
        return "No news articles found."

    # Format the news text
    news_text = "\nrelated news:\n"
    for i, article in enumerate(articles, start=1):
        title = article.get("title", "No title")
        published_at = article.get("publishedAt", "No date")
        description = article.get("description", "No description provided")
        url_link = article.get("url", "")
        
        news_text += f"\nHeadline {i}: {title}\n"
        news_text += f"Published: {published_at}\n"
        news_text += f"Description: {description}\n"
        news_text += f"URL: {url_link}\n"

    return news_text

def find_more_news(news: str, ticker: str, service: str, to_date: datetime, lookback_days: int = 14, limit: int = 6) -> Optional[str]:
    """
    Given a string of news headlines, generate a targeted search query and fetch additional news articles.
    """
    # given news headings, prompt gpt4 to generate a targeted search query
    if not "No recent news articles found." in news:
        prompt = f"Given the following news headlines, generate a targeted search query relevant to the future performance of {ticker} given these headlines. Using the real company name, generate an extremely concise query of only a few keywords, with minimal punctuation:\n\n"
        prompt += news
        search_query = generate_response(prompt)
        if "Search Query:" in search_query:
            search_query = search_query.replace("Search Query:", "").strip()
        print("\nGenerated Search Query: ", search_query)

        # Search news again using the generated search query
        try:
            additional_news = fetch_news(service, search_query, to_date, lookback_days, limit)
            if additional_news == "No news articles found.":
                print("No additional news found for generated query. Retrying with simplified query.")
                prompt = "Simplify the following search query to improve search results. Be extremely concise, ie less than 3 words:\n\n"
                prompt += search_query
                simplified_query = generate_response(prompt)
                if "Search Query:" in simplified_query:
                    simplified_query = simplified_query.replace("Search Query:", "").strip()
                print("\nSimplified Search Query: ", simplified_query)
                additional_news = fetch_news(service, simplified_query, to_date, lookback_days, limit)
            print("\nAdditional News Articles:")
            print(additional_news)
        #     # Append additional news to the complete news report
        #     complete_news_report = news + "\n\nAdditional News:\n" + additional_news
        #     print("\nComplete News Report:")
        #     print(complete_news_report)
        except Exception as e:
            print(f"Error fetching additional news: {e}")
            return None
    else:
        print("No recent news articles found to generate a search query.")
        return None
    
    return news + additional_news


if __name__ == "__main__":
    from llm_prompt import generate_response

    # Example usage
    ticker = "AAPL"  # Apple Inc.
    lookback_days = 14  # Look back two weeka
    limit = 6  # Limit to 5 articles
    service = "newsapi"  # Choose between "newsapi" and "finlight"
    to_date = datetime(2025, 4, 1)  # Use a specific date in 2023
 
    try:
        news = fetch_news(service, ticker, to_date, lookback_days, limit)
        print(news)
    except Exception as e:
        print(f"Error fetching news: {e}")
    
    all_news = find_more_news(news, ticker, service, to_date, lookback_days, limit)
    print(all_news)
