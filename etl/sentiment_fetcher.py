import feedparser
import streamlit as st
import requests

class SocialSentimentFetcher:
    def __init__(self):
        # RSS Sources
        self.rss_feeds = [
            "https://www.marketwatch.com/rss/marketupdate",
            "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
            "https://search.cnbc.com/rs/search/view.xml?partnerId=2000&keywords={ticker}&searchtype=news"
        ]
        
        # PullPush API Base
        self.pullpush_base = "https://api.pullpush.io/reddit/search"

    def fetch_reddit_posts(self, ticker, limit=15):
        """Fetches recent posts and comments about a ticker from PullPush (Reddit archive)."""
        posts = []
        
        # We search both submissions and comments
        endpoints = ["submission", "comment"]
        
        for ep in endpoints:
            try:
                # Search across popular subreddits by querying PullPush
                # size=limit, q=ticker, sort=desc (newest)
                url = f"{self.pullpush_base}/{ep}/?q={ticker}&size={limit}&sort=desc"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json().get("data", [])
                    for item in data:
                        subreddit = item.get("subreddit", "unknown")
                        title = item.get("title", "") # only in submissions
                        body = item.get("selftext") or item.get("body") or ""
                        
                        content = f"Source: Reddit (r/{subreddit})\n"
                        if title: content += f"Title: {title}\n"
                        content += f"Content: {body[:500]}"
                        posts.append(content)
            except Exception as e:
                continue # Silent fail for background fetching
                
        return posts

    def fetch_rss_news(self, ticker):
        """Fetches news from various RSS feeds about a ticker."""
        news_items = []
        for feed_url in self.rss_feeds:
            formatted_url = feed_url.format(ticker=ticker)
            try:
                feed = feedparser.parse(formatted_url)
                for entry in feed.entries:
                    # Filter for relevance if ticker is in title or summary
                    if ticker.upper() in entry.title.upper() or ticker.upper() in entry.get('summary', '').upper():
                        news_items.append(f"Source: {feed_url}\nTitle: {entry.title}\nSummary: {entry.get('summary', '')[:500]}")
            except Exception as e:
                continue
        return news_items

    def get_all_context(self, ticker):
        """Aggregates all textual context for a ticker."""
        reddit_posts = self.fetch_reddit_posts(ticker)
        rss_news = self.fetch_rss_news(ticker)
        return reddit_posts + rss_news
