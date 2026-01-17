"""
Data Collection Script - Tabi vs Samba Analysis
Uses SerpAPI to collect Reddit posts mentioning each shoe
"""

import os
from serpapi import GoogleSearch
import pandas as pd
from datetime import datetime
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RedditDataCollector:
    """Collect Reddit posts about fashion items using SerpAPI"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('SERPAPI_KEY')
        if not self.api_key:
            raise ValueError("SerpAPI key not found! Set SERPAPI_KEY in .env file")
    
    def search_reddit(self, query, subreddit=None, num_results=100):
        """
        Search Reddit using SerpAPI
        
        Args:
            query: Search term (e.g., "tabi boots")
            subreddit: Optional specific subreddit (e.g., "malefashionadvice")
            num_results: Number of results to retrieve
        
        Returns:
            List of result dictionaries
        """
        search_query = f"{query} site:reddit.com"
        if subreddit:
            search_query = f"{query} site:reddit.com/r/{subreddit}"
        
        params = {
            "q": search_query,
            "api_key": self.api_key,
            "num": min(num_results, 100),  # SerpAPI max per request
            "engine": "google"
        }
        
        print(f"Searching: {search_query}")
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            organic_results = results.get("organic_results", [])
            print(f"  Found {len(organic_results)} results")
            
            return organic_results
            
        except Exception as e:
            print(f"  Error: {e}")
            return []
    
    def extract_reddit_data(self, results, shoe_type):
        """
        Extract relevant fields from search results
        
        Args:
            results: List of search results from SerpAPI
            shoe_type: 'tabi' or 'samba'
        
        Returns:
            DataFrame with extracted data
        """
        data = []
        
        for result in results:
            # Extract Reddit-specific info from URL
            url = result.get('link', '')
            
            # Skip if not actually a Reddit post
            if 'reddit.com/r/' not in url:
                continue
            
            # Parse subreddit from URL
            try:
                subreddit = url.split('/r/')[1].split('/')[0]
            except:
                subreddit = 'unknown'
            
            data.append({
                'url': url,
                'title': result.get('title', ''),
                'snippet': result.get('snippet', ''),
                'subreddit': subreddit,
                'shoe_type': shoe_type,
                'collected_at': datetime.now(),
                'position': result.get('position', 0)
            })
        
        return pd.DataFrame(data)
    
    def collect_shoe_data(self, shoe_keywords, subreddits, results_per_search=100):
        """
        Collect data for specific shoe across multiple subreddits
        
        Args:
            shoe_keywords: Dict like {'tabi': ['tabi', 'margiela tabi'], ...}
            subreddits: List of subreddit names
            results_per_search: Results per query
        
        Returns:
            DataFrame with all collected data
        """
        all_data = []
        
        for shoe_type, keywords in shoe_keywords.items():
            print(f"\n{'='*60}")
            print(f"Collecting data for: {shoe_type.upper()}")
            print(f"{'='*60}")
            
            for keyword in keywords:
                for subreddit in subreddits:
                    # Search this keyword in this subreddit
                    results = self.search_reddit(
                        query=keyword,
                        subreddit=subreddit,
                        num_results=results_per_search
                    )
                    
                    # Extract data
                    df = self.extract_reddit_data(results, shoe_type)
                    all_data.append(df)
                    
                    # Rate limiting
                    time.sleep(2)
        
        # Combine all results
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates (same URL found multiple times)
        final_df = final_df.drop_duplicates(subset=['url'])
        
        return final_df


def main():
    """Main execution function"""
    
    # Initialize collector
    collector = RedditDataCollector()
    
    # Define search parameters
    SHOE_KEYWORDS = {
        'tabi': [
            'tabi boots',
            'margiela tabi',
            'maison margiela tabi'
        ],
        'samba': [
            'adidas samba',
            'adidas sambas',
            'samba sneakers'
        ]
    }
    
    SUBREDDITS = [
        'malefashionadvice',
        'sneakers',
        'streetwear',
        'femalefashionadvice'
    ]
    
    # Collect data
    print("\nStarting data collection...")
    print(f"Shoes: {list(SHOE_KEYWORDS.keys())}")
    print(f"Subreddits: {SUBREDDITS}")
    print(f"Estimated API calls: {len(SHOE_KEYWORDS) * len(SHOE_KEYWORDS['tabi']) * len(SUBREDDITS)}")
    
    df = collector.collect_shoe_data(
        shoe_keywords=SHOE_KEYWORDS,
        subreddits=SUBREDDITS,
        results_per_search=50  # Adjust based on your API limits
    )
    
    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    output_file = f'data/raw/reddit_posts_{datetime.now().strftime("%Y%m%d")}.csv'
    df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    print(f"Total posts collected: {len(df)}")
    print(f"\nPosts by shoe:")
    print(df['shoe_type'].value_counts())
    print(f"\nPosts by subreddit:")
    print(df['subreddit'].value_counts())
    print(f"\nData saved to: {output_file}")


if __name__ == "__main__":
    main()