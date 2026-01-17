"""
Database Setup - Load data into SQLite and create analytical tables
"""

import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime

class DatabaseManager:
    """Manage SQLite database for fashion trend analysis"""
    
    def __init__(self, db_path='data/fashion_trends.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        print(f"Connected to database: {db_path}")
    
    def load_reddit_data(self, csv_path):
        """Load Reddit posts from CSV into database"""
        
        df = pd.read_csv(csv_path)
        
        # Convert collected_at to datetime
        df['collected_at'] = pd.to_datetime(df['collected_at'])
        
        # Extract date from URL if possible (Reddit post date)
        # This is approximate - would need actual scraping for real dates
        df['post_date'] = df['collected_at'].dt.date
        
        # Load into database
        df.to_sql('reddit_posts', self.conn, if_exists='replace', index=False)
        
        print(f"✓ Loaded {len(df)} posts into 'reddit_posts' table")
        
        return df
    
    def create_analytical_views(self):
        """Create SQL views for analysis"""
        
        # View 1: Post counts by shoe and subreddit
        self.conn.execute("""
            CREATE VIEW IF NOT EXISTS post_frequency AS
            SELECT 
                shoe_type,
                subreddit,
                COUNT(*) as post_count,
                AVG(position) as avg_position
            FROM reddit_posts
            GROUP BY shoe_type, subreddit
            ORDER BY post_count DESC
        """)
        
        # View 2: Temporal analysis (by collection date as proxy)
        self.conn.execute("""
            CREATE VIEW IF NOT EXISTS temporal_trends AS
            SELECT 
                date(collected_at) as date,
                shoe_type,
                COUNT(*) as daily_posts,
                AVG(position) as avg_position
            FROM reddit_posts
            GROUP BY date(collected_at), shoe_type
            ORDER BY date
        """)
        
        # View 3: Title analysis - keyword frequency
        self.conn.execute("""
            CREATE VIEW IF NOT EXISTS title_keywords AS
            SELECT 
                shoe_type,
                subreddit,
                title,
                CASE 
                    WHEN LOWER(title) LIKE '%worth%' THEN 1 ELSE 0 
                END as mentions_worth,
                CASE 
                    WHEN LOWER(title) LIKE '%buy%' OR LOWER(title) LIKE '%cop%' THEN 1 ELSE 0 
                END as mentions_buying,
                CASE 
                    WHEN LOWER(title) LIKE '%hype%' OR LOWER(title) LIKE '%trend%' THEN 1 ELSE 0 
                END as mentions_hype
            FROM reddit_posts
        """)
        
        print("✓ Created analytical views")
    
    def run_analysis_queries(self):
        """Run key SQL queries for the project"""
        
        print("\n" + "="*60)
        print("SQL ANALYSIS RESULTS")
        print("="*60)
        
        # Query 1: Overall post frequency
        print("\n1. Post Frequency by Shoe Type:")
        query1 = """
            SELECT 
                shoe_type,
                COUNT(*) as total_posts,
                COUNT(DISTINCT subreddit) as num_subreddits,
                ROUND(AVG(position), 2) as avg_search_position
            FROM reddit_posts
            GROUP BY shoe_type
        """
        df1 = pd.read_sql_query(query1, self.conn)
        print(df1.to_string(index=False))
        
        # Query 2: Engagement by subreddit (using search position as proxy)
        print("\n2. Best Performing Subreddits (by search position):")
        query2 = """
            SELECT 
                subreddit,
                shoe_type,
                COUNT(*) as posts,
                ROUND(AVG(position), 2) as avg_position
            FROM reddit_posts
            GROUP BY subreddit, shoe_type
            HAVING posts >= 3
            ORDER BY avg_position ASC
        """
        df2 = pd.read_sql_query(query2, self.conn)
        print(df2.to_string(index=False))
        
        # Query 3: Window function - rank posts by position within each shoe type
        print("\n3. Top Posts by Shoe Type (Window Function):")
        query3 = """
            WITH ranked_posts AS (
                SELECT 
                    shoe_type,
                    title,
                    subreddit,
                    position,
                    RANK() OVER (PARTITION BY shoe_type ORDER BY position) as rank_within_type
                FROM reddit_posts
            )
            SELECT *
            FROM ranked_posts
            WHERE rank_within_type <= 3
        """
        df3 = pd.read_sql_query(query3, self.conn)
        print(df3.to_string(index=False))
        
        # Query 4: Keyword analysis
        print("\n4. Discussion Themes by Shoe:")
        query4 = """
            SELECT 
                shoe_type,
                SUM(mentions_worth) as worth_discussions,
                SUM(mentions_buying) as buying_discussions,
                SUM(mentions_hype) as hype_discussions,
                COUNT(*) as total_posts
            FROM title_keywords
            GROUP BY shoe_type
        """
        df4 = pd.read_sql_query(query4, self.conn)
        print(df4.to_string(index=False))
        
        # Save queries to file
        queries_file = Path('sql/analysis_queries.sql')
        queries_file.parent.mkdir(exist_ok=True)
        
        with open(queries_file, 'w') as f:
            f.write("-- Fashion Trend Analysis Queries\n\n")
            f.write("-- Query 1: Post Frequency\n")
            f.write(query1 + ";\n\n")
            f.write("-- Query 2: Subreddit Engagement\n")
            f.write(query2 + ";\n\n")
            f.write("-- Query 3: Top Posts (Window Function)\n")
            f.write(query3 + ";\n\n")
            f.write("-- Query 4: Discussion Themes\n")
            f.write(query4 + ";\n")
        
        print(f"\n✓ Queries saved to: {queries_file}")
        
        return df1, df2, df3, df4
    
    def close(self):
        """Close database connection"""
        self.conn.close()
        print("\n✓ Database connection closed")


def main():
    """Main execution"""
    
    # Initialize database
    db = DatabaseManager('data/fashion_trends.db')
    
    # Load data
    csv_path = 'data/raw/reddit_posts_20260117.csv'
    df = db.load_reddit_data(csv_path)
    
    # Create analytical views
    db.create_analytical_views()
    
    # Run analysis queries
    results = db.run_analysis_queries()
    
    # Close connection
    db.close()
    
    print("\n" + "="*60)
    print("DATABASE SETUP COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Run NLP analysis: python src/nlp_analysis.py")
    print("2. Create visualizations: python src/visualization.py")
    print("3. Check sql/analysis_queries.sql for all queries")


if __name__ == "__main__":
    main()