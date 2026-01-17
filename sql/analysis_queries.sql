-- Fashion Trend Analysis Queries

-- Query 1: Post Frequency

            SELECT 
                shoe_type,
                COUNT(*) as total_posts,
                COUNT(DISTINCT subreddit) as num_subreddits,
                ROUND(AVG(position), 2) as avg_search_position
            FROM reddit_posts
            GROUP BY shoe_type
        ;

-- Query 2: Subreddit Engagement

            SELECT 
                subreddit,
                shoe_type,
                COUNT(*) as posts,
                ROUND(AVG(position), 2) as avg_position
            FROM reddit_posts
            GROUP BY subreddit, shoe_type
            HAVING posts >= 3
            ORDER BY avg_position ASC
        ;

-- Query 3: Top Posts (Window Function)

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
        ;

-- Query 4: Discussion Themes

            SELECT 
                shoe_type,
                SUM(mentions_worth) as worth_discussions,
                SUM(mentions_buying) as buying_discussions,
                SUM(mentions_hype) as hype_discussions,
                COUNT(*) as total_posts
            FROM title_keywords
            GROUP BY shoe_type
        ;
