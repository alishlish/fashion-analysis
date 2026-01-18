"""
NLP Analysis Pipeline - Text Processing, Clustering, and Cultural Signal Analysis
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import re
import json

# NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class FashionNLPAnalyzer:
    """NLP analysis for fashion trend discussions"""
    
    def __init__(self, db_path='data/fashion_trends.db'):
        self.conn = sqlite3.connect(db_path)
        self.stop_words = set(stopwords.words('english'))
        
        # Add fashion-specific stopwords
        self.stop_words.update(['shoe', 'shoes', 'boot', 'boots', 'sneaker', 'sneakers'])
        
        print("✓ NLP Analyzer initialized")
    
    def load_data(self):
        """Load Reddit posts from database"""
        query = """
            SELECT 
                url,
                title,
                snippet,
                subreddit,
                shoe_type,
                collected_at
            FROM reddit_posts
        """
        
        self.df = pd.read_sql_query(query, self.conn)
        
        # Combine title and snippet for analysis
        self.df['full_text'] = self.df['title'] + ' ' + self.df['snippet'].fillna('')
        
        print(f"✓ Loaded {len(self.df)} posts")
        return self.df
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        return ' '.join(tokens)
    
    def analyze_cultural_terms(self):
        """Quantify usage of cultural terms like 'niche', 'chic', 'sauce'"""
        
        cultural_terms = {
            'niche': ['niche', 'unique', 'different', 'unconventional', 'avant-garde'],
            'accessible': ['basic', 'mainstream', 'popular', 'everyone', 'everywhere'],
            'quality': ['quality', 'premium', 'luxury', 'expensive', 'investment'],
            'hype': ['hype', 'trending', 'viral', 'hot', 'moment'],
            'style': ['stylish', 'fashionable', 'cool', 'clean', 'aesthetic']
        }
        
        results = []
        
        for idx, row in self.df.iterrows():
            text = row['full_text'].lower()
            
            term_counts = {}
            for category, terms in cultural_terms.items():
                term_counts[category] = sum(1 for term in terms if term in text)
            
            results.append({
                'shoe_type': row['shoe_type'],
                'subreddit': row['subreddit'],
                **term_counts
            })
        
        term_df = pd.DataFrame(results)
        
        # Aggregate by shoe type
        print("\n" + "="*60)
        print("CULTURAL TERM ANALYSIS")
        print("="*60)
        
        summary = term_df.groupby('shoe_type').sum()
        print("\nTerm frequency by shoe type:")
        print(summary)
        
        # Save results
        output_dir = Path('data/processed')
        output_dir.mkdir(exist_ok=True)
        summary.to_csv(output_dir / 'cultural_terms_analysis.csv')
        
        print(f"\n✓ Saved to: {output_dir / 'cultural_terms_analysis.csv'}")
        
        return term_df, summary
    
    def create_tfidf_features(self):
        """Create TF-IDF features for clustering"""
        
        # Preprocess all texts
        print("\nPreprocessing texts...")
        self.df['clean_text'] = self.df['full_text'].apply(self.preprocess_text)
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=100,  # Top 100 terms
            min_df=2,  # Must appear in at least 2 documents
            max_df=0.8,  # Can't appear in more than 80% of documents
            ngram_range=(1, 2)  # Include bigrams
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['clean_text'])
        
        print(f"✓ Created TF-IDF matrix: {self.tfidf_matrix.shape}")
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Top terms by shoe type
        print("\n" + "="*60)
        print("TOP TERMS BY SHOE TYPE (TF-IDF)")
        print("="*60)
        
        for shoe in self.df['shoe_type'].unique():
            mask = (self.df['shoe_type'] == shoe).values  # Convert to numpy array
            shoe_matrix = self.tfidf_matrix[mask]
            
            # Average TF-IDF scores
            avg_scores = np.asarray(shoe_matrix.mean(axis=0)).flatten()
            top_indices = avg_scores.argsort()[-10:][::-1]
            
            print(f"\n{shoe.upper()}:")
            for idx in top_indices:
                print(f"  {feature_names[idx]}: {avg_scores[idx]:.3f}")
        
        return self.tfidf_matrix
    
    def perform_clustering(self, n_clusters=5):
        """Cluster posts based on TF-IDF features"""
        
        print("\n" + "="*60)
        print(f"CLUSTERING ({n_clusters} clusters)")
        print("="*60)
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = kmeans.fit_predict(self.tfidf_matrix)
        
        # Analyze clusters
        print("\nCluster distribution:")
        print(self.df['cluster'].value_counts().sort_index())
        
        print("\nCluster composition by shoe type:")
        cluster_comp = pd.crosstab(
            self.df['cluster'], 
            self.df['shoe_type'], 
            normalize='index'
        ) * 100
        print(cluster_comp.round(1))
        
        # Top terms per cluster
        feature_names = self.vectorizer.get_feature_names_out()
        
        print("\n" + "="*60)
        print("TOP TERMS PER CLUSTER")
        print("="*60)
        
        cluster_terms = {}
        
        for cluster_id in range(n_clusters):
            mask = (self.df['cluster'] == cluster_id).values  # Convert to numpy array
            cluster_matrix = self.tfidf_matrix[mask]
            
            # Average TF-IDF for this cluster
            avg_scores = np.asarray(cluster_matrix.mean(axis=0)).flatten()
            top_indices = avg_scores.argsort()[-5:][::-1]
            
            top_terms = [feature_names[idx] for idx in top_indices]
            cluster_terms[f'cluster_{cluster_id}'] = top_terms
            
            print(f"\nCluster {cluster_id} ({mask.sum()} posts):")
            print(f"  Top terms: {', '.join(top_terms)}")
            
            # Sample titles from this cluster
            sample_titles = self.df[mask]['title'].head(3).tolist()
            print(f"  Sample posts:")
            for title in sample_titles:
                print(f"    - {title[:80]}...")
        
        # Save cluster assignments
        output_dir = Path('data/processed')
        self.df[['url', 'title', 'shoe_type', 'cluster']].to_csv(
            output_dir / 'clustered_posts.csv', 
            index=False
        )
        
        # Save cluster terms
        with open(output_dir / 'cluster_terms.json', 'w') as f:
            json.dump(cluster_terms, f, indent=2)
        
        print(f"\n✓ Saved cluster results to {output_dir}")
        
        return kmeans
    
    def visualize_clusters(self):
        """Create 2D visualization of clusters"""
        
        print("\nCreating cluster visualization...")
        
        # Reduce to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.df)-1))
        coords_2d = tsne.fit_transform(self.tfidf_matrix.toarray())
        
        self.df['x'] = coords_2d[:, 0]
        self.df['y'] = coords_2d[:, 1]
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Colored by cluster
        for cluster in self.df['cluster'].unique():
            mask = self.df['cluster'] == cluster
            axes[0].scatter(
                self.df[mask]['x'], 
                self.df[mask]['y'],
                label=f'Cluster {cluster}',
                alpha=0.6,
                s=100
            )
        axes[0].set_title('Posts Colored by Cluster', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Colored by shoe type
        colors = {'tabi': '#FF6B6B', 'samba': '#4ECDC4'}
        for shoe in self.df['shoe_type'].unique():
            mask = self.df['shoe_type'] == shoe
            axes[1].scatter(
                self.df[mask]['x'], 
                self.df[mask]['y'],
                label=shoe.capitalize(),
                alpha=0.6,
                s=100,
                color=colors.get(shoe, 'gray')
            )
        axes[1].set_title('Posts Colored by Shoe Type', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_dir = Path('outputs/figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'cluster_visualization.png', dpi=300, bbox_inches='tight')
        
        print(f"✓ Saved visualization to {output_dir / 'cluster_visualization.png'}")
        
        plt.close()
    
    def generate_summary_stats(self):
        """Generate summary statistics for resume/report"""
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS FOR RESUME")
        print("="*60)
        
        stats = {
            'total_posts_analyzed': len(self.df),
            'unique_clusters_identified': self.df['cluster'].nunique(),
            'subreddits_analyzed': self.df['subreddit'].nunique(),
            'vocabulary_size': len(self.vectorizer.get_feature_names_out()),
            'tfidf_features': self.tfidf_matrix.shape[1]
        }
        
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Save stats
        output_dir = Path('outputs')
        with open(output_dir / 'project_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    """Main execution"""
    
    print("="*60)
    print("FASHION TREND NLP ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = FashionNLPAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Analyze cultural terms
    analyzer.analyze_cultural_terms()
    
    # Create TF-IDF features
    analyzer.create_tfidf_features()
    
    # Perform clustering
    analyzer.perform_clustering(n_clusters=5)
    
    # Visualize
    analyzer.visualize_clusters()
    
    # Summary stats
    analyzer.generate_summary_stats()
    
    # Close
    analyzer.close()
    
    print("\n" + "="*60)
    print("NLP ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - data/processed/cultural_terms_analysis.csv")
    print("  - data/processed/clustered_posts.csv")
    print("  - data/processed/cluster_terms.json")
    print("  - outputs/figures/cluster_visualization.png")
    print("  - outputs/project_stats.json")
    print("\nNext step: Create visualizations for Tableau")


if __name__ == "__main__":
    main()