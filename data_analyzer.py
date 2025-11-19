import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import io
import base64


class SpotifyDataAnalyzer:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        self.clean_data()
        self.train_model()

    def clean_data(self):
        """Clean and prepare the dataset"""
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()

        # Get available audio features
        self.features = ['danceability', 'energy', 'valence', 'acousticness',
                         'instrumentalness', 'liveness', 'speechiness', 'tempo']
        self.features = [f for f in self.features if f in self.df.columns]

    def train_model(self):
        """Train ML model for predictions"""
        if 'popularity' in self.df.columns and len(self.features) > 0:
            # Create popularity categories
            self.df['pop_category'] = pd.cut(self.df['popularity'],
                                             bins=3,
                                             labels=['Low', 'Medium', 'High'])

            X = self.df[self.features]
            y = self.df['pop_category']

            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.model.fit(X, y)

    def get_summary(self):
        """Get dataset summary"""
        summary = f"""
**Dataset Summary:**
• Total songs: {len(self.df)}
• Features: {len(self.df.columns)}
• Audio features: {', '.join(self.features[:5])}
"""
        if 'popularity' in self.df.columns:
            pop_stats = self.df['popularity'].describe()
            summary += f"• Average popularity: {pop_stats['mean']:.1f}/100\n"

        return summary

    def analyze_popularity(self):
        """Analyze popularity patterns"""
        if 'popularity' not in self.df.columns:
            return "Popularity data not available.", None

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        self.df['popularity'].hist(bins=20, alpha=0.7, color='skyblue')
        plt.title('Popularity Distribution')
        plt.xlabel('Popularity Score')

        plt.subplot(1, 2, 2)
        top_features = self.df[self.features].corrwith(self.df['popularity']).abs()
        top_features.sort_values().tail(5).plot(kind='barh', color='lightgreen')
        plt.title('Top Features Correlated with Popularity')

        plt.tight_layout()
        img = self.plot_to_image()

        analysis = f"🎵 **Popularity Analysis:**\nAverage: {self.df['popularity'].mean():.1f}/100"
        return analysis, img

    def show_correlations(self):
        """Show feature correlations"""
        if len(self.features) < 2:
            return "Not enough features.", None

        plt.figure(figsize=(10, 8))
        corr_matrix = self.df[self.features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Audio Features Correlation Heatmap')
        plt.tight_layout()

        img = self.plot_to_image()
        analysis = "🔗 **Feature Correlations**\nRed = positive, Blue = negative"
        return analysis, img

    def analyze_feature(self, feature_name):
        """Analyze specific feature"""
        if feature_name not in self.df.columns:
            return f"Feature '{feature_name}' not found.", None

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        self.df[feature_name].hist(bins=20, alpha=0.7, color='orange')
        plt.title(f'{feature_name.title()} Distribution')

        plt.subplot(1, 2, 2)
        if 'popularity' in self.df.columns:
            plt.scatter(self.df[feature_name], self.df['popularity'], alpha=0.5)
            plt.xlabel(feature_name)
            plt.ylabel('Popularity')
            plt.title(f'{feature_name} vs Popularity')

        plt.tight_layout()
        img = self.plot_to_image()

        stats = self.df[feature_name].describe()
        analysis = f" **{feature_name.title()}:** Avg: {stats['mean']:.3f}, Range: {stats['min']:.3f}-{stats['max']:.3f}"
        return analysis, img

    def plot_to_image(self):
        """Convert plot to base64 image"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return img_base64

    def predict_song(self, feature_values):
        """Predict song popularity with robust error handling"""
        if not hasattr(self, 'model'):
            return "Model not trained."

        input_features = []

        for feat in self.features:
            val = feature_values.get(feat)

            try:

                if val is None or str(val).lower() == 'none':
                    clean_val = self.df[feat].mean()
                else:

                    clean_val = float(val)
            except (ValueError, TypeError):
                clean_val = self.df[feat].mean()
            # -----------------------------

            input_features.append(clean_val)

        input_df = pd.DataFrame([input_features], columns=self.features)

        try:
            prediction = self.model.predict(input_df)[0]
            return f"🎵 Predicted: {prediction} popularity"
        except Exception as e:
            return f"Prediction Error: {str(e)}"