import datetime
import random
import json

class CognitiveAtlas:
    """
    Manages the collection and structure of personal knowledge entries.
    """
    def __init__(self):
        """Initializes the Cognitive Atlas with an empty list of entries."""
        self.knowledge_entries = []

    def add_entry(self, title: str, source: str, entry_type: str, tags: list, notes: str = ""):
        """
        Adds a new knowledge entry to the atlas.

        Args:
            title (str): The title of the knowledge piece (e.g., chapter, article name).
            source (str): The origin of the knowledge (e.g., book title, website).
            entry_type (str): The type of content (e.g., 'Book', 'Article', 'Video', 'Podcast').
            tags (list): A list of keywords or concepts associated with the entry.
            notes (str, optional): Personal notes or summary. Defaults to "".
        """
        entry = {
            "id": f"entry_{len(self.knowledge_entries) + 1}",
            "title": title,
            "source": source,
            "type": entry_type,
            "date_added": datetime.date.today().isoformat(),
            "tags": sorted(list(set(tag.lower() for tag in tags))), # Standardize tags
            "notes": notes
        }
        self.knowledge_entries.append(entry)
        print(f"Added entry: '{title}'")

    def load_sample_data(self):
        """Loads a predefined set of sample data into the atlas."""
        print("Loading sample knowledge entries...")
        sample_data = [
            {
                "title": "Chapter 3: The Bias-Variance Tradeoff",
                "source": "Introduction to Statistical Learning",
                "type": "Book",
                "tags": ["machine-learning", "bias", "variance", "overfitting", "regularization", "supervised-learning"],
                "notes": "Key concept for model complexity. High bias (underfit) vs. High variance (overfit)."
            },
            {
                "title": "Understanding t-SNE",
                "source": "Distill.pub",
                "type": "Article",
                "tags": ["data-visualization", "machine-learning", "dimensionality-reduction", "t-sne", "nlp"],
                "notes": "Great interactive explanation of how t-SNE creates clusters for high-dimensional data."
            },
            {
                "title": "Creating 3D Scatter Plots",
                "source": "Plotly Documentation",
                "type": "Documentation",
                "tags": ["data-visualization", "python", "plotly", "3d-plotting"],
                "notes": "Scatter3d is the go.Figure object to use. Can customize markers, colors, and hover text."
            },
            {
                "title": "Introduction to NetworkX",
                "source": "NetworkX Tutorial",
                "type": "Tutorial",
                "tags": ["python", "network-analysis", "graph-theory", "data-visualization"],
                "notes": "Useful for creating and analyzing graph structures. Can be used to find paths between nodes."
            },
            {
                "title": "L1 vs L2 Regularization",
                "source": "Andrew Ng's ML Course",
                "type": "Video",
                "tags": ["machine-learning", "regularization", "lasso", "ridge", "supervised-learning"],
                "notes": "Lasso (L1) can shrink coefficients to zero, performing feature selection. Ridge (L2) shrinks them but rarely to zero."
            }
        ]

        for item in sample_data:
            # Simulate adding entries over a few days
            entry_date = datetime.date.today() - datetime.timedelta(days=random.randint(1, 10))
            item["date_added"] = entry_date.isoformat()
            self.knowledge_entries.append(item)
        
        print(f"Loaded {len(sample_data)} sample entries.")


if __name__ == "__main__":
    # Create a new Cognitive Atlas
    my_atlas = CognitiveAtlas()

    # Load it with some sample data
    my_atlas.load_sample_data()

    # You can also add a new entry manually
    my_atlas.add_entry(
        title="What Are Word Embeddings?",
        source="Towards Data Science",
        entry_type="Article",
        tags=["nlp", "machine-learning", "word2vec", "embeddings"],
        notes="Representing words as dense vectors to capture semantic meaning."
    )

    # Print the first entry to see the structure
    print("\n--- Sample of First Entry ---")
    print(json.dumps(my_atlas.knowledge_entries[0], indent=2))
    print(f"\nTotal entries in the atlas: {len(my_atlas.knowledge_entries)}")