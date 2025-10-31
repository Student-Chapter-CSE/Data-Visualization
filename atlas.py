import sqlite3
import datetime
import random
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import networkx as nx

DB_FILE = "cognitive_atlas.db"

def init_db():
    """Initialize the SQLite database and create the table if it doesn't exist."""
    if Path(DB_FILE).exists():
        print(f"Database '{DB_FILE}' already exists.")
        return
    print("Initializing new database...")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            source TEXT,
            entry_type TEXT,
            date_added TEXT,
            tags TEXT,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()
    print(f"Database '{DB_FILE}' created successfully.")

def add_entry_to_db(title: str, source: str, entry_type: str, tags: list, notes: str = ""):
    """Adds a new knowledge entry to the database."""
    if not title or not source:
        raise ValueError("Title and source are required fields.")
    
    tags_json = json.dumps(sorted(list(set(tag.lower() for tag in tags))))
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO knowledge (title, source, entry_type, date_added, tags, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (title, source, entry_type, datetime.date.today().isoformat(), tags_json, notes)
        )
        conn.commit()
        print(f"✓ Entry added: {title}")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise
    finally:
        conn.close()

def get_all_entries():
    """Retrieves all entries from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM knowledge")
    entries = cursor.fetchall()
    conn.close()
    return entries

def delete_entry_from_db(entry_id: int):
    """Deletes an entry from the database by ID."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM knowledge WHERE id = ?", (entry_id,))
    conn.commit()
    deleted = cursor.rowcount
    conn.close()
    return deleted > 0

def load_sample_data_to_db():
    """Loads sample data into the database if it's empty."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM knowledge")
    if cursor.fetchone()[0] > 0:
        conn.close()
        print("Database already contains data. Skipping sample data load.")
        return

    print("Loading sample data into database...")
    sample_data = [
        {"title": "Chapter 3: The Bias-Variance Tradeoff", "source": "Introduction to Statistical Learning", "type": "Book", "tags": ["machine-learning", "bias", "variance", "overfitting", "regularization"], "notes": "Key concept for model complexity."},
        {"title": "Understanding t-SNE", "source": "Distill.pub", "type": "Article", "tags": ["data-visualization", "machine-learning", "dimensionality-reduction", "t-sne"], "notes": "Interactive explanation of t-SNE."},
        {"title": "Creating 3D Scatter Plots", "source": "Plotly Documentation", "type": "Documentation", "tags": ["data-visualization", "python", "plotly", "3d-plotting"], "notes": "Using go.Scatter3d."},
        {"title": "Introduction to NetworkX", "source": "NetworkX Tutorial", "type": "Tutorial", "tags": ["python", "network-analysis", "graph-theory"], "notes": "For creating and analyzing graphs."},
        {"title": "L1 vs L2 Regularization", "source": "Andrew Ng's ML Course", "type": "Video", "tags": ["machine-learning", "regularization", "lasso", "ridge"], "notes": "Lasso (L1) vs. Ridge (L2)."},
        {"title": "What Are Word Embeddings?", "source": "Towards Data Science", "type": "Article", "tags": ["nlp", "machine-learning", "word2vec", "embeddings"], "notes": "Representing words as dense vectors."}
    ]
    for item in sample_data:
        add_entry_to_db(item["title"], item["source"], item["type"], item["tags"], item["notes"])
    print("✓ Sample data loaded successfully.")
    conn.close()

# --- FastAPI Application ---
app = FastAPI(
    title="Cognitive Atlas API",
    description="An API to serve personal knowledge graph data.",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main HTML frontend."""
    try:
        with open("atlas_frontend.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: atlas_frontend.html not found.</h1><p>Please create the frontend file.</p>",
            status_code=500
        )

@app.get("/api/graph-data")
async def get_graph_data():
    """
    Processes the knowledge entries and returns a graph structure (nodes and edges)
    for visualization.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM knowledge", conn)
        conn.close()

        if df.empty:
            return {"nodes": [], "edges": [], "message": "No data available"}

        # Use tags to create a document for TF-IDF
        df['tags_str'] = df['tags'].apply(lambda x: ' '.join(json.loads(x)))

        # --- NLP and Dimensionality Reduction ---
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(df['tags_str'])

        # Use t-SNE to get 2D coordinates for the graph
        perplexity = max(min(len(df) - 1, 5), 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=500)
        coords = tsne.fit_transform(tfidf_matrix.toarray())

        # --- Build Graph with NetworkX ---
        G = nx.Graph()
        nodes_for_vis = []
        
        for idx, row in df.iterrows():
            G.add_node(row['id'], label=row['title'], group=row['entry_type'], notes=row['notes'])
            nodes_for_vis.append({
                "id": int(row['id']),
                "label": row['title'],
                "group": row['entry_type'],
                "x": float(coords[idx, 0] * 100),
                "y": float(coords[idx, 1] * 100),
                "value": len(json.loads(row['tags'])),
                "title": f"<b>{row['title']}</b><br>Source: {row['source']}<br>Type: {row['entry_type']}<br>Notes: {row['notes']}"
            })

        # Create edges between nodes that share tags
        tag_map = {}
        for idx, row in df.iterrows():
            tags = json.loads(row['tags'])
            for tag in tags:
                if tag not in tag_map:
                    tag_map[tag] = []
                tag_map[tag].append(row['id'])

        for tag, nodes in tag_map.items():
            if len(nodes) > 1:
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        if not G.has_edge(nodes[i], nodes[j]):
                            G.add_edge(nodes[i], nodes[j], weight=0)
                        G[nodes[i]][nodes[j]]['weight'] += 1

        edges_for_vis = [
            {"from": u, "to": v, "value": data['weight']}
            for u, v, data in G.edges(data=True)
        ]

        return {
            "nodes": nodes_for_vis,
            "edges": edges_for_vis,
            "stats": {
                "total_nodes": len(nodes_for_vis),
                "total_edges": len(edges_for_vis),
                "total_entries": len(df)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating graph: {str(e)}")

@app.post("/api/add-entry")
async def add_entry(title: str, source: str, entry_type: str, tags: list, notes: str = ""):
    """API endpoint to add a new knowledge entry."""
    try:
        add_entry_to_db(title, source, entry_type, tags, notes)
        return {"success": True, "message": "Entry added successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error adding entry: {str(e)}")

@app.get("/api/entries")
async def list_entries():
    """API endpoint to retrieve all entries."""
    try:
        entries = get_all_entries()
        return {"entries": entries, "count": len(entries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving entries: {str(e)}")

@app.delete("/api/entries/{entry_id}")
async def delete_entry(entry_id: int):
    """API endpoint to delete an entry by ID."""
    try:
        success = delete_entry_from_db(entry_id)
        if success:
            return {"success": True, "message": f"Entry {entry_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Entry {entry_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting entry: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "database": DB_FILE}

if __name__ == "__main__":
    print("🚀 Starting Cognitive Atlas Server...")
    print("=" * 50)

    # 1. Initialize the database
    init_db()

    # 2. Load sample data if the DB is empty
    load_sample_data_to_db()

    # 3. Start the FastAPI server
    print("=" * 50)
    print("✅ Server is running!")
    print("📖 Open your browser to http://127.0.0.1:8000")
    print("📊 API Documentation: http://127.0.0.1:8000/docs")
    print("=" * 50)
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")