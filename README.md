# Flavor Navigator 

A desktop search engine for discovering dishes from around the world based on ingredients, cuisine type, or dish name. Explore culinary traditions and find recipes through an intuitive GUI powered by NLP and semantic search.

## Overview

Flavor Navigator helps food enthusiasts explore international cuisines with an intelligent search system. Whether you're looking for Spanish paella, Tunisian lablebi, or trying to remember a dish you once had, this application uses natural language processing and TF-IDF vectorization to find the most relevant dishes in its database.

**Key Capability**: Fuzzy string matching to handle misspellings and typos in your search queries.

## Features

**Smart Search Engine**
- TF-IDF vectorization for semantic similarity matching
- Cosine similarity ranking for relevant results
- Fuzzy string matching for typo tolerance (80% threshold)
- Real-time search with instant results

**Diverse Cuisine Database**
- Dishes from Spain and Tunisia
- Multi-language support for dish names
- Rich descriptions and origin information
- Recipe links for each dish

**Beautiful Desktop GUI**
- Tkinter-based user interface
- Image display with dynamic resizing
- Scrollable results panel
- Professional styling with custom fonts and colors

**Recipe Integration**
- Direct links to external recipe websites
- One-click browser access to full recipes
- Clickable recipe links with hover effects

## Tech Stack

### Core Technologies
- **Python 3.7+** - Primary language
- **Tkinter** - Desktop GUI framework (built-in with Python)
- **Pandas** - Data loading and manipulation
- **NumPy** - Numerical operations

### NLP & Search Libraries
- **Scikit-learn** - TF-IDF vectorization and text processing
- **NLTK** - Natural Language Toolkit
  - Stopword removal
  - Porter Stemmer lemmatization
- **FuzzyWuzzy** - Fuzzy string matching for typo tolerance

### Image Processing
- **Pillow (PIL)** - Image loading and resizing
- **ImageTk** - Tkinter image integration

### Additional Tools
- **webbrowser** - Open recipe links in default browser

## Architecture

```
┌─────────────────────────────────────────┐
│         Tkinter GUI Layer               │
│  ├── Search Input Field                 │
│  ├── Image Display Panel                │
│  └── Results Visualization              │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│      Search & Ranking Engine            │
│  ├── TF-IDF Vectorization               │
│  ├── Cosine Similarity Calculation      │
│  └── Top-3 Result Selection             │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│      Text Preprocessing Pipeline        │
│  ├── Lowercasing                        │
│  ├── Punctuation Removal                │
│  ├── Stopword Filtering                 │
│  └── Porter Stemming                    │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│     Fuzzy Matching & Input              │
│  ├── Misspelling Detection              │
│  ├── Query Correction                   │
│  └── CSV Dataset Loading                │
└─────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- tkinter (usually comes with Python)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ghaidabs/Flavor-Navigator.git
   cd Flavor-Navigator
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install pandas nltk scikit-learn pillow fuzzywuzzy python-Levenshtein
   ```

4. **Download NLTK stopwords data** (required for NLP processing)
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

5. **Verify all files are present**
   ```
   ✓ main.py (application code)
   ✓ dataset.csv (dish database)
   ✓ icon.png (window icon)
   ✓ masfouf.jpg, lablebi.jpg, paella.jpg, gazpacho.jpg (dish images)
   ```

### Running the Application

```bash
python main.py
```

The application window will launch with the search interface.

## Usage

### How to Search

1. **Launch the application**
   ```bash
   python main.py
   ```

2. **Enter your search query** in the text field
   - Search by dish name: `"paella"`, `"gazpacho"`
   - Search by country: `"spain"`, `"tunisia"`
   - Search by ingredient: `"chickpea"` (for Lablebi)
   - Misspellings are handled automatically!

3. **Click the "Search" button** or press Enter
   - System preprocesses your query
   - Calculates semantic similarity to all dishes
   - Displays top 3 most relevant results

4. **View results**
   - Dish name and origin country
   - Beautiful high-quality image
   - Detailed description
   - Direct link to external recipe

5. **Access recipes**
   - Click any recipe link (blue underlined text)
   - Opens in your default web browser
   - Follow the full ingredient list and instructions

## Dataset Format

### CSV Structure

The `dataset.csv` contains:

```csv
image,dish,description,country,recipe
masfouf.jpg,Masfouf,"Sweet couscous...",Tunisia,https://recipe-link...
lablebi.jpg,Lablebi,"Chickpea soup...",Tunisia,https://recipe-link...
paella.jpg,Paella,"Saffron rice...",Spain,https://recipe-link...
gazpacho.jpg,Gazpacho,"Tomato soup...",Spain,https://recipe-link...
```

### Column Details

| Column | Type | Description |
|--------|------|-------------|
| `image` | String | Filename of dish image |
| `dish` | String | Name of the dish |
| `description` | String | Detailed dish description |
| `country` | String | Country of origin |
| `recipe` | String | URL link to external recipe |

### Current Dishes

1. **Masfouf** (Tunisia/Morocco)
   - Sweet couscous with nuts, dates, and sugar
   - Traditional Berber dessert

2. **Lablebi** (Tunisia)
   - Spiced chickpea soup
   - Served with bread and poached egg

3. **Paella** (Spain)
   - Saffron-flavored rice with seafood/meat
   - Mediterranean classic

4. **Gazpacho** (Spain)
   - Cold tomato-based soup
   - Perfect summer dish

**Note**: This application is designed as a desktop utility for discovering international dishes. The dataset can be expanded with more cuisines and recipes. Recipe links are external sources and may change over time.
