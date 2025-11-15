import pandas
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import tkinter
from tkinter import ttk, Scrollbar
from PIL import Image, ImageTk
from fuzzywuzzy import process
import numpy
import webbrowser

def preprocess_data(data):
    # Lowercasing
    data = data.lower()
    
    # Removing punctuation
    data = re.sub(r'[^\w\s]', '', data)
    
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in data.split() if word not in stop_words]
    
    # Lemmatization
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    preprocessed_data = ' '.join(stemmed_tokens)
    
    return preprocessed_data

# Data extracting and preprocessing
dataset = pandas.read_csv('dataset.csv')
dataset['data'] = dataset['dish'] + ' ' + dataset['country'] + ' ' + dataset['description']
data = dataset['data'].apply(preprocess_data)

# Tf-Idf vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data)

class SearchEngine:
    def __init__(self, root):
        
        # Main window
        self.root = root
        root.title("")
        root.iconphoto(True, tkinter.PhotoImage(file="icon.png"))
        root.geometry("800x600")
        root.state('zoomed') 

        # Input field
        self.instruction_label = ttk.Label(root, text="Flavor Navigator", font=("Sans-serif",40,"bold","italic"),foreground="#4f6f52")     
        self.instruction_label.pack()
        self.search_frame = ttk.Frame(root)
        self.search_frame.pack(fill="y", pady=30)
        self.query_entry = ttk.Entry(self.search_frame, width=15, font=("Sans-serif", 20))
        self.query_entry.grid(row=0, column=0)
        self.search_button_style = ttk.Style()
        self.search_button_style.configure("Search.TButton", font=("Sans-serif", 20), width=7)
        self.search_button = ttk.Button(self.search_frame, text="Search", style="Search.TButton", command=self.search_images)
        self.search_button.grid(row=0, column=1, padx=10)
        
        # Output field
        self.canvas = tkinter.Canvas(root)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar = Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.image_frame = ttk.Frame(self.canvas)
        self.image_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.image_frame, anchor="nw")

    def search_images(self):
        query = self.query_entry.get().lower()
        options = ["spain", "tunisia", "masfouf", "lablebi", "paella", "gazpacho", "chickpea"]
        
        # Misspelling handling
        threshold = 80  
        match = process.extractOne(query, options)
        if match[1] >= threshold:
            query = match[0]

        # Calculating cosine similarities & Selecting top indices 
        query_vector = vectorizer.transform([query])
        cosine_similarities = tfidf_matrix.dot(query_vector.T).toarray().flatten()
        relevant_indices = numpy.where(cosine_similarities > 0.1)[0]
        top_indices = relevant_indices[numpy.argsort(cosine_similarities[relevant_indices])][-3:][::-1]

        # Clear previous search results
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        # Display results
        for i, idx in enumerate(top_indices):
            image_filename = dataset.iloc[idx]['image']
            dish_name = dataset.iloc[idx]['dish']
            country_name = dataset.iloc[idx]['country']
            description = dataset.iloc[idx]['description']
            recipe = dataset.iloc[idx]['recipe']

            image = Image.open(image_filename)
            image_width = 300  
            image_height = int((image_width / float(image.size[0])) * image.size[1])
            image = image.resize((image_width, image_height))
            photo = ImageTk.PhotoImage(image)
            image_label = ttk.Label(self.image_frame, image=photo)
            image_label.image = photo 
            image_label.grid(row=i * 8, column=0, padx=10, pady=5, rowspan=8, sticky="w")

            ttk.Label(self.image_frame, text=f"Dish Name:", font=("Sans-serif", 18,"bold"),foreground="#4f6f52").grid(row=i * 8, column=1, padx=10, pady=2, sticky="w")
            ttk.Label(self.image_frame, text=f"{dish_name}", font=("Sans-serif", 15,"underline")).grid(row=i * 8, column=2, padx=10, pady=2, sticky="w")
            ttk.Label(self.image_frame, text=f"Country of Origin: ", font=("Sans-serif", 18, "bold"),foreground="#4f6f52").grid(row=i *8 + 1, column=1, padx=10, pady=2, sticky="w")
            ttk.Label(self.image_frame, text=f"{country_name}", font=("Sans-serif", 15)).grid(row=i * 8+ 1, column=2, padx=10, pady=2, sticky="w")
            ttk.Label(self.image_frame, text=f"Description: ", font=("Sans-serif", 18, "bold"), foreground="#4f6f52").grid(row=i * 8 + 2, column=1, padx=10, pady=2, sticky="w")
            ttk.Label(self.image_frame, text=f"{description}", font=("Sans-serif", 15), wraplength=700).grid(row=i *8 + 2, column=2, padx=10, pady=2, sticky="w")
            ttk.Label(self.image_frame, text=f"Recipe: ", font=("Sans-serif", 18, "bold"), foreground="#4f6f52").grid(row=i * 8+ 3, column=1, padx=10, pady=2, sticky="w")
            recipe_link=ttk.Label(self.image_frame, text=f"{recipe}", font=("Sans-serif", 15), wraplength=700, cursor="hand2")
            recipe_link.grid(row=i *8 + 3, column=2, padx=10, pady=2, sticky="w")
            recipe_link.bind("<Button-1>", lambda event , link=recipe: webbrowser.open_new(link))
            
# Create Tkinter window
root = tkinter.Tk()
app = SearchEngine(root)
root.mainloop()




