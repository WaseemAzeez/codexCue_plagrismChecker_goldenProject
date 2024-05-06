import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class PlagiarismCheckerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Plagiarism Checker")
        self.master.geometry("500x400")

        self.create_widgets()

    def create_widgets(self):
        self.label1 = ttk.Label(self.master, text="Enter Original Text:")
        self.label1.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.original_text = scrolledtext.ScrolledText(self.master, width=40, height=5)
        self.original_text.grid(row=1, column=0, padx=10, pady=5)

        self.label2 = ttk.Label(self.master, text="Enter Candidate Text:")
        self.label2.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.candidate_text = scrolledtext.ScrolledText(self.master, width=40, height=5)
        self.candidate_text.grid(row=3, column=0, padx=10, pady=5)

        self.check_button = ttk.Button(self.master, text="Check Plagiarism", command=self.check_plagiarism)
        self.check_button.grid(row=4, column=0, padx=10, pady=5)

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        preprocessed_text = ' '.join(lemmatized_tokens)
        return preprocessed_text

    def calculate_similarity(self, text1, text2):
        preprocessed_text1 = self.preprocess_text(text1)
        preprocessed_text2 = self.preprocess_text(text2)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([preprocessed_text1, preprocessed_text2])

        similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return similarity_score

    def check_plagiarism(self):
        original_text = self.original_text.get("1.0", tk.END)
        candidate_text = self.candidate_text.get("1.0", tk.END)

        similarity_score = self.calculate_similarity(original_text, candidate_text)

        if similarity_score >= 0.7:  # Adjust threshold as needed
            messagebox.showinfo("Plagiarism Check", "The texts are similar. Possible plagiarism detected!")
        else:
            messagebox.showinfo("Plagiarism Check", "The texts are dissimilar. No plagiarism detected.")

def main():
    root = tk.Tk()
    app = PlagiarismCheckerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
