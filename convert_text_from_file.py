from clean_str import clean_str
from get_vec import get_vec
import numpy as np
def convert_text_from_file(file_path, emb_dim, t_model):
    with open(file_path, encoding = 'utf-8') as text_file:
        data = text_file.read()
        words = data.split(' ')
        # Removing empty strings
        words = list(filter(None, words))
        word_vectors = list()
        for word in words:
            token = clean_str(word)
            
            word_vector = get_vec(t_model, emb_dim, token)
            
               
            # We don't need zero vectors
            if np.any(word_vector):
                word_vectors.append(word_vector)

        return word_vectors