import pickle
import pandas as pd

def load_wikipedia_sentence(mention):
    chunk_id = mention['chunk_id'].item()
    sentence_id = mention['sentence_id'].item()
    raw_data = _find_chunk(chunk_id)
    df = pd.DataFrame(raw_data)
    return df[df['sentence_id'] == sentence_id]


def _find_chunk(chunk_id):
    with open('wikipedia_dataset/corpus.pickle', "rb") as f:
        chunk_counter = 0
        while True:
            try:
                raw_data, _ = pickle.load(f)
                if chunk_counter == chunk_id:
                    return raw_data
            except EOFError:
                return None
            print(f"skipping chunk {chunk_counter}")
            chunk_counter += 1
