from gensim.models.word2vec import Word2Vec
import numpy as np

from Data import event_loader


if __name__ == '__main__':
    dataset = event_loader.load_dataset(event_loader.MAESTRO_EVENTS_FULL_240)

    vector_dim = 256

    string_dataset = [[str(n) for n in seq] for seq in dataset]
    model = Word2Vec(string_dataset, min_count=1, size=vector_dim, sg=1, window=8)
    model.train(string_dataset, total_examples=len(string_dataset), epochs=10)

    print(len(list(model.wv.vocab)))
    print(model.get_latest_training_loss())

    matrix = np.array([model.wv.vectors[model.wv.index2word.index(str(event))]
                       if str(event) in model.wv.vocab else np.zeros(vector_dim)
                       for event in list(range(240))])
    print(matrix.shape)

    np.save("Embedding/Embeddings/240event_256dim_smallwindow", matrix)
