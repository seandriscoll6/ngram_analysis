import pandas as pd
import numpy as np
#import glob
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':

    #designate a source file and do a count of the number of occurrences for each n-gram
    #Since it has to be done in chunks, use Dask at the end of the process to make sure that proper totals are counted

    gram = 3
    output_filename = str(gram)+'_grams_queries.csv'

    #all_files = glob.glob('seans_sem_toolbox/new_keywords/QE/query_files/n-gram/*.csv')
    all_files = ["3311190029.csv"]

    for index, file_ in enumerate(all_files):

        print(file_)

        for chunk_index, chunk in enumerate(pd.read_csv(file_, usecols=['Keyword'], chunksize=10000)):

            #chunk = chunk.drop_duplicates(subset=['Keyword'])

            #word_vectorizer = CountVectorizer(ngram_range=(gram, gram), analyzer='word', stop_words='english', min_df=1)
            word_vectorizer = CountVectorizer(ngram_range=(gram, gram), analyzer='word', min_df=1)

            chunk = chunk.fillna('')

            sparse_matrix = word_vectorizer.fit_transform(chunk['Keyword'])
            print(sparse_matrix)
            train_data_features = sparse_matrix.toarray()
            print(train_data_features)
            #vocab = word_vectorizer.get_feature_names()
            vocab = word_vectorizer.get_feature_names_out()
            print(vocab)
            dist = np.sum(train_data_features, axis=0)

            list_of_tuples = list(zip(vocab, dist))

            temp_df = pd.DataFrame(list_of_tuples, columns=['Name', 'Occurrence'])

            if chunk_index == 0:
                temp_df.to_csv(output_filename,index=False)
            else:
                temp_df.to_csv(output_filename, index=False, header=False, mode='a')

    #Then use Dask to merge the data
    import dask.dataframe as dd

    df = dd.read_csv(output_filename, sep=',')

    df = df.groupby(['Name']).sum().compute()
    df= df.sort_values(by=['Occurrence'], ascending=False)
    #df= df.head(10000)
    df.to_csv(output_filename.replace(".csv","_sorted.csv"))