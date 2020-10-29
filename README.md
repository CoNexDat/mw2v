# mw2v

Folders:
Each of the folders corresponds to a different dataset. 
* NYT: files for New York Times newspaper dataset. Slices are years, from 1990 to 2016.
* TG: files for The Guardian newspaper dataset. Slices are years, from 1999 to 2016.
* NYT-TG: files for the two-source model, where one slice is the NYT and the other one is the TG, both during the 2010-2016 period.

Files description:
* Files starting with "delta_" and "mean_" contain the trained words embedding for each of the datasets.
* Files starting with "slices_" contain the slices names for each dataset.
* Files starting with "sampling_tables_" and "word_index_" are used to perform the negative/positive samplings during the parallel training of each slice embedding. 
* Files starting with "testset_1_" contain the list with the triplets (word, label, slice) to be used for semantic similarity evaluation of the embeddings. 
* Files starting with "testset_2_" contain the list with the pairs (word_1, slice_1) , (word_2, slice_2) to be used for alignment quality evaluation of the embeddings. 

