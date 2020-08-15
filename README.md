# mw2v

Folders:
Each of the folders correspond to a different dataset: 
* NYT: files for New York Times newspaper dataset. Slices are years, from.... 
* TG: files for The Guardian newspaper dataset. Slices are years, from....
* NYT-TG: files for the two-source model, where one slice is the NYT during ... period and the other one is the TG during the ... period.

Files description:
* Files starting with delta and mean contain the trained words embedding for each of the datasets.
* Files starting with sampling_tables and word_index are used to perform the negative/positive sampling during the parallel training of each slice embedding. 
* Files starting with triplets contains the list with the triples (word, label, slice) to be used for semantic similarity evaluation. 
* 
Alignment test for alignment quality
