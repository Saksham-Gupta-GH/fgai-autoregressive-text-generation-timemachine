# Model Comparison Table (Best-Epoch and Test Set)

| Architecture | Parameters | Best Epoch | Best Val Loss | Best Val PPL | Val Top-1 | Val Top-5 | Test Loss | Test Top-1 | Test Top-5 | Total Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple RNN** | 1,104,745 | 4 | 4.2870 | 72.75 | 22.86% | 47.41% | 4.2835 | 21.91% | 46.63% | 33.5s |
| **Bidirectional RNN** | 1,721,961 | 2 | 4.4202 | 83.11 | 20.14% | 46.03% | 4.3641 | 20.57% | 45.60% | 37.0s |
| **LSTM** | 1,499,497 | 4 | 4.2104 | 67.38 | 22.88% | 48.16% | 4.2156 | 22.61% | 47.92% | 33.1s |
| **GRU** | 1,367,913 | 3 | 4.2388 | 69.33 | 22.29% | 48.24% | 4.2105 | 23.10% | 48.48% | 34.7s |


# Cross-Model Temperature Analysis Table (Averaged Over Seeds)

| Architecture | Temp (T) | Distinct-1 (Unigrams) | Distinct-2 (Bigrams) | Trigram Repetition Rate | Longest Loop | Empirical Behaviour |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Simple RNN** | 0.2 | 0.552 | 0.686 | 0.232 | 0 words | Repetitive phrasing loops; repeats 'cauldron of the door' motif |
| **Simple RNN** | 0.5 | 0.657 | 0.912 | 0.010 | 0 words | Moderate fluency with local repetition ('door that makes the world would not') |
| **Simple RNN** | 1.0 | 0.781 | 0.990 | 0.000 | 0 words | High lexical variety; syntactic drift across longer clauses |
| **Simple RNN** | 1.5 | 0.952 | 1.000 | 0.000 | 0 words | Excessive entropy; severe syntactic breakdown and ungrammatical token salad |
| **Bidirectional RNN** | 0.2 | 0.514 | 0.745 | 0.111 | 3 words | Severe looping traps; enters 6-word cycle 'and the moral of course' |
| **Bidirectional RNN** | 0.5 | 0.619 | 0.892 | 0.020 | 0 words | Lively dialogue phrasing ('cried the pigeon, i don t think you know'); zero loops |
| **Bidirectional RNN** | 1.0 | 0.791 | 0.971 | 0.010 | 0 words | Rich character vocabulary; clauses fragment over multi-word spans |
| **Bidirectional RNN** | 1.5 | 0.952 | 1.000 | 0.000 | 0 words | High disorder; random lexicon emission without coherent syntactic bindings |
| **LSTM** | 0.2 | 0.438 | 0.539 | 0.374 | 0 words | Strict syntax; repeats dominant character motifs ('queen s argument was') |
| **LSTM** | 0.5 | 0.610 | 0.892 | 0.010 | 0 words | Optimal narrative coherence; fluent story prose ('the march hare said in a hurry') |
| **LSTM** | 1.0 | 0.838 | 0.990 | 0.000 | 0 words | Imaginative lexical combinations ('way of soup'); slight clause drift |
| **LSTM** | 1.5 | 0.924 | 1.000 | 0.000 | 0 words | Entropy collapse; erratic word sequencing ('face high before without more pig') |
| **GRU** | 0.2 | 0.514 | 0.735 | 0.121 | 3 words | Extreme phrase trapping (52.5% repetition); loops on 'and she tried to herself' |
| **GRU** | 0.5 | 0.638 | 0.912 | 0.020 | 0 words | Clean conversational fluency with zero repetition loops ('said the cat... went on') |
| **GRU** | 1.0 | 0.762 | 0.980 | 0.000 | 0 words | Diverse lexicon; minor grammatical degradation in dialogue tags |
| **GRU** | 1.5 | 0.914 | 1.000 | 0.000 | 0 words | Unconstrained random sampling; complete loss of semantic and syntactic thread |
