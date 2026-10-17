

# Cross-Model Temperature Analysis Table (Averaged Over Seeds)

| Architecture | Temp (T) | Distinct-1 (Unigrams) | Distinct-2 (Bigrams) | Trigram Repetition Rate | Longest Loop | Empirical Behaviour |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Simple RNN** | 0.2 | 0.553 | 0.687 | 0.230 | 0 words | Repetitive phrasing loops; repeats 'great difficulty of the time machine' motif |
| **Simple RNN** | 0.5 | 0.657 | 0.910 | 0.010 | 0 words | Moderate fluency with local repetition ('in the great hall... patient of the morlocks') |
| **Simple RNN** | 1.0 | 0.780 | 0.990 | 0.000 | 0 words | High lexical variety; syntactic drift across longer clauses ('the difference began hazy') |
| **Simple RNN** | 1.5 | 0.953 | 1.000 | 0.000 | 0 words | Excessive entropy; severe syntactic breakdown and ungrammatical token salad ('solstice plausible under a great do') |
| **Bidirectional RNN** | 0.2 | 0.513 | 0.747 | 0.110 | 3 words | Local looping traps; enters repetitive cadence ('i saw, i saw, i went, i was a little people') |
| **Bidirectional RNN** | 0.5 | 0.620 | 0.890 | 0.020 | 0 words | Sci-fi narrative phrasing ('the lamp dimension the time traveller... flickered man'); low repetition |
| **Bidirectional RNN** | 1.0 | 0.790 | 0.970 | 0.010 | 0 words | Rich vocabulary; clauses fragment over multi-word spans ('merged, sat, bearing, i stared him arms') |
| **Bidirectional RNN** | 1.5 | 0.950 | 1.000 | 0.000 | 0 words | High disorder; random lexicon emission without coherent syntactic bindings ('smothered ruinous and heel going away') |
| **LSTM** | 0.2 | 0.440 | 0.540 | 0.370 | 0 words | Strict syntax; repeats dominant character motifs ('i was not a little of the time traveller') |
| **LSTM** | 0.5 | 0.610 | 0.890 | 0.010 | 0 words | Optimal narrative coherence; fluent Victorian sci-fi prose ('the morlocks had not been a little way') |
| **LSTM** | 1.0 | 0.840 | 0.990 | 0.000 | 0 words | Imaginative lexical combinations ('felt to my camphor... iron going followed'); slight clause drift |
| **LSTM** | 1.5 | 0.923 | 1.000 | 0.000 | 0 words | Entropy collapse; erratic word sequencing ('forth things large class yet taken nor body among reach') |
| **GRU** | 0.2 | 0.513 | 0.737 | 0.120 | 3 words | Phrase trapping; loops on recurring narrative motifs ('people of the palace of green porcelain') |
| **GRU** | 0.5 | 0.640 | 0.910 | 0.020 | 0 words | Clean conversational sci-fi fluency with zero repetition loops ('medical man, and then i began to the lamp') |
| **GRU** | 1.0 | 0.760 | 0.980 | 0.000 | 0 words | Diverse lexicon; minor grammatical degradation in descriptive clauses ('cracked gold wrong on that came inhuman') |
| **GRU** | 1.5 | 0.910 | 1.000 | 0.000 | 0 words | Unconstrained random sampling; complete loss of semantic and syntactic thread ('fancy dimensions devoted great large ventilation') |
