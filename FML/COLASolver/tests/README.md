# Tests

This folder contains some tests. This is mainly comparing to L-PICOLA / MG-PICOLA.
We have the option of using the same random number generator and the same time-stepping.
When these things are the same we should get *very* close results (and we do do, the agreement is < 0.1%).
For these tests I also modified the power-spectrum evaluation to do exactly the same in the two codes.
Will add more tests later.

NB: the k we output is sligtly different (we output the mean of the k-modes in each bin, while the picola results are the raw k-bins thus only the P(k) rows will agree)
