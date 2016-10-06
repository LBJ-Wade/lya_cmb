This package is used for statistically cross-correlating lyman alpha forest signal with Cosmic microwave background.

It includes:

1. A cython module using linked list to do a chaining mesh algorithm to do cross
correlations.

`python setup.py build_ext --inplace`

2. A nilcrun/milcarun python file for data preprocessing.

3. A jackrun python file to do the error bars on cross-correlations.
