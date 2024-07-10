""" 
Train-validation-test split in proportions 72% - 18% - 10% (by défault).

Transferred to a function from the scripts split_PCA1_SVM...ipynb

Example: see the script split_PCA1_SVM_v5bis.ipynb.
A simpler example can be added.
"""

from sklearn.model_selection import train_test_split


def train_val_test_split(data, target, test_val_sizes=(0.1, 0.2), random_state=None):
    """
    Train-validation-test split in 3 parts by 2 consecutive splits.
    This function takes 2 input arrays.
    See: https://datascience.stackexchange.com/a/15136.
    
    Args:
        data (lists, numpy array, scipy-sparse matrix or pandas dataframe)
        target (lists, numpy array, scipy-sparse matrix or pandas dataframe):
            arrays to be split
        test_val_sizes (tuple, optional): proportions of split. Defaults to (0.1, 0.2).
        random_state (optional):
            Common random state for both splits.
            Pass an int for reproducible output across multiple function calls. 
            Defaults to None.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: the subdivided datasets.
    
    Possible improvements. 
    1: take any number of input arrays.
    2: accept the final proportions of val , test sets as arguments.
    3: split into any number of parts.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        data,
        target,
        test_size=test_val_sizes[0],
        random_state=random_state)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=test_val_sizes[1],
        random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
