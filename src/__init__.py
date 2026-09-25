"""Project package.

One global warning filter lives here, deliberately and narrowly scoped.

scikit-learn emits `sklearn.utils.parallel.delayed should be used with
sklearn.utils.parallel.Parallel` from inside HistGradientBoosting's own threading. The
permutation nulls in exp3/exp6 fit thousands of small models, so this fired ~345,000 times
and turned one experiment log into a 55 MB file with 98 useful lines in it. The warning is
about scikit-learn's internals, not about anything this project does, and there is no way
to act on it from here - so it is filtered at the package boundary rather than left to
drown the logs. Every other warning still propagates.
"""
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
    category=UserWarning,
)
