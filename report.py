"""Build reports/summary.md, summary.csv and figures from finished runs.

    python report.py                         # runs/ -> reports/
    python report.py --runs runs_smoke --out reports/smoke
"""

from src.report import main

if __name__ == "__main__":
    main()
