This repository is for the final project for SJSU CS256 Spring 2020, professor Teng Moh. It includes:
1. Folder for data:
    1) creditcard.csv: original data from Kaggle (https://www.kaggle.com/mlg-ulb/creditcardfraud).
    2) creditcard_testing.csv: 30% of original data for testing, with duplicates removed as well as time and amount scaled.
2. Folder for documentations:
    1) CS256FinalReport_YanChen.pdf: Final report.
    2) CS256Presentation_YanChen.pptx: ppt (without animations).
    3) CS256Proposal_YanChen: Project proposal.
2. Folder for saved models:
    Six .joblib files for LOF, SVM with RBF kernel, SVM with the polynomial kernel, and the models trained using resampled data (resampled 70% of original data for training).
3. Code in .ipynb (Jupyter notebook):
    1) CS256_Project.ipynb: for training and saving models through different resampling combination. Output included.
    2) CS256_Demo.ipynb: load the saved model and train the test data (30% of original data, creditcard_testing.csv). Output included.
4. Code in .py
   1) CS256_Project.py: .py version of CS256_Project.ipynb. Output excluded.
   2) CS256_Demo.py: .py version of CS256_Demo.ipynb.  Output excluded.

The CS256_Project file takes a long time to run since it includes training many models. So try the CS256_Demo is recommended.

If run CS256_Project, note that the result may be different since the 5 folds are separated randomly.

Original env was Anaconda + Jupyter (Anaconda download: https://www.anaconda.com/products/individual).

Packages are included in Anaconda except for imbalance-learn and joblib, which need to be downloaded/installed first:

  imbalance-learn: https://imbalanced-learn.readthedocs.io/en/stable/install.html
  joblib: https://joblib.readthedocs.io/en/latest/installing.html 
