# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was created by Bryan Mischler. The created model uses a Random Forest Classifier developed in July 2025 
using scikit-learn. The model predicts whether a person's annual income exceeds $50,000 based on demographic and
employment data. The model was trained using a fixed random seed of 42 for repeatability. This model was
developed as part of WGU's Machine Learning DevOps course (D501). For additional details or feedback, please 
contact bmischl@wgu.edu.

## Intended Use

This model is intended for academic purposes to predict whether an individual's annual income exceeds $50,000
based on U.S. Census data. The primary users for this model are students learning about machine learning and
DevOps at WGU. This model is not intended for use in real-world applications. 

## Training Data

The model was trained on 80% of the census data located in the provided data/census.csv file. This dataset
contains demographic and employment information on adults in the United States. 

Before training, all categorical features (such as workclass, education, and occupation) were converted to numerical
format using one-hot encoding. The salary column was converted to binary format using label binarization. 

## Evaluation Data

The evaluation data consists of a 20% subset of the original U.S. Census Bureau dataset. The subset was grouped
by class to maintain label distribution. This test set was not used during training. Evaluation on this data helps us 
see how well the model works for new information. I used the same steps to prepare the test data as we did for 
the training data. 

The same preprocessing pipeline was applied to the evaluation data, including one-hot encoding for categorical variables
and binarization of the salary label. 

## Metrics

The model’s performance was evaluated using the following metrics:
- Precision:0.7353
- Recall:0.6378
- F1 Score:0.6831

Additionally, performance was computed for slices of the data grouped by categorical features. For example, the 
F1 score for those with a Bachelor's degrees was 0.7618.  While the F1 score for those who had a private work
class was 0.6760. Please see slice_output.txt for more details. 

## Ethical Considerations

This model uses data about people's age, race, sex, and marital status, which are considered sensitive or 
protected information.  The model is not meant to be used for real world applications such as making monetary
loan decisions. The model should not be used in situation where a wrong predictions could cause harm to
someone or affect their rights.

## Caveats and Recommendations

- This model is for educational purposes only. 
- The model was trained only on U.S. census data from a specific period. This data set should not be used to 
generalize other populations or time frames.
- Model performance can be improved by tuning or using more advanced models.

