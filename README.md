# Heart Disease Risk Prediction Using Logistic Regression

In this project we implement logistic regression from scratch for heart disease prediction.

---

## Table of Contents

- [Dataset Description](#dataset-description)
- [Step 1: Load and Prepare the Dataset](#step-1-load-and-prepare-the-dataset)
- [Step 2: Implement Basic Logistic Regression](#step-2-implement-basic-logistic-regression)
- [Step 3: Visualize Decision Boundaries](#step-3-visualize-decision-boundaries)
- [Step 4: Regularization](#step-4-regularization)
- [Step 5: SageMaker Deployment](#step-5-sagemaker-deployment)
- [Conclusion](#conclusion)

---

## Dataset Description

| Property | Value |
|----------|-------|
| **Source** | [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/neurocipher/heartdisease) |
| **Samples** | 270 patients |
| **Features** | 14 clinical attributes |
| **Target** | Binary (1 = disease presence, 0 = absence) |
| **Class distribution** | ~55% presence, ~45% absence |

**Features used in this project:**

| Feature | Description | Range |
|---------|-------------|-------|
| Age | Patient age in years | 29-77 |
| Cholesterol | Serum cholesterol | 126-564 mg/dL |
| BP | Resting blood pressure | 94-200 mm Hg |
| Max HR | Maximum heart rate achieved | 71-202 bpm |
| ST depression | ST depression induced by exercise | 0-6.2 |
| Number of vessels fluro | Major vessels colored by fluoroscopy | 0-3 |

---

## Step 1: Load and Prepare the Dataset

### What we did

1. **Downloaded** the dataset from Kaggle
2. **Loaded** the CSV file using Pandas
3. **Binarized** the target column: "Presence" → 1, "Absence" → 0
4. **Explored** the data with summary statistics and visualizations

### EDA Findings

- **No missing values** found in any column
- **Outliers detected** in Cholesterol (some values > 400 mg/dL) - kept them since they represent real patient conditions
- **Class distribution**: Moderately balanced (~55% disease presence)

### Data Preparation

- **Selected 6 features**: Age, Cholesterol, BP, Max HR, ST depression, Number of vessels fluro
- **Split**: 70% training (189 samples), 30% test (81 samples)
- **Stratified split**: Keeps the same disease rate in both sets (~55%)
- **Normalization**: Z-score normalization using training set statistics

### Reporting

> Downloaded from Kaggle; 270 samples with 14 features. After binarizing target, we have ~55% disease presence rate. No missing values. Stratified 70/30 split ensures balanced classes in train/test. Normalized features using training mean and std to help gradient descent converge faster.

---

## Step 2: Implement Basic Logistic Regression


We implemented three core components from scratch:

**1. Sigmoid function** - Converts any value to probability (0 to 1):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**2. Cost function** - Binary cross-entropy measures prediction errors:

$$J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

**3. Gradient descent** - Updates weights iteratively to minimize cost:

$$w = w - \alpha \cdot \frac{1}{m} X^T (\hat{y} - y)$$

### Training Parameters

| Parameter | Value |
|-----------|-------|
| Learning rate (α) | 0.01 |
| Iterations | 1000 |
| Threshold | 0.5 |

### Results

**Cost convergence:**
- Initial cost: ~0.69
- Final cost: ~0.46
- The cost decreased smoothly, showing good convergence

**Performance Metrics:**

| Metric | Train (%) | Test (%) |
|--------|-----------|----------|
| Accuracy | 80.95 | 79.01 |
| Precision | 83.02 | 81.40 |
| Recall | 83.02 | 81.40 |
| F1 Score | 83.02 | 81.40 |

**Feature Weights (what the model learned):**

| Feature | Weight | Interpretation |
|---------|--------|----------------|
| Number of vessels fluro | +0.74 | More blocked vessels -> higher risk |
| ST depression | +0.54 | More ST depression -> higher risk |
| Age | +0.27 | Older age -> higher risk |
| BP | +0.16 | Higher blood pressure -> higher risk |
| Cholesterol | +0.09 | Higher cholesterol -> slightly higher risk |
| Max HR | -0.53 | Higher max heart rate -> lower risk |

### Reporting

> Cost decreased from ~0.69 to ~0.46 over 1000 iterations showing good convergence. Test accuracy of ~79% is close to train accuracy (~81%), indicating no severe overfitting. The learned weights make medical sense: blocked vessels and ST depression increase risk, while a healthy max heart rate decreases risk.

---

## Step 3: Visualize Decision Boundaries

### What is a decision boundary?

A decision boundary is the line that separates the two classes. Since logistic regression is a linear model, it can only draw straight lines. We visualize in 2D by training separate models with just 2 features.

### Feature Pairs Selected

We chose 3 clinically meaningful pairs:

1. **Age vs Cholesterol** - Both are major risk factors
2. **BP vs Max HR** - Cardiovascular indicators
3. **ST Depression vs Number of Vessels** - Direct heart condition indicators

### 2D Model Accuracies

| Feature Pair | Train Acc | Test Acc |
|--------------|-----------|----------|
| Age vs Cholesterol | 62.4% | 58.0% |
| BP vs Max HR | 68.8% | 65.4% |
| ST Depression vs Number of Vessels | 77.2% | 75.3% |

### Insights

- **Age vs Cholesterol**: Poor separation (~58% test). These features alone don't separate classes well - there's a lot of overlap.
- **BP vs Max HR**: Moderate separation (~65% test). We can see some pattern but still much overlap.
- **ST Depression vs Number of Vessels**: Best separation (~75% test). These are direct heart condition indicators, so they separate classes better.

### Reporting

> Decision boundaries show that logistic regression creates straight lines only. The pair "ST Depression vs Number of Vessels" gives the best separation because these features are direct indicators of heart problems. Age and Cholesterol alone are not enough to separate classes well - we need multiple features together for better predictions.

---

## Step 4: Regularization

### What is L2 Regularization?

Regularization adds a penalty for large weights to prevent overfitting. The model is forced to keep weights small, which makes it generalize better.

**Regularized cost function:**

$$J_{reg} = J + \frac{\lambda}{2m}\sum_{j=1}^n w_j^2$$

- λ = 0: No regularization
- λ too high: Underfitting (model too simple)
- λ optimal: Best balance

### Lambda Tuning Results

| λ | Train Acc (%) | Test Acc (%) | Train F1 (%) | Test F1 (%) | ‖w‖ (norm) |
|---|---------------|--------------|--------------|-------------|------------|
| 0 | 80.95 | 79.01 | 83.02 | 81.40 | 1.13 |
| 0.001 | 80.95 | 79.01 | 83.02 | 81.40 | 1.13 |
| 0.01 | 80.95 | 79.01 | 83.02 | 81.40 | 1.12 |
| 0.1 | 80.42 | 79.01 | 82.57 | 81.40 | 1.05 |
| 1 | 78.31 | 76.54 | 80.77 | 79.07 | 0.74 |

### Optimal Lambda

**Best λ = 0** (or 0.001/0.01 - same performance)

In this case, regularization didn't improve test accuracy. This suggests:
- The model wasn't overfitting much to begin with
- The dataset is small (270 samples), so regularization benefits are limited
- The gap between train and test accuracy was already small (~2%)

### Effect of Regularization on Weights

| Feature | w (λ=0) | w (λ=0.1) |
|---------|---------|-----------|
| Number of vessels fluro | 0.74 | 0.69 |
| ST depression | 0.54 | 0.51 |
| Max HR | -0.53 | -0.49 |
| Age | 0.27 | 0.25 |
| BP | 0.16 | 0.14 |
| Cholesterol | 0.09 | 0.08 |

As λ increases, all weights shrink towards zero (that's the regularization effect).

### Reporting

> Tested λ values [0, 0.001, 0.01, 0.1, 1]. Optimal λ = 0 (no regularization needed). The model wasn't overfitting since train/test accuracy were already close. Higher λ (like 1) actually hurt performance by making weights too small. Regularization is more useful when you have many features or when there's a big gap between train and test performance.

---

## Step 5: SageMaker Deployment

### Objective

The goal was to deploy the trained logistic regression model to Amazon SageMaker, enabling real-time inference for heart disease risk prediction. This would allow healthcare applications to send patient data and receive probability scores instantly.

### Planned Approach

1. **Export the best model**: Save weights (`w`) and bias (`b`) as NumPy arrays
2. **Create SageMaker notebook instance**: Upload training notebook and data
3. **Build inference handler**: Create a script to load the model and process patient inputs
4. **Deploy endpoint**: Host the model for real-time predictions
5. **Test endpoint**: Invoke with sample patient data (e.g., Age=60, Cholesterol=300)

### Deployment Status: Not Completed

The deployment could not be completed due to IAM permission restrictions.

We were using the **LabRole** user provided by AWS Academy, which has limited permissions for security reasons. Specifically, the LabRole lacks the necessary IAM policies to:

- Create SageMaker endpoint configurations
- Deploy model artifacts to SageMaker hosting
- Create and manage SageMaker inference endpoints
- Access certain S3 bucket operations required for model deployment

### Error Encountered

When attempting to create the endpoint, the following permission error was encountered:

```
AccessDeniedException: User: arn:aws:sts::xxx:assumed-role/LabRole/... 
is not authorized to perform: sagemaker:CreateEndpoint
```

### Benefits of Deployment

Even though we couldn't complete the deployment, it's important to note the benefits:

- **Real-time risk scoring**: Healthcare providers could get instant predictions
- **Scalability**: SageMaker auto-scales based on traffic
- **Integration**: REST API enables integration with hospital systems
- **Monitoring**: CloudWatch metrics for model performance tracking

---

## Conclusion

### What we built

In this project, we implemented logistic regression completely from scratch using only NumPy, Pandas, and Matplotlib.

| Component | Purpose |
|-----------|---------|
| Sigmoid function | Converts linear output to probability |
| Cost function | Measures how wrong predictions are |
| Gradient descent | Learns optimal weights iteratively |
| L2 regularization | Prevents overfitting (adds weight penalty) |

### Key Results

- **Test Accuracy**: ~79%
- **Best λ**: 0 (regularization not needed for this dataset)
- **Best feature pair**: ST Depression vs Number of Vessels (75% accuracy with just 2 features)

### What we learned

1. **Data preparation matters**: Normalization helps gradient descent converge faster
2. **Stratified splits**: Keep class balance in train/test sets
3. **Feature selection**: Some features separate classes better than others
4. **Regularization**: Not always needed - depends on the overfitting situation
5. **Medical insight**: The model learned patterns that make clinical sense (blocked vessels -> higher risk)

### Limitations

- Logistic regression can only draw straight lines (linear boundaries)

---

