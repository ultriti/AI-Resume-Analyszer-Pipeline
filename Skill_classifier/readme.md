# AI Job Category Classification System

## Project Overview, Dataset Design, Architecture, Structure, and End-to-End Flow

## 1. Project Overview

This project is an NLP-based job-category classification system. It
takes structured job-posting information---especially the job title,
experience level, and required skills---and predicts which broad
professional category the role belongs to.

The current dataset contains job postings from seven categories:

1.  Business
2.  Cloud & Infrastructure
3.  Data & AI
4.  Design
5.  Marketing
6.  Product & Management
7.  Software Engineering

The current model-training notebook uses `required_skills` as its
primary text input (`X`) and `category` as its target label (`y`). It
combines CSV files for the different categories, tokenizes the skills
text, converts it into padded integer sequences, and trains recurrent
neural-network models.

The supplied implementation experiments with: - Simple RNN - LSTM - GRU

It evaluates the models on a held-out test split and saves a trained
Keras model and its tokenizer as artifacts.

### Current scope

The implementation demonstrated by the supplied code is a **job-category
classifier**. It is not, by itself, a complete resume parser,
resume-to-job matching engine, or job recommendation platform. Those can
be added as later modules.

A future resume-analysis workflow could use this classifier to identify
the likely job category of a resume or job description, then use
separate skill extraction, experience validation, and matching logic to
compare a resume against a selected job.

------------------------------------------------------------------------

## 2. Problem Statement

Job descriptions contain a mixture of role names, skills, experience
requirements, employment details, work arrangements, and descriptive
text. Manually assigning every job to a broad career category is
time-consuming and can be inconsistent.

The objective of this project is to learn patterns in job-posting text
and predict the category of a job.

For example, a posting containing skills such as:

``` text
Python; Pandas; Power BI; Reporting; A/B Testing; Statistics; Tableau
```

may be classified as **Data & AI**.

A posting containing:

``` text
React; Node.js; Express.js; PostgreSQL; JavaScript; REST APIs
```

may be classified as **Software Engineering**.

The model learns associations between the training text and the category
labels. Its prediction is a model estimate, not a guarantee that a role
belongs exclusively to one category.

------------------------------------------------------------------------

## 3. Dataset

### 3.1 Dataset source and organization

The supplied training code loads multiple CSV files from a `dataset/`
directory. The files are separated by source category and then
concatenated into one Pandas DataFrame.

The code references files similar to:

``` text
dataset/
├── aiml pre.csv
├── business pre.csv
├── cloud_infrastructure pre.csv
├── design pre.csv
├── developer pre.csv
├── marketing pre.csv
└── product_management pre.csv
```

The notebook also uses Hugging Face `datasets.load_dataset` to load the
CSV files as named splits, and uses `glob` plus Pandas to combine CSV
files for model training.

The combined dataset is the source for the train/test split.

### 3.2 Dataset schema

The job-posting dataset uses the following fields:

  -------------------------------------------------------------------------
  Field                     Meaning                 Example
  ------------------------- ----------------------- -----------------------
  `job_title`               Name of the advertised  Junior Data Analyst
                            role                    

  `category`                Target job family to    Data & AI
                            predict                 

  `experience_level`        Human-readable          Junior
                            seniority               

  `experience_level_rank`   Numeric encoding of     1
                            seniority               

  `employment_type`         Employment arrangement  Contract

  `work_setting`            Work arrangement        Remote

  `industry`                Industry or business    EdTech
                            domain                  

  `min_experience_years`    Minimum requested       1
                            experience              

  `max_experience_years`    Maximum requested       3
                            experience              

  `avg_experience_years`    Representative          2
                            experience value        

  `required_skills`         Semicolon-separated     Pandas; Power BI; SQL
                            required skills         

  `skill_count`             Number of listed skills 3

  `description`             Original generated job  Full description text
                            description             

  `cleaned_description`     Normalized/cleaned      Cleaned description
                            description             text

  `nlp_corpus_text`         Combined text prepared  Title, category, level,
                            for NLP                 skills, and description
  -------------------------------------------------------------------------

### 3.3 Example records

A representative record may look like:

``` csv
job_title,category,experience_level,experience_level_rank,employment_type,work_setting,industry,min_experience_years,max_experience_years,avg_experience_years,required_skills,skill_count
Junior Data Analyst,Data & AI,Junior,1,Contract,Remote,EdTech,1,3,2,"Pandas; Power BI; Reporting; A/B Testing; Statistics; Tableau; Data Cleaning",7
Intern Data Engineer,Data & AI,Intern,0,Full-time,Hybrid,EdTech,0,1,0.5,"Hadoop; AWS; Airflow; Kafka; Spark; Python",6
Senior Data Analyst,Data & AI,Senior,3,Contract,Hybrid,EdTech,6,10,8,"Python; Data Cleaning; Statistics; Excel; SQL",5
Principal Data Engineer,Data & AI,Principal,5,Full-time,Remote,SaaS,12,20,16,"ETL; AWS; Data Modeling; Python; Spark; Kafka",6
```

The values above illustrate the schema and the kind of records supplied.
They do not imply that every dataset row has the same values or that the
dataset contains only these examples.

### 3.4 Which columns are used by the current model?

The supplied notebook explicitly sets:

``` python
X = df["required_skills"].fillna("")
y = df["category"]
```

Therefore:

-   **Input (`X`):** `required_skills`
-   **Target (`y`):** `category`

The other columns are available in the dataset but are not currently
passed into the model as separate features.

The notebook contains a commented-out alternative that would use several
columns:

``` python
# X = df[[
#     "category",
#     "experience_level_rank",
#     "required_skills",
#     "nlp_corpus_text"
# ]]
```

That alternative should not be used as written because `category` is the
target label. Including it in the input would leak the answer into the
features. A multi-column model should instead select legitimate input
fields such as `job_title`, `experience_level_rank`, `required_skills`,
and/or `nlp_corpus_text`, while keeping `category` exclusively as the
target.

### 3.5 Dataset quality checks

Before training, the dataset should be checked for:

-   Missing target labels.
-   Empty or missing skill strings.
-   Duplicate job postings.
-   Unexpected category names or spelling variations.
-   Inconsistent skill separators.
-   Invalid experience ranges, such as minimum experience greater than
    maximum experience.
-   Invalid or inconsistent `experience_level_rank` values.
-   Category imbalance.
-   Duplicate or near-duplicate records that could appear in both train
    and test sets.

The supplied notebook checks some null-related conditions, but a
production pipeline should implement explicit validation and report the
results.

### 3.6 Important evaluation consideration

The notebook uses a random train/test split. If multiple rows are
generated from very similar templates, random splitting can put
near-duplicate descriptions in both training and testing data. This can
make the reported test accuracy look better than performance on
genuinely new job descriptions.

For a stronger evaluation: - Remove exact duplicates. - Consider
grouping near-duplicate postings before splitting. - Use a stratified
split so category proportions are represented in both sets. - Keep a
final untouched test set for the final model evaluation.

------------------------------------------------------------------------

## 4. Category Taxonomy

The model predicts one of seven broad categories.

  -----------------------------------------------------------------------
  Model category                      Typical roles
  ----------------------------------- -----------------------------------
  Business                            Business analyst, operations
                                      analyst, business development roles

  Cloud & Infrastructure              Cloud engineer, DevOps engineer,
                                      infrastructure engineer

  Data & AI                           Data analyst, data engineer, data
                                      scientist, ML engineer, AI engineer

  Design                              UI designer, UX designer, product
                                      designer

  Marketing                           SEO analyst, digital marketing
                                      specialist, growth marketer

  Product & Management                Product manager, associate product
                                      manager, product operations roles

  Software Engineering                Backend developer, frontend
                                      developer, full-stack developer,
                                      software engineer
  -----------------------------------------------------------------------

These examples explain the intended taxonomy. The actual category label
for each training row comes from the dataset.

The labels in the supplied notebook are:

``` python
labels = [
    "Business",
    "Cloud & Infrastructure",
    "Data & AI",
    "Design",
    "Marketing",
    "Product & Management",
    "Software Engineering"
]
```

**Consistency requirement:** the order of this list must exactly match
the class-index mapping used by the fitted label encoder. Prefer saving
and loading the fitted `LabelEncoder` rather than manually maintaining a
second label list.

------------------------------------------------------------------------

## 5. Technology Stack

### Data processing

-   Python
-   Pandas
-   NumPy
-   CSV datasets
-   Hugging Face `datasets` for CSV loading

### Machine learning and NLP

-   TensorFlow / Keras
-   Scikit-learn
-   Keras `Tokenizer`
-   `pad_sequences`
-   `LabelEncoder`
-   RNN, LSTM, and GRU layers

### Model artifacts

-   Keras `.keras` model file
-   Pickle tokenizer file

The supplied code does not demonstrate a web API or frontend. Those are
possible application layers around the trained classifier, but should be
documented as separate components if they are implemented.

------------------------------------------------------------------------

## 6. Proposed Project Structure

The following is a recommended maintainable structure based on the
supplied training code. It is a proposed organization, not a claim that
these exact files already exist in the uploaded project.

``` text
job-category-classifier/
│
├── dataset/
│   ├── aiml pre.csv
│   ├── business pre.csv
│   ├── cloud_infrastructure pre.csv
│   ├── design pre.csv
│   ├── developer pre.csv
│   ├── marketing pre.csv
│   └── product_management pre.csv
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_dataset.py
│   │   └── validate_dataset.py
│   │
│   ├── preprocessing/
│   │   ├── text_cleaning.py
│   │   ├── tokenizer.py
│   │   └── sequence_padding.py
│   │
│   ├── models/
│   │   ├── rnn_model.py
│   │   ├── lstm_model.py
│   │   ├── gru_model.py
│   │   └── model_factory.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   └── callbacks.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── metrics.py
│   │
│   └── inference/
│       ├── load_artifacts.py
│       └── predict.py
│
├── Artifacts/
│   ├── job_category_model.keras
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   └── metadata.json
│
├── tests/
│   ├── test_dataset_validation.py
│   ├── test_preprocessing.py
│   └── test_prediction.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Responsibilities of the main directories

  -----------------------------------------------------------------------
  Directory                           Responsibility
  ----------------------------------- -----------------------------------
  `dataset/`                          Stores the category-specific CSV
                                      files used to build the combined
                                      dataset

  `notebooks/`                        Supports exploration, experiments,
                                      training, and evaluation

  `src/data/`                         Loads and validates raw job-posting
                                      data

  `src/preprocessing/`                Cleans text and converts it into
                                      model-ready sequences

  `src/models/`                       Defines the neural-network
                                      architectures

  `src/training/`                     Runs model training and manages
                                      callbacks

  `src/evaluation/`                   Calculates and reports evaluation
                                      metrics

  `src/inference/`                    Loads saved artifacts and predicts
                                      categories for new text

  `Artifacts/`                        Stores the model and all
                                      preprocessing objects needed for
                                      inference

  `tests/`                            Contains automated checks for data,
                                      preprocessing, and prediction
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 7. End-to-End Data and Model Flow

``` text
Category-specific CSV files
          |
          v
Load CSV files
(Pandas / Hugging Face Datasets)
          |
          v
Combine into one DataFrame
          |
          v
Inspect and validate dataset
(columns, missing values, labels, duplicates)
          |
          v
Select input and target
X = required_skills
y = category
          |
          v
Train / test split
          |
          +--------------------------+
          |                          |
          v                          v
      Training data               Test data
          |                          |
          v                          |
Fit Tokenizer on training text      |
          |                          |
          v                          |
Convert text to integer sequences   |
          |                          |
          +------------+-------------+
                       |
                       v
              Pad / truncate sequences
                  to length 50
                       |
                       v
              Encode category labels
                       |
                       v
          Train RNN / LSTM / GRU models
                       |
                       v
              Evaluate on test set
                       |
                       v
              Compare model metrics
                       |
                       v
              Select final model
                       |
                       v
        Save model + tokenizer + label encoder
                       |
                       v
             Load artifacts for inference
                       |
                       v
       New job skills / job-posting text
                       |
                       v
       Tokenize and pad using saved settings
                       |
                       v
             Neural-network prediction
                       |
                       v
       Predicted category (and probabilities)
```

### Flow explanation

#### Step 1 --- Load the category datasets

The notebook loads individual CSV files and combines them:

``` python
files = glob.glob("dataset/*.csv")

df = pd.concat(
    [pd.read_csv(file) for file in files],
    ignore_index=True
)
```

This creates one DataFrame containing records from all available
categories.

#### Step 2 --- Explore the dataset

The notebook inspects: - Total row count. - Column names. - Category
distribution. - Sample records. - The category and skill columns.

This is the initial exploratory data analysis stage.

#### Step 3 --- Select model input and target

The current implementation uses required skills as the text input and
category as the output label:

``` python
X = df["required_skills"].fillna("")
y = df["category"]
```

The model therefore learns to map a skills string to one of the seven
category labels.

#### Step 4 --- Split the data

The notebook uses an 80/20 train/test split with `random_state=42`:

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

For a robust classification experiment, use `stratify=y` where class
counts permit it. Fit all learned preprocessing only on the training
partition.

#### Step 5 --- Tokenize the text

Keras `Tokenizer` builds a vocabulary from the training text:

``` python
max_words = 10000

tokenizer = Tokenizer(
    num_words=max_words,
    oov_token="<unk>"
)
tokenizer.fit_on_texts(X_train)
```

Each recognized word is mapped to an integer. Words outside the retained
vocabulary are handled by the out-of-vocabulary token.

The tokenizer must be fitted on training text only and reused unchanged
for test data and inference.

#### Step 6 --- Convert text into sequences

``` python
train_sequence = tokenizer.texts_to_sequences(X_train)
test_sequence = tokenizer.texts_to_sequences(X_test)
```

For example, a skill string is transformed from words into a sequence of
token IDs. The actual IDs depend on the fitted tokenizer vocabulary.

#### Step 7 --- Encode category labels

`LabelEncoder` maps category strings to integer class IDs:

``` python
label_encoder = LabelEncoder()

train_labels = label_encoder.fit_transform(y_train)
test_labels = label_encoder.transform(y_test)
```

The mapping must be saved with the model. At prediction time, the
predicted class index is converted back using the same fitted encoder.

#### Step 8 --- Pad sequences

The supplied implementation uses a maximum sequence length of 50:

``` python
max_padding = 50

padded_train_sequence = pad_sequences(
    train_sequence,
    maxlen=max_padding,
    padding="post",
    truncating="post"
)

padded_test_sequence = pad_sequences(
    test_sequence,
    maxlen=max_padding,
    padding="post",
    truncating="post"
)
```

Short sequences are padded at the end; long sequences are truncated at
the end. The same maximum length and padding settings must be used
during inference.

#### Step 9 --- Train the recurrent models

The notebook trains three architectures.

**Simple RNN** - Embedding layer: 128-dimensional word embeddings. -
SimpleRNN layer: 128 units, returning sequences. - Dropout: 0.5. -
SimpleRNN layer: 64 units. - Dropout: 0.5. - Dense softmax output: one
unit per category.

**LSTM** - Embedding layer: 128-dimensional word embeddings. - LSTM
layer: 128 units, returning sequences. - Dropout: 0.5. - LSTM layer: 64
units. - Dropout: 0.5. - Dense softmax output.

**GRU** - Embedding layer: 128-dimensional word embeddings. - GRU layer:
128 units, returning sequences. - Dropout: 0.5. - GRU layer: 64 units. -
Dropout: 0.5. - Dense softmax output.

All three models use: - Adam optimizer. - Sparse categorical
cross-entropy loss. - Accuracy as a metric. - Early stopping that
monitors validation loss, with patience 3 and best-weight restoration.

The supplied notebook uses the test split as `validation_data` during
training. For a cleaner evaluation, split the training partition again
into training and validation subsets, use validation data for early
stopping, and reserve the test partition for final evaluation only.

#### Step 10 --- Evaluate and compare

Each model is evaluated on the held-out test arrays. The notebook
creates a comparison table containing loss and accuracy, sorted by
accuracy.

Accuracy alone is not sufficient when categories are imbalanced. A more
complete evaluation should also report: - Precision, recall, and
F1-score per category. - Macro and weighted F1-score. - Confusion
matrix. - Support (number of test examples) per category.

#### Step 11 --- Save model artifacts

The supplied code saves a Keras model and tokenizer. The model and
tokenizer must be treated as a matched pair.

For reproducible inference, also save: - The fitted label encoder. -
Maximum vocabulary size. - Sequence length. - Padding and truncation
settings. - Category mapping. - Model version and training metadata.

------------------------------------------------------------------------

## 8. Inference Flow

Once the model is trained, a new job posting can be classified.

### Input example

``` text
Senior NLP Engineer
Python; PyTorch; Transformers; Hugging Face; NLP; BERT
```

### Inference steps

1.  Receive the job title and/or skill text.
2.  Construct the same text representation used during training.
3.  Convert text to token IDs using the saved tokenizer.
4.  Pad or truncate the sequence to 50 tokens.
5.  Pass the sequence to the saved Keras model.
6.  Read the softmax probability for each category.
7.  Select the category with the highest probability.
8.  Convert the class index into its category name using the saved label
    encoder.

Example inference pattern:

``` python
sequence = tokenizer.texts_to_sequences([input_text])

padded = pad_sequences(
    sequence,
    maxlen=50,
    padding="post",
    truncating="post"
)

probabilities = loaded_model.predict(padded, verbose=0)[0]

predicted_index = int(np.argmax(probabilities))
predicted_category = label_encoder.inverse_transform(
    [predicted_index]
)[0]
```

The probability vector can be returned to an application as category
scores. These are model confidence scores from the classifier; they
should not automatically be interpreted as calibrated real-world
probabilities.

------------------------------------------------------------------------

## 9. Important Implementation Notes and Corrections

The supplied notebook is useful as an experimental baseline. The
following points should be addressed before treating it as a production
training pipeline.

### 9.1 Model and tokenizer filename mismatch

The training code saves:

``` text
Artifacts/BiGRU_Model_v2.keras
Artifacts/tokenizer_v2.pkl
```

The later loading code expects:

``` text
Artifacts/BiGRU_Model.keras
Artifacts/tokenizer.pkl
```

These are different filenames. Align the save and load paths, or use a
single artifact manifest/configuration so inference cannot accidentally
load an older model or tokenizer.

### 9.2 The saved model is not necessarily the selected best model

The notebook creates a comparison table for RNN, LSTM, and GRU, but then
saves `gru_model` directly. It does not programmatically select the best
model from the comparison table.

The final training script should select the model based on a chosen
validation metric, then evaluate that selected model once on the
untouched test set.

### 9.3 Label mapping should not be manually duplicated

The notebook defines a manual `labels` list for decoding predictions.
The safer approach is to save the fitted `LabelEncoder` and call
`inverse_transform`. This prevents a mismatch between the model output
indices and the displayed category names.

### 9.4 Validation data should be separate from final test data

The supplied code uses `padded_test_sequence` as validation data for
`fit()` and later evaluates on that same data. This means the test set
influences early-stopping decisions.

Create three partitions: - Training set: fit model weights and
tokenizer. - Validation set: early stopping and model selection. - Test
set: final unbiased evaluation.

### 9.5 Feature design

The current classifier only uses `required_skills`. This is a valid
baseline, but some roles may share skills across categories.

A future experiment can compare: - Skills only. - Job title plus
skills. - `nlp_corpus_text`. - A structured model combining text
features with numeric/categorical fields.

Do not include the target `category` in the input features.

### 9.6 Multi-label or multi-category scoring

The current Dense softmax layer represents a single-label, seven-class
classification task: one predicted category per input.

If a future product requirement is to show multiple category match
scores, the seven softmax outputs can be displayed as a ranked
distribution, provided they are clearly described as scores from a
single-label classifier. If the task changes to genuinely independent
multi-label classification, the target representation, output
activation, loss, and evaluation strategy must also change.

### 9.7 Training metrics

The notebook's model comparison is based on accuracy and loss. Add
per-class precision, recall, F1, a confusion matrix, and a final test
report. Consider calibration checks before describing softmax outputs as
probabilities.

------------------------------------------------------------------------

## 10. How This Can Become a Resume Analysis Platform

The current job-category classifier can be one component of a broader
resume-analysis system.

A possible future workflow is:

``` text
Resume upload
     |
     v
Text extraction from PDF / DOCX
     |
     v
Resume section and skill extraction
     |
     v
Job-category classification
     |
     v
Experience and skill normalization
     |
     v
Compare resume with selected job description
     |
     +-----------------------------+
     |                             |
     v                             v
Required-skill coverage      Experience requirement check
     |                             |
     +--------------+--------------+
                    |
                    v
             Match analysis
                    |
                    v
      Missing skills / strengths / recommendations
```

This is an extension design, not functionality demonstrated by the
supplied notebook.

A job-matching component should compare the candidate's extracted skills
and experience against a specific job posting. It should not treat the
job-category classifier's output as a job-match percentage.

Potential future modules:

  -----------------------------------------------------------------------
  Module                              Responsibility
  ----------------------------------- -----------------------------------
  Resume parser                       Extract text and sections from
                                      uploaded resumes

  Skill extractor                     Identify skills, aliases, and skill
                                      categories

  Job classifier                      Predict one of the seven job
                                      categories

  Experience analyzer                 Compare candidate experience with
                                      job requirements

  Job matcher                         Compare resume evidence with a
                                      selected job description

  Recommendation engine               Suggest relevant skills or roles
                                      based on identified gaps

  API layer                           Expose parsing, classification, and
                                      matching functionality

  Frontend                            Present analysis, scores, evidence,
                                      and recommendations
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 11. Running the Current Experiment

The supplied files do not include a complete dependency lockfile or a
production entry point. The following is an illustrative setup for the
notebook/script environment.

### Install dependencies

``` bash
pip install tensorflow pandas numpy scikit-learn datasets
```

### Run

Open and execute the training notebook in order, or move the notebook
cells into a Python training script.

Ensure the CSV paths point to the actual `dataset/` directory.

### Expected artifacts

The supplied code currently writes:

``` text
Artifacts/
├── BiGRU_Model_v2.keras
└── tokenizer_v2.pkl
```

The loading code must be updated to use these exact names, or the save
code must be changed to the filenames expected by the loader. Add the
fitted label encoder before relying on the model for repeatable
inference.

------------------------------------------------------------------------

## 12. Summary

The project currently demonstrates an end-to-end experimental NLP
classification pipeline:

1.  Load category-specific job-posting CSV files.
2.  Combine the records into a unified dataset.
3.  Use `required_skills` as the input text and `category` as the
    target.
4.  Split the data into training and test partitions.
5.  Fit a Keras tokenizer on training text.
6.  Convert text into integer sequences and pad them to 50 tokens.
7.  Encode category labels.
8.  Train RNN, LSTM, and GRU classifiers.
9.  Evaluate and compare their test metrics.
10. Save a Keras model and tokenizer for inference.
11. Load the artifacts and classify new job-skill text.

The dataset also contains job title, seniority, employment type, work
setting, industry, experience range, skill count, and description
fields. These fields create opportunities for richer classification and
later resume-to-job matching, but they are not all used by the current
model.

**Core distinction:** the implemented model predicts a broad job
category from skills text. Resume parsing, experience matching,
skill-gap analysis, job recommendations, and a full web application are
separate capabilities that require additional implementation.
