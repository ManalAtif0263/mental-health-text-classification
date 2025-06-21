# Mental Health Text Classification 🧠

Welcome to the **Mental Health Text Classification** repository! This project uses a multi-model machine learning approach to classify mental health-related content from social media text. We compare BERT, LSTM, and SVM models, achieving an accuracy of 83.6% across seven mental health categories. This work emphasizes research-focused methodologies with a strong commitment to ethical AI considerations.

[Download the latest release](https://github.com/ManalAtif0263/mental-health-text-classification/releases) to explore the models and results.

---

## Table of Contents

- [Introduction](#introduction)
- [Models Used](#models-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Ethical Considerations](#ethical-considerations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Introduction

Mental health issues are increasingly discussed on social media. Analyzing this text can provide insights into public sentiment and awareness. This repository presents a structured approach to classify mental health-related texts into categories such as anxiety, depression, and more.

We utilize three distinct models:

- **BERT**: A transformer-based model known for its contextual understanding.
- **LSTM**: A type of recurrent neural network that excels in sequence prediction.
- **SVM**: A robust classifier that separates data points in high-dimensional spaces.

The goal is to not only classify text but also to promote understanding and discussion around mental health topics.

---

## Models Used

### BERT (Bidirectional Encoder Representations from Transformers)

BERT leverages attention mechanisms to understand the context of words in relation to each other. This model is particularly effective for tasks that require understanding the nuances of language.

- **Pros**: High accuracy, context-aware.
- **Cons**: Requires significant computational resources.

### LSTM (Long Short-Term Memory)

LSTM networks are designed to remember information for long periods. They are effective for sequence data and are widely used in natural language processing tasks.

- **Pros**: Handles long-range dependencies well.
- **Cons**: Slower training times compared to some other models.

### SVM (Support Vector Machines)

SVMs are supervised learning models that find the optimal hyperplane to separate classes. They are effective for smaller datasets and can work well with high-dimensional data.

- **Pros**: Effective in high dimensions, robust against overfitting.
- **Cons**: Not suitable for large datasets.

---

## Dataset

The dataset used in this project consists of social media posts related to mental health. It includes various categories such as:

- Anxiety
- Depression
- Stress
- Eating Disorders
- Substance Abuse
- PTSD
- General Mental Health Awareness

Data was collected from public posts, ensuring compliance with ethical standards. We performed necessary preprocessing steps, including tokenization and normalization, to prepare the data for modeling.

---

## Installation

To get started, clone this repository:

```bash
git clone https://github.com/ManalAtif0263/mental-health-text-classification.git
cd mental-health-text-classification
```

Next, install the required packages:

```bash
pip install -r requirements.txt
```

Ensure you have the following libraries installed:

- TensorFlow or PyTorch (for model training)
- Scikit-learn (for SVM)
- Transformers (for BERT)
- NLTK or SpaCy (for text preprocessing)

---

## Usage

To run the models, use the following commands:

### For BERT

```bash
python run_bert.py --data_path <path_to_data> --output_dir <output_directory>
```

### For LSTM

```bash
python run_lstm.py --data_path <path_to_data> --output_dir <output_directory>
```

### For SVM

```bash
python run_svm.py --data_path <path_to_data> --output_dir <output_directory>
```

After training, you can evaluate the models using:

```bash
python evaluate.py --model <model_name> --data_path <path_to_test_data>
```

For more details, refer to the individual script comments.

---

## Results

The models were evaluated based on accuracy, precision, recall, and F1 score. Here are the results:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| BERT  | 83.6%    | 0.85      | 0.84   | 0.84     |
| LSTM  | 78.5%    | 0.80      | 0.79   | 0.79     |
| SVM   | 75.3%    | 0.76      | 0.75   | 0.75     |

These results highlight the strengths of BERT in understanding the context of mental health discussions.

---

## Ethical Considerations

This project emphasizes ethical AI practices. We acknowledge the sensitivity surrounding mental health topics and aim to handle data responsibly. Key considerations include:

- **Data Privacy**: We use publicly available data and anonymize any personal information.
- **Bias Mitigation**: We actively work to identify and reduce biases in our models.
- **Transparency**: We provide clear documentation and code for reproducibility.

We encourage users to consider these factors when using the models and data.

---

## Contributing

We welcome contributions from the community. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

Please ensure your code follows our style guidelines and includes relevant tests.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

We would like to thank the contributors of the libraries and tools that made this project possible:

- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

For more details on our latest releases, visit the [Releases section](https://github.com/ManalAtif0263/mental-health-text-classification/releases).

---

This repository serves as a foundation for exploring mental health classification using machine learning. We hope it contributes to the ongoing conversation about mental health awareness and support.