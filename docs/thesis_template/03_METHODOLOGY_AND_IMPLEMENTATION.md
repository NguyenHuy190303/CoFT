# THESIS WRITING GUIDE: CHAPTER 3 - METHODOLOGY AND IMPLEMENTATION

This chapter is the "heart" of the practical part of your thesis. It describes in detail **how you did it** to answer the research questions. The highest goal of this chapter is to provide enough information for another researcher to independently reproduce your work.

---

### 3.1. System Design / Experimental Setup

**Purpose:**
To provide a high-level overview of the architecture of the system you built or the overall process of the experiments you conducted.

**Required Content:**
- **Overall Architecture Diagram:** Use a block diagram or data flow diagram to illustrate the main components of the system and how they interact.
    - E.g., Input -> Data Preprocessing -> Feature Extraction -> Model Training -> Evaluation -> Output.
- **Component Description:** Briefly explain the function of each block in the diagram.
- **Workflow:** Describe the sequence of steps, from obtaining raw data to the final results.

---

### 3.2. Tools and Technologies

**Purpose:**
To list and justify the choice of software tools, libraries, and hardware used.

**Required Content:**
- **Programming Language:** E.g., Python (version 3.9).
- **Key Libraries/Frameworks:**
    - E.g., TensorFlow (version 2.x), PyTorch, Scikit-learn, Pandas, NumPy.
    - Briefly explain the role of each library: "Pandas was used for data reading and manipulation, while TensorFlow was used to build and train the deep learning model."
- **Development Environment:**
    - E.g., Jupyter Notebook, Google Colab, Visual Studio Code.
    - Explain why that environment was suitable (e.g., "Google Colab was chosen for its provision of free GPU resources...").
- **Hardware (if important):**
    - Especially important for performance-intensive tasks.
    - E.g., "Experiments were conducted on an NVIDIA RTX 4090 GPU with 24GB of VRAM."

---

### 3.3. Dataset Description and Preprocessing

**Purpose:**
To describe in detail the dataset you used and all the steps you took to clean and prepare it for the model. This is an **extremely important** section for transparency and reproducibility.

**Required Content:**
- **Data Source:** Where did the data come from? Provide a download link if it's a public dataset.
- **Raw Data Statistics:**
    - Present in a table.
    - E.g., number of samples, number of users/items, number of features, class distribution, etc.
- **Step-by-step Preprocessing:**
    - **Data Cleaning:** How did you handle missing values, noisy data, or outliers?
    - **Feature Extraction/Selection:** How did you create new features from raw data or select a subset of important features?
    - **Data Transformation:** How did you normalize, encode (e.g., one-hot encoding, label encoding), or re-index the data?
    - **Provide Pseudo-code or Code Snippets:** For complex processing steps, including a short piece of code can help clarify the idea.
- **Post-processing Data Statistics:** Present a similar statistics table as for the raw data to show the changes after preprocessing.

---

### 3.4. Model Training and Evaluation

**Purpose:**
To describe exactly how you trained and evaluated your model.

**Required Content:**
- **Model Configuration:** What were the hyperparameters used?
    - Present in a table.
    - E.g., learning rate, batch size, number of epochs, embedding size, dropout rate, activation function, etc.
    - If you performed hyperparameter tuning, describe it.
- **Training Process:**
    - The optimizer used (e.g., Adam, SGD).
    - The loss function being optimized.
    - How the model was trained (e.g., from scratch, or fine-tuning from a pre-trained model).
- **Evaluation Process:**
    - Reiterate how the evaluation method mentioned in Chapter 2 was applied in practice.
    - E.g., "For each user, their last interaction was held out as the test set, and the remaining interactions were used as the training set."
    - How the metrics were calculated and aggregated (e.g., "The HR@10 metric was calculated for each user and then averaged over the entire test set."). 