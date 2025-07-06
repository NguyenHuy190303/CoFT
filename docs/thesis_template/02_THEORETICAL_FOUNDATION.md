# THESIS WRITING GUIDE: CHAPTER 2 - THEORETICAL FOUNDATION

This chapter is where you demonstrate your deep understanding of the research field. The goal is to provide the reader with all the necessary background knowledge to understand the later parts of the thesis, while also positioning your work within the context of existing research.

---

### 2.1. Introduction to Core Concepts

**Purpose:**
To define and explain the fundamental terms, principles, and theories upon which the entire thesis is built.

**Required Content:**
- **Precise Definitions:** Provide standard academic definitions for central concepts (e.g., "A recommender system is...", "Deep learning is..."). Always cite the sources for these definitions.
- **Intuitive Explanations:** After the definition, use examples, analogies, or simple illustrations to clarify complex concepts.
- **Classification:** If there are different types or approaches within a concept, present them systematically (e.g., "Recommender systems are generally divided into three main types: collaborative filtering, content-based filtering, and hybrid...").

**Tips:**
- Imagine you are explaining to someone with technical knowledge but who is not an expert in your specific subfield.
- Use block diagrams and figures to visualize concepts. A picture is worth a thousand words.

---

### 2.2. State-of-the-Art Review

**Purpose:**
To systematically present the most important published research related to your topic. This section demonstrates that you have conducted a thorough literature survey and understand the "research gap" you are trying to fill.

**Required Content:**
- **Summarize Related Works:** Don't just list them; briefly summarize the objective, method, and key results of each work.
- **Group and Synthesize:** Group related studies by theme, method, or school of thought. Compare and contrast their approaches.
- **Point out Limitations:** Analyze the limitations or unanswered questions from previous studies. This is how you justify the need for your own topic. E.g., "Although the method by [Author A, Year] achieved good results, it requires immense computational resources, making practical deployment difficult."
- **Position Your Research:** Conclude this section by clearly stating how your work addresses these limitations or builds upon these foundations.

---

### 2.3. Foundation Models/Technologies

**Purpose:**
To delve into the technical details of the specific model(s), algorithm(s), or technology that you use as the foundation for your research.

**Required Content:**
- **Architectural Description:** Detail the model's architecture. Use architectural diagrams with clear annotations.
- **Mathematical Formulas:** If applicable, present the core mathematical formulas (e.g., loss function, prediction formula, update rule). Explain the meaning of each variable and symbol in the formula.
- **Algorithm/Operational Flow:** Use pseudo-code to describe the algorithm or the model's operational flow. This helps the reader understand the step-by-step logic.
- **Pros and Cons:** Discuss the strengths and weaknesses of this foundational technology/model in the context of the problem you are solving.

---

### 2.4. Evaluation Methodologies and Metrics

**Purpose:**
To describe and justify the methods you will use to evaluate your experimental results.

**Required Content:**
- **Evaluation Protocol:**
    - Describe how you split the data (e.g., hold-out, k-fold cross-validation, leave-one-out).
    - Explain why this protocol is suitable for your problem.
- **Evaluation Metrics:**
    - Clearly define each metric you use (e.g., Accuracy, Precision, Recall, F1-Score, RMSE, MAE, HR, NDCG).
    - Provide the mathematical formula for each metric.
    - Explain the meaning of the metric (e.g., "Precision measures the accuracy among positive predictions, while Recall measures the ability to find all positive samples").
    - Justify why you chose these metrics over others. What aspect of performance are you interested in measuring? 