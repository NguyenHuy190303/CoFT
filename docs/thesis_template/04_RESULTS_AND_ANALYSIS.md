# THESIS WRITING GUIDE: CHAPTER 4 - RESULTS AND ANALYSIS

This is the chapter where you present what you found through your experiments. The goal is not just to present numbers, but also to analyze and interpret them to answer the research questions posed in Chapter 1.

---

### 4.1. Presentation of Experimental Results

**Purpose:**
To report objectively and systematically the results obtained from the experimental process described in Chapter 3.

**Required Content:**
- **Direct Link to Research Questions (RQs):** The structure of this chapter should be driven by the RQs. Each subsection can be dedicated to answering one RQ.
    - E.g., Start the subsection with: "To answer the first research question (RQ1): 'Does model X perform better than Y?', we conducted experiments and obtained the results shown in Table 4.1."
- **Use of Tables and Figures:**
    - **Tables:** Used to present numerical data accurately (e.g., values of HR, NDCG, Accuracy metrics). Tables must have clear titles and full labels for rows and columns.
    - **Charts/Figures:** Used to visualize trends and compare performance. E.g., bar charts to compare models, line charts to show learning curves. Figures must also have clear titles and captions.
    - **Rule:** Every table and figure must be numbered and referenced in the text (e.g., "As shown in Figure 4.2...", "Table 4.1 summarizes the results...").
- **Description of Results:** Write paragraphs describing what the reader should see in the table or figure. Just describe, do not analyze deeply yet.
    - E.g., "From Table 4.1, it can be seen that Model A achieved an HR@10 of 0.75, which is higher than the 0.68 of Model B. A similar trend was also observed for the NDCG@10 metric."

---

### 4.2. Analysis and Discussion of Results

**Purpose:**
To interpret the meaning of the results. This is where you demonstrate your critical thinking and deep understanding. **Why** are the results the way they are?

**Required Content:**
- **Interpreting "Why?":**
    - Try to explain the reasons behind the observed results.
    - E.g., "The reason Model A performs better may be due to its attention mechanism, which allows it to focus on more important features, whereas Model B treats all features with equal weight."
- **Comparison with State-of-the-Art:**
    - How do your results compare to those published in the papers you surveyed in Chapter 2?
    - If better, why? If worse, why?
- **Relating back to Theory:**
    - Do these results support or contradict the theories presented in Chapter 2?
- **Error Analysis:**
    - Examine cases where your model performed poorly. This demonstrates deep insight and can suggest future improvements.
    - E.g., "The model tends to make incorrect predictions for users with few interactions, indicating that the 'cold-start' problem has not been fully resolved."

---

### 4.3. Threats to Validity

**Purpose:**
To demonstrate honesty and academic integrity by acknowledging factors that may have influenced or limited your research findings.

**Required Content:**
- **Internal Validity:** Factors within the experiment that could affect the results.
    - E.g., Was there a bug in the code implementation? Were the chosen hyperparameters optimal?
- **External Validity:** Factors related to the generalizability of the results.
    - E.g., "These results have only been validated on the [Dataset Name] dataset, which has its own unique characteristics. Application to other domains may yield different results."
    - "The study only used one type of model architecture; other architectures might perform better."
- **How you mitigated these threats:**
    - Describe the steps you took to make your research as reliable as possible.
    - E.g., "To minimize the impact of hyperparameter selection, we conducted a small grid search." 