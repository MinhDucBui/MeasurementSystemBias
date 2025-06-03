# 📏 On Generalization across Measurement Systems: LLMs Entail More Test-Time Compute for Underrepresented Cultures

### 🧠 Abstract

Measurement systems (e.g., currencies) differ across cultures, but the conversions between them are well defined so that humans can state facts using any measurement system of their choice. Being available to users from diverse cultural backgrounds, large language models (LLMs) should also be able to provide accurate information irrespective of the measurement system at hand. Using newly compiled datasets we test if this is the case for seven open-source LLMs, addressing three key research questions: 
(RQ1) What is the default system used by LLMs for each type of measurement? (RQ2) Do LLMs' answers and their accuracy vary across different measurement systems? (RQ3) Can LLMs mitigate potential challenges w.r.t. underrepresented systems via reasoning? 
Our findings show that LLMs default to the measurement system predominantly used in the data. Additionally, we observe considerable instability and variance in performance across different measurement systems. While this instability can in part be mitigated by employing reasoning methods such as chain-of-thought (CoT), this implies longer responses and thereby significantly increases test-time compute (and inference costs), marginalizing users from cultural backgrounds that use underrepresented measurement systems.

### 🗂️ Directory Structure

```
.
├── data
│   ├── raw_data       # Contains raw numerical data (weights, prices, distances)
│   └── prompts        # Contains prompt examples for evaluation
├── scripts
│   └── inference.py   # Main script to run model inference
```

## 🚀 Running Inference

### 1. Install Dependencies
To get started, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Inference

To run inference with your preferred LLM (e.g., LLaMA 3.3 70B), execute:

```bash
python scripts/inference.py --model_name meta-llama/Llama-3.3-70B-Instruct --gt_file data/prompts/your_prompt_file.csv
```


### ⚙️ Arguments

- ```--model_name```: The Hugging Face model identifier.
- ```--gt_file```: Path to the prompt file (located in data/prompts).

You can find example prompts for the default setup (Figure 2) in folder ```data/prompts```.

📌 If your prompt file requires CoT reasoning, add:

```bash
--modes "free_generation"
```
