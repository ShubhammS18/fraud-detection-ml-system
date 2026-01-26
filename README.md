# End-to-End fraud-detection-ml-system

* This project is a production-ready Machine Learning system built using the IEEE-CIS Fraud Detection dataset. Instead of just writing a script, I engineered a full lifecycle pipeline—from raw data exploration to a containerized API with automated CI/CD.

### 🚀 The "Robustness-First" Engineering Philosophy - 

* This project evolved from a simple baseline to a high-security **Identity-Enhanced System**. After deploying the initial baseline, I identified silent failures where the model was too trusting of high-value transactions from "clean" devices. This led to **Phase 2: The Robustness Upgrade**, where I built a hybrid decision engine that balances ML probability with hardcoded financial safety nets.

### 🛠️ Project Architecture -

The system is built with a modern MLOps stack:

* **Model:** Identity-Enhanced LightGBM trained on merged Transaction + Identity data.

* **Robustness Layer:** Custom `FeatureBuilder` calculating real-time **Interaction Ratios** (comparing transaction amounts to device-level historical maximums).

* **API:** FastAPI for single-record, low-latency inference.

* **DevOps:** Docker for environment parity and GitHub Actions for CI/CD.


### 📈 The Development Journey -

1. **Identity & Advanced EDA (01_eda.ipynb)**

   * I merged the Transaction and Identity datasets to see if "who" is making the transaction (Device, Browser, OS) matters as much as "what" the transaction is.

   * **The Pivot:** I realized that looking at identity alone wasn't enough. Fraudsters can use clean devices, so I started focusing on **Behavioral Anomalies**—like a device suddenly spending 100x its usual amount.

2. **The Robustness Upgrade (04_final_model_training.ipynb)**

   I moved past the baseline to build a "Battle-Hardened" model.
   * **The Safety Net:** I created a `reference_stats.json` file. This acts as the system's "memory," allowing it to calculate if a transaction is a massive outlier compared to historical norms.
   * **Threshold Optimization:** I didn't just use the default 0.5. I optimized the threshold to **0.474** to get the best balance between catching fraud and not bothering legitimate customers.

3. **Productionizing the Hybrid Logic (src/models/predict.py)**

   I updated the code to be "Defensive."
   * **Two-Tier Check:** The system now does a double-check. Even if the ML model thinks a transaction looks "clean," if the amount is over $100k or shows a high Spend-Ratio, the system automatically flags it for a human to check.
   * **Type Safety:** I added code to handle "garbage data" (like strings instead of numbers) so the API doesn't crash in a real-world environment.

4. **Testing & Automated Deployment (.github/workflows/)**

   I pushed the automation even further:
   * **Adversarial Testing:** I added a specific test case for a $35 Million "Clean Identity" attack. If the system approves it, the CI/CD fails. This ensures I never deploy a "weak" model again.
   * **Full Dockerization:** Every push now triggers a full rebuild, packaging the new model, the identity schemas, and the robustness stats into a single ready-to-deploy image.

### ⚙️ How to Run -

You don't need to install Python or dependencies to test this. If you have Docker, run:


1. **Pull the image from Docker Hub**

`docker pull <imshubham18>/fraud-detection-api:latest`

2. **Run the container**

`docker run -p 8000:8000 <imshubham18>/fraud-detection-api:latest`

Then visit `http://localhost:8000/docs` to interact with the API via Swagger UI.

### 🛡️ Self-Correction & Learning (The "Engineering" Part) -

During the transition from the baseline to the production-ready system, I encountered and solved several architectural challenges:

  * **The "Clean Device" Trap & Logic Overlays:** I initially relied on the model's high validation accuracy (99%), but adversarial testing revealed a critical "silent failure": fraudsters using high-reputation device fingerprints (e.g., Chrome on Windows) could bypass the identity filters. I recognized that even a highly optimized model needs a Business Logic Overlay. I implemented a deterministic "Safety Net" for high-exposure transactions (>$100k), ensuring that financial risk is mitigated even when the ML probability is low due to identity spoofing.

  * **Feature Parity & Serialization Integrity:** A major challenge was ensuring that the complex Interaction Ratios calculated during training were perfectly replicated during real-time inference. I moved away from "ad-hoc" preprocessing and implemented a centralized reference_stats.json and a unified FeatureBuilder class. This ensures Training-Serving Parity, preventing the "Data Drift" issues that typically crash models when they move from a research notebook to a live Docker container.

### 📊 Final System Outcomes -

After running the final system against 500,000+ test records, here is the proof that the engineering worked:
* **Total Records Processed:** 506,691
* **ML Automated Blocks:** 9,649 (High-confidence fraud)
* **Manual Review Flags:** 52 (Sneaky outliers caught by the safety net)
* **Outlier Success Rate:** **100%** (Successfully caught the $35M simulated attack that tricked my baseline model).