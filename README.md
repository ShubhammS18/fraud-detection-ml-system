# End-to-End fraud-detection-ml-system

* This project is a production-ready Machine Learning system built using the IEEE-CIS Fraud Detection dataset. Instead of just writing a script, I engineered a full lifecycle pipeline—from raw data exploration to a containerized API with automated CI/CD.

### 🚀 The "Baseline-First" Engineering Philosophy -

* I intentionally built this project in two phases. Phase 1 (this version) focuses on a robust "Baseline Model" using the Transaction dataset. By starting with a baseline, I was able to verify the system architecture, debugging the environment, and establishing a Green CI/CD pipeline before increasing model complexity.

### 🛠️ Project Architecture -
  
The system is built with a modern MLOps stack:

  * Model: LightGBM (Gradient Boosting) for high-performance tabular classification.

  * API: FastAPI for low-latency inference.

  * DevOps: Docker for environment parity.

  * Automation: GitHub Actions for Continuous Integration (CI) and Continuous Deployment (CD).

### 📈 The Development Journey -
1. **Data & EDA (01_eda.ipynb)**

   I analyzed over 390 features to understand the massive class imbalance (only ~3.5% of transactions are fraud).

   * Key Insight: Transaction amounts and card types showed significant distribution shifts between legitimate and fraudulent behavior.

   * Strategy: Focused on handling missing values and preparing for high-cardinality categorical encoding.



2. **Feature Engineering & Training**(`02_feature_engineering.ipynb, 03_model_training.ipynb`)

   * Schema Enforcement: Created a JSON-based schema to ensure the API receives data in the exact format the model expects.

   * Model Selection: Chose LightGBM due to its native support for missing values and categorical features, which are prevalent in the IEEE-CIS data.

   * Pruning: Optimized the model size to ensure fast inference and small Docker image footprints.



3. **Productionizing with FastAPI (src/api/)**

   I moved the logic from notebooks to structured Python scripts.

   * Logging: Implemented a custom logger to track API requests and errors.

   * Predictor Class: Built a dedicated FraudPredictor class to handle model loading and preprocessing in a thread-safe manner.

4. **Containerization & CI/CD (.github/workflows/)**

   This was the most challenging part of the build. I established a pipeline that:

   * Tests: Automatically runs pytest on every push to ensure the API logic is unbroken.

   * Verifies: Confirms that the model artifacts and schemas are present and loadable.

   * Builds: Packages the entire environment into a Docker Image.

   * Deploys: Automatically pushes the verified image to Docker Hub.

### ⚙️ How to Run -

You don't need to install Python or dependencies to test this. If you have Docker, run:


1. **Pull the image from Docker Hub**

`docker pull <imshubham18>/fraud-detection-api:latest`

2. **Run the container**

`docker run -p 8000:8000 <imshubham18>/fraud-detection-api:latest`

Then visit `http://localhost:8000/docs` to interact with the API via Swagger UI.

### 🛡️ Self-Correction & Learning (The "Engineering" Part) -

During development, I encountered several "production" hurdles that I had to solve:

  * The .gitignore Trap: Initially ignored the `artifacts/` folder, causing the CI pipeline to fail. I learned to manage model versions by including essential schema files while keeping logs ignored.

  * Environmental Paths: Solved `PYTHONPATH` and directory creation issues (`logs/`) within the GitHub Actions Linux runner to ensure the code works on any machine, not just my local laptop.

### 🔮 Next Steps: Phase 2 - 
Now that the baseline infrastructure is "Green," the next iteration will involve:

  * Identity Merging: Joining the train_identity and train_transaction files to capture device-level fraud patterns.

  * Model Drift Monitoring: Implementing checks to see if fraud patterns change over time.