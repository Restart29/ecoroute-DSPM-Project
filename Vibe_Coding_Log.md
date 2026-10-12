# Vibe Coding & AI Orchestration Log: EcoRoute AI

**Team:** 8–9 Member Capstone Cohort  
**Domain:** Supply Chain Scope 3 Decarbonization  
**Platforms Orchestrated:** Google AI Studio (Gemini), VS Code, Python 3.11  

### 1. Prompt Engineering Strategy & Iterative Architecture
* **System Persona & Prompt Chaining:** We prompted LLM agents with explicit system constraints ("Senior Logistics ML Architect"). Workflows were decomposed into isolated chains: (1) Spatial feature engineering via Haversine geometry, (2) Dual-model benchmarking pipelines, and (3) Reactive Streamlit UI development.
* **Zero-Shot Scaffolding to Few-Shot Refinement:** High-level boilerplate (Streamlit page config, sidebar layouts) was drafted using zero-shot generation. Domain-specific constraints (GLEC v3.0 default emission factors, ISO 14083 activity equations, SHAP waterfall array alignment) were refined using few-shot context injection with exact mathematical targets.

### 2. Human-in-the-Loop Quality Assurance & Code Audits
* **Geospatial & Road Circuity Verification:** AI-suggested Euclidean distance was flagged during testing as inadequate for surface freight. The team audited and injected a Great-Circle Haversine implementation with a 1.25 Brazilian highway circuity multiplier to match real-world transit topography.
* **SHAP Feature Alignment Audit:** An initial runtime exception (`IndexError: index 123 is out of bounds for axis 0`) occurred when feeding raw inputs to the explainer. The engineering team audited the tree explainer pipeline, discovering that `OneHotEncoder` expanded the 9 input dimensions into 120+ sparse feature vectors. We manually corrected the script to pass `preprocessor.get_feature_names_out()` dynamically to `shap.Explanation`.
* **Standard Verification:** AI output initially mixed US EPA ton-mile factors with European GLEC factors. Human oversight normalized all calculations to international ISO 14083 / GLEC Tier-3 metrics (0.102 kg CO2e / tonne-km for diesel road freight; 0.030 for rail; 0.015 for cabotage maritime).

### 3. Engineering Velocity & Productivity Multipliers
* **Development Acceleration:** Full delivery of data cleaning, multi-modal feature calculations, dual-model training, K-Means clustering, and a 5-tab executive dashboard was achieved rapidly.
* **Productivity Multiplier:** We estimate an **8.5x velocity multiplier** relative to manual coding. AI handled repetitive syntax, matrix reshaping, Plotly CSS injection, and cross-validation scaffolding, allowing the engineering team to focus entirely on carbon accounting logic, model convergence, and business trade-off modeling.
