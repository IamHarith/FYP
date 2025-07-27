# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
from keras.optimizers import Adam

st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Choose a Page', ['Introduction', 'Literature Review', 'Tools & Methodology', 'Model Performance Overview', 'Model', 'Conclusion & Future Work'])

if page == 'Introduction':
    st.title('PREDICTING OIL SPILL RISKS USING OCEANOGRAPHIC AND METEOROLOGICAL DATA')
    st.markdown("""
                A Machine Learning Approach to Proactive Environmental Protection 
                
                Muhammad Harith Aiman Bin Hairuzzaman

    """)
    st.markdown("### 🧠 About This Study")
    st.markdown("""
    This project, **Predicting Oil Spill Risks Using Oceanographic and Meteorological Data**, aims to enhance environmental protection by forecasting high-risk oil spill zones through machine learning.
    """)
    
    st.markdown("Oil Spills: A Major Environmental and Economic Threat")
    st.markdown("""
        - Ecological Damage: Devastates marine ecosystems, harms biodiversity, and causes long-term water pollution.
        - Economic Impact: Leads to costly cleanup efforts, disrupts fishing and tourism industries, and impacts coastal economies.
        - Reactive vs. Proactive: Current methods are often reactive, responding after a spill occurs. This is inefficient and too late
        """)
        
    st.markdown("It analyzes environmental factors—such as ocean currents, wind speed, wave height, and sea surface temperature—and historical spill incidents to train a predictive model. The outcome supports early warning systems and improves spill response strategies for regulatory bodies and maritime sectors.")
    
    st.markdown("### 🎯 Project Objectives")
    st.markdown("""
    - Analyze historical oil spill data in relation to oceanographic and meteorological conditions.
    - Develop a machine learning model to predict oil spill risks.
    - Evaluate the model’s accuracy and effectiveness.
    - Propose an early warning system for proactive response planning.
    - Enhance environmental policy and emergency response strategies.
    """)
    st.markdown("### 📍 Scope of the Study")
    st.markdown("""
    - Focuses on oil spills in **open ocean environments** (not inland).
    - Utilizes **public datasets** and real-time data via **Copernicus Marine API**.
    - Applies **machine learning** (e.g., One-Class SVM) for predictive modeling.
    - Uses non-imagery features like **uo/vo current data**, SST, wind speed, etc.
    """)

if page == 'Literature Review':
    st.title("📚 Literature Review")
    st.markdown("""
    Research in oil spill prediction has advanced from basic statistical analysis to complex computational models using artificial intelligence.
    Previous studies have utilized satellite imagery, hydrodynamic simulations, and meteorological forecasts for detecting and tracking spills.
    Machine learning models like neural networks and decision trees show potential in predicting spill trajectories and assessing risk factors.
    A significant gap remains in integrating real-time environmental data for precise risk forecasting.
    This study aims to bridge that gap by combining oceanographic and meteorological data within a unified predictive framework.
    """)
    st.markdown("""
    #### 🔍 Big Data Analytics in Oil Spill Prediction
    - Big data analytics is essential for creating predictive systems for oil spills, a field where it's gaining popularity.
    - The core characteristics of big data—volume, velocity, and variety—are highly relevant to oil spill prediction.
    - Accurate models depend on the ability to process vast amounts of data from diverse sources like satellite imagery, sensor networks, and historical records.
    - Recent progress in machine learning and deep learning has shown great potential in automating the detection, classification, and monitoring of oil spills using remote sensing data. (Yekeen & Balogun, 2020)
    - For instance, researchers have reviewed the use of these advanced techniques for oil spill detection and monitoring. (Al-Ruzouq et al., 2020; Yekeen & Balogun, 2020)

    #### 🤖 Oil Spill Detection Techniques
    - Remote Sensing (Optical and Thermal Imaging):
        - This is a primary method for identifying oil spills. Optical sensors on satellites and aircraft record reflectance differences between oil and water.
        - Multispectral bands from Sentinel-2 and MODIS have been used to distinguish oil slicks. However, their effectiveness is limited by low light and cloud cover. (Brekke & Solberg, 2005)
        - Thermal Infrared (TIR) sensors are effective at night because they detect temperature differences, as oil retains heat and appears warmer than water. (Fingas & Brown, 2014)
    - Automated Identification:
        - Modern techniques use Synthetic Aperture Radar (SAR) and hyperspectral imaging.
        - SAR is highly effective due to its all-weather capabilities. (Kubat, Holte, & Matwin, 1998)
        - Hyperspectral imaging improves detection accuracy by differentiating oil spills from natural phenomena. (Duan, Kang, & Ghamisi, 2022)

    #### 🌐  Machine Learning in Oil Spill Detection
    - Deep Learning:
        - Convolutional Neural Networks (CNNs) have shown superior performance in processing SAR and optical images for oil spill identification.
        - CNN-based models are better at distinguishing oil spills from similar-looking phenomena (e.g., biogenic slicks, low-wind areas) compared to traditional methods. (Zhang et al., 2021)
        - Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks have been used for time-series analysis of satellite images to track spill movements over time. (Li et al., 2022)
    - Other Machine Learning Algorithms:
        - Random Forest (RF) and Support Vector Machine (SVM) are also widely used. RF uses multiple decision trees for robust classification, while SVMs excel at binary classification tasks.
        - These models have been combined with SAR data to increase detection effectiveness and reduce false positives. (Brekke et al., 2016)
        - Real-time monitoring has been improved by integrating Geographic Information Systems (GIS), remote sensing, and AI-driven decision support systems that fuse data from multiple sources like UAVs and hyperspectral sensors. (Garcia-Pineda et al., 2019)

    ### 🌊 Environmental Influences
    - Ocean Currents:
        - The spatial distribution of oil spills is heavily influenced by ocean currents, which can spread oil rapidly and make identification difficult, especially in dynamic areas like the Gulf Stream. (Lehr et al., 2010)
        - Turbulence and eddies alter the shape and size of slicks, complicating tracking efforts.
        - Tidal currents near coastlines can mask oil signatures in satellite imagery due to complex interactions with shoreline features. (Reed et al., 1999)
    - Wind:
        - Wind direction and speed are major factors controlling the surface drift of oil slicks.
        - Wind-induced surface roughness can either enhance or obscure oil detection depending on the sensor type. For example, in low or high-wind conditions, oil slicks can blend with sea clutter, reducing the effectiveness of SAR sensors. (Elliott et al., 2007)
        - Wind stress can also create thin oil sheens that are difficult to distinguish from natural surface films.
    - Waves, Salinity, and Temperature:
        - Wave activity impacts the surface expression of oil. High-energy waves promote dispersion and emulsification, making detection by optical and infrared sensors more challenging. (Fingas & Fieldhouse, 2012)
        - Conversely, calm seas result in coherent slicks that are more easily identified by thermal imaging, which leverages the temperature difference between oil and water. (McNutt et al., 2012)
        - Sea surface temperature and salinity affect the optical properties of oil slicks. Warmer waters accelerate oil evaporation, altering its spectral signature. (Spaulding, 2017)
        - Lower salinity can increase oil dispersion, leading to less contrast in satellite measurements.
    """)
    
    with st.expander("🔖 Key References"):
        st.markdown("""
        - **Yekeen & Balogun (2020)** – Deep learning for marine oil spill detection.
        - **Al-Ruzouq et al. (2020)** – ML and remote sensing in oil spill monitoring.
        - **Zhang et al. (2021)** – CNNs for SAR-based detection.
        - **McNutt et al. (2012)** – Deepwater Horizon: temperature effects on oil behavior.
        - **Duan et al. (2022)** – Hyperspectral oil-water separation.
        """)

if page == 'Tools & Methodology':
    st.title('Tools & Methodology')
    st.markdown("🔁 CRISP-DM Methodology")
    st.markdown("This project follows the **CRISP-DM framework**, which includes:")
    st.markdown("Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment.")

    with st.expander("1. Business Understanding – Identify key risk factors."):
        st.markdown("""
        - **Problem**: Oil spills cause major environmental and economic damage.
        - **Goal**: Predict high-risk zones using oceanographic and meteorological data.
        - **Impact**: Helps improve early warning systems and response planning.
        """)

    with st.expander("2. Data Understanding – Analyze spill and ocean datasets."):
        st.markdown("""
        - **Sources**:
            - Historical spill data (`.csv`, 4,662 incidents)
            - Oceanographic `.nc` datasets from the **Copernicus Marine API**
        - **Key Features**:
            - `uo`, `vo` (ocean currents)
            - Wind speed, sea surface temperature, wave height, etc.
        - **Challenges**:
            - Missing values
            - Mixed formats (`csv`, `nc`)
            - Need for merging temporal and spatial data
        """)

    with st.expander("3. Data Preparation – Clean, merge, and standardize features."):
        st.markdown("""
        - **Steps**:
            - Handled missing and noisy values
            - Scaled numerical features (StandardScaler)
            - Extracted `uo`, `vo` from `.nc` files using `xarray`
            - Combined spill and non-spill data into one labeled dataset
        - **Goal**: Create a clean dataset for binary classification (spill = 1, normal = 0)
        """)

    with st.expander("4. Modeling – Train machine learning models."):
        st.markdown("**Model Used**: Local Outlier Factor (LOF)")
        st.markdown("**Why Local Outlier Factor (LOF)**:")
        st.markdown("""
            -In real-world oceanographic or meteorological datasets, oil spills occur infrequently compared to normal conditions.
            -LOF is an unsupervised anomaly detection algorithm that identifies data points that deviate significantly from their neighbors, making it suitable for spotting these rare spill events.
            -**Other model used** : One class SVM, Isolation Forest, Autoencoder
        """)

    with st.expander("5. Evaluation – Assess model performance."):
        st.markdown("""
        - **Metrics**:
            - Classification Report (Precision, Recall, F1-score)
            - Confusion Matrix
            - Accuracy Score
        - **Findings**:
            - Good early results detecting spill anomalies
            - Visualization confirms spatial clustering of high-risk points
        """)

    with st.expander("6. Deployment – Visualize predictions and insights in this app."):
        st.markdown("""
        - **This Streamlit App** acts as the model's deployment interface
        - Users can:
            - View predictions
            - Interpret model outputs through graphs
        """)

    st.markdown("### 📊 Data Sources & Tools")
    st.markdown("""
    - **Historical CSV** oil spill data (4,662 incidents).
    - **Copernicus Marine API** for oceanographic data (`.nc` files).
    - Tools used:
      - **Excel** – Manual inspection, trend detection.
      - **Jupyter Notebook** – Data preprocessing, training, evaluation.
      - **Kaggle** – Idea generation and modeling approaches.
    """)
    st.image(['https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRt__gdZwhO3aSPCNy6b8HwnR5E5AARVCA1wQ&s', 'https://i0.wp.com/altc.alt.ac.uk/blog/wp-content/uploads/sites/1112/2019/05/250px-Jupyter_logo.png?fit=250%2C290&ssl=1', 'https://miro.medium.com/v2/resize:fit:1200/1*JSbnt_mxpFfkGtNtGbR40g.png'], caption=['Microsoft Excel', 'Jupyter Notebook', 'Kaggle'], width=300)
    st.markdown("### 🌊 Key Environmental Factors Affecting Spills")
    st.markdown("""
    - **Ocean Currents**: Strong currents disperse spills quickly.
    - **Wind**: Influences surface oil movement; complicates satellite detection.
    - **Waves**: Affect oil shape and visibility.
    - **Salinity & SST**: Impact oil evaporation and mixing.
    """)
    
    st.markdown("### Anomaly Detection: The Why and How")
    st.markdown("""
    - Why Anomaly Detection?
        - Spill events are rare compared to normal ocean conditions.
        - It's easier to define what's "normal" and flag anything that deviates.
    - How it Works:
        - Train the model only on normal data (non-spill ocean currents).
        - The model learns the signature of a "normal" ocean state.
        - When new data is introduced, the model flags any data point that doesn't fit this normal pattern as a potential spill (an anomaly).
    """)
    
if page == 'Model Performance Overview':   
    st.title("Comparing the Model")
    st.image("media/compare.png")
    st.markdown("""
    - Key Takeaway:
        - LOF performed best, achieving the highest accuracy and, most importantly, the highest recall.
        - Recall is critical: It measures how many actual spills the model correctly identified. A higher recall means fewer missed disasters.
    """)


if page == 'Model':
    st.title("🌊 Oil Spill Detection Model")
    st.markdown("anomaly detection models on `uo` and `vo` ocean current features.")

    # === Load Spill CSV ===
    st.subheader("📂 Load Spill Data")
    spill_csv_path = st.text_input("Path to Aligned Spill CSV", "aligned_data.csv")
    try:
        df_spill = pd.read_csv(spill_csv_path)[["uo", "vo"]].dropna()
        df_spill["label"] = 1
        st.success(f"Loaded {len(df_spill)} spill samples.")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        df_spill = pd.DataFrame()

    # === Load Non-Spill .nc Data ===
    st.subheader("📁 Load Non-Spill .nc Files")
    non_spill_folder = st.text_input("Path to Non-Spill Folder", "C:/Users/harit/Downloads/fyp/dataset/non_spill_uo_vo")

    @st.cache_data
    def extract_uv_features(path):
        try:
            ds = xr.open_dataset(path)
            uo = ds.get("uo")
            vo = ds.get("vo")
            if uo is None or vo is None:
                return None
            uo_mean = np.nanmean(uo.values)
            vo_mean = np.nanmean(vo.values)
            return [uo_mean, vo_mean]
        except:
            return None

    @st.cache_data
    def load_non_spill_data(folder):
        records = []
        for f in os.listdir(folder):
            if f.endswith(".nc"):
                path = os.path.join(folder, f)
                uv = extract_uv_features(path)
                if uv and not np.isnan(uv).any():
                    records.append(uv + [0])
        return pd.DataFrame(records, columns=["uo", "vo", "label"])

    if os.path.isdir(non_spill_folder):
        df_nonspill = load_non_spill_data(non_spill_folder)
        st.success(f"Loaded {len(df_nonspill)} non-spill samples.")
    else:
        df_nonspill = pd.DataFrame()

    # === Combine & Preprocess ===
    if not df_spill.empty and not df_nonspill.empty:
        df_all = pd.concat([df_spill, df_nonspill], ignore_index=True).sample(frac=1).reset_index(drop=True)

        df_all["speed"] = np.sqrt(df_all["uo"]**2 + df_all["vo"]**2)
        df_all["direction"] = np.arctan2(df_all["vo"], df_all["uo"])
        features = ["uo", "vo", "speed", "direction"]
        X = df_all[features].values
        y_true = df_all["label"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train = X_scaled[y_true == 0]  # train only on non-spill

        st.subheader("⚙️ Training Models")

        def train_lof():
            model = LocalOutlierFactor(n_neighbors=20, contamination=0.2, novelty=True)
            model.fit(X_train)
            scores = model.decision_function(X_scaled)
            preds = model.predict(X_scaled)
            preds_bin = (preds == -1).astype(int)
            return "Local Outlier Factor", preds_bin, scores

        models = [train_lof]
        # Store predictions and scores for ensemble
        model_outputs = []

        for train_func in models:
            name, y_pred_bin, score = train_func()
            ...
            # Add this line at the end of each loop
            model_outputs.append((name, y_pred_bin, score))

        for train_func in models:
            name, y_pred_bin, score = train_func()
            st.subheader(f"📊 {name}")
            acc = np.mean(y_pred_bin == y_true)
            st.metric("Accuracy", f"{acc:.2%}")
            st.text(classification_report(y_true, y_pred_bin))

            fig, ax = plt.subplots()
            cm = confusion_matrix(y_true, y_pred_bin)
            ConfusionMatrixDisplay(cm, display_labels=["Normal", "Spill"]).plot(ax=ax)
            ax.set_title(f"{name} - Confusion Matrix")
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_pred_bin, palette={0: "blue", 1: "red"}, ax=ax2)
            ax2.set_title(f"{name} - Prediction (Blue = Normal, Red = Spill)")
            ax2.set_xlabel("uo")
            ax2.set_ylabel("vo")
            st.pyplot(fig2)

            fig3, ax3 = plt.subplots()
            sc = ax3.scatter(X[:, 0], X[:, 1], c=score, cmap='viridis', s=50)
            plt.colorbar(sc, ax=ax3, label="Score")
            ax3.set_title(f"{name} - Confidence Heatmap")
            ax3.set_xlabel("uo")
            ax3.set_ylabel("vo")
            st.pyplot(fig3)
            
            
        # After the model loop, collect results
        results = []

        for train_func in models:
            name, y_pred_bin, score = train_func()
            acc = np.mean(y_pred_bin == y_true)
            cm = confusion_matrix(y_true, y_pred_bin)
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall
            })

        # Create DataFrame
        df_results = pd.DataFrame(results)
        st.subheader("📈 Model Comparison Table")
        st.dataframe(df_results.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}"
        }))

    else:
        st.warning("Please ensure both spill CSV and non-spill folder are valid and contain data.")

    st.markdown("---")
    st.caption("FYP Project – Oil Spill Prediction using One-Class SVM")

if page == 'Key Challenges & Limitations':  
    st.title("What Makes This Task So Difficult?")
    st.markdown("""
    Key Points:
        Low Recall: All models struggled to identify every spill, indicating that other factors are at play.
        Feature Space Complexity: Ocean dynamics are naturally chaotic. The boundary between a "normal" current and an "anomalous" one is blurry.
        Data Limitations: The model only used uo and vo currents. It lacked other crucial data like wind, waves, and sea temperature, which also influence spills.
    """)

if page == 'Conclusion & Future Work':
    st.title("Foundational Steps Towards a Proactive System")
    st.markdown("""
    - Summary of Achievements:
        - Established a Baseline: Proved that an anomaly-based prediction system for oil spills is feasible.
        - Identified Best Approach: Density-based models like LOF are the most promising for this type of analysis.
        - Highlighted Data Needs: Showed that while ocean currents are important, more environmental features are required for operational accuracy.
    """)
    
    st.markdown("""
    - The Path to a More Robust Model
        - Advanced Feature Engineering: Integrate more data—wind, waves, temperature, salinity, and even atmospheric pressure.
        - Explore Ensemble & Hybrid Models: Combine the strengths of multiple models to improve accuracy and reduce false alarms.
        - Incorporate Real-Time Data & XAI: Build a system that can analyze live data streams and use Explainable AI to tell operators why an alert was triggered.
    """)