# Shortest_path_map_ADSA# 🇱🇰 Shortest Path Finder for Sri Lankan Cities

This project implements a **Shortest Path Analysis Tool** for Sri Lankan cities using multiple graph algorithms.  
It allows users to compare the performance of **Dijkstra’s Algorithm, Bellman-Ford Algorithm, and A* Search Algorithm** in finding the shortest path between two cities.  

The project includes:
- **Final.ipynb** – Jupyter Notebook containing the full algorithmic analysis, testing, and experimentation.  
- **app.py** – A **Streamlit web application** that provides an interactive user interface to explore shortest paths on a map.  

---

## 📂 Project Structure

```
├── Data/
│   └── Cities_of_SriLanka.csv    # Dataset containing cities and their coordinates
├── Final.ipynb                   # Jupyter Notebook for algorithm analysis
├── app.py                        # Streamlit Web App
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

## ⚙️ Features

- **Graph Construction**
  - Uses real latitude and longitude of Sri Lankan cities.
  - Each city is connected to its *K* nearest neighbors (default: 10).
  - Distances are computed using the **Haversine formula** (great-circle distance).  

- **Algorithms Implemented**
  - **Dijkstra’s Algorithm** – single-source shortest path.
  - **Bellman-Ford Algorithm** – supports negative weights (for completeness, though not needed here).
  - **A* Search Algorithm** – heuristic-based shortest path using Haversine as heuristic.  

- **Web Application (Streamlit)**
  - Select **source** and **destination** cities.  
  - Compute shortest paths with **all algorithms at once**.  
  - Compare results in a **table** (distance, path, time taken).  
  - Visualize the path on an **interactive Folium map**.  

- **Notebook (Final.ipynb)**
  - Step-by-step explanation of graph construction.  
  - Implementation and testing of algorithms.  
  - Performance comparison with different city pairs.  
  - Useful for academic submission and experimentation.  

---

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/sri-lanka-shortest-path.git
cd sri-lanka-shortest-path
```

### 2. Install Requirements

1. Open in VsCode and Reopne with the Devcontainer. everything will be installed
2. Get the token from the terminal and gain access to the jupyter server

# Jupyter Server Token Guide

When you start a Jupyter server (Notebook or Lab), it asks for a **token** for authentication.  
This guide shows you the quickest ways to get it.

---

## 🔹 2.1. Get the Token from the Terminal
When you run:
```bash
jupyter notebook

```
or 
```bash
jupyter lab
```

you'll see and output like:

```bash
[I 10:23:45 NotebookApp] The Jupyter Notebook is running at:
http://localhost:8888/?token=abcd1234efgh5678

```
3. enter your own login PW for the server


### 3. Run Jupyter Notebook
```bash
jupyter notebook Final.ipynb
```
This lets you explore the algorithms in detail and run test cases.

### 4. Run Web App
```bash
streamlit run app.py
```
This will open the app in your browser (default: [http://localhost:8501](http://localhost:8501)).

---

## 📊 Example Output

- **Results Table**

| Algorithm     | Path (Cities)        | Distance (km) | Time (s) |
|---------------|----------------------|---------------|----------|
| Dijkstra      | Colombo → Kandy → … | 120.5         | 0.0723   |
| Bellman-Ford  | Colombo → Kandy → … | 120.5         | 0.0965   |
| A* Search     | Colombo → Kandy → … | 120.5         | 0.0018   |

- **Map Visualization**

An interactive Folium map shows the path from **source** to **destination**.

---

## 📘 Notes

- The dataset (`Cities_of_SriLanka.csv`) contains city names and coordinates.  
- The notebook (`Final.ipynb`) is suitable for academic evaluation and detailed documentation.  
- The Streamlit app (`app.py`) is optimized for **user interaction** and **visualization**.  

---

## 🧑‍💻 Authors
Developed as part of a university assignment on **Graph Algorithms and Shortest Path Analysis**.
By Faculty of Computing, University of Sri Jayewardenepura.   
