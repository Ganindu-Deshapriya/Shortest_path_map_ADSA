# 🗺️ Shortest Path Finding in Sri Lankan Cities
## 📚 Advanced Data Structures and Algorithms Project Report

### 👥 Team Members
- FC211004 - A.K.G.N. Deshapriya
- FC211036 - S.M.A.S.A. Sewwandi
- FC211007 - H.Y.M.T.P. Wickramasinghe

## 1. 🎯 Project Overview

### 1.1 📋 Introduction
This project implements and compares three fundamental pathfinding algorithms (Dijkstra's, Bellman-Ford, and A*) to find the shortest path between cities in Sri Lanka. The implementation includes an interactive web application that visualizes the paths and provides detailed performance metrics for algorithm comparison.

### 1.2 🐍 Python Libraries and Methods
1. **Core Libraries**
   - `streamlit`: Web application framework for creating interactive dashboard.
   - `pandas`: Data manipulation and analysis.
   - `folium`: Interactive map visualization

2. **Custom Methods**
   - `haversine()`: Calculate great-circle distances


### 1.3 Objectives
- Implement and compare different pathfinding algorithms
- Create an interactive visualization of paths between Sri Lankan cities
- Analyze and compare algorithm performance metrics
- Provide a user-friendly interface for route finding
- Demonstrate practical applications of graph algorithms

## 2. 💻 Technical Implementation

### 2.1 🔧 Data Structures
1. **Graph Representation**
   - Adjacency list implementation using Python dictionaries
   - K-nearest neighbors (K=10) approach for graph construction
   - City coordinates stored as (latitude, longitude) pairs

2. **Priority Queues**
   - Used in Dijkstra's and A* algorithms
   - Implemented using Python's heapq module
   - Stores (distance, node, path) tuples

### 2.2 🔍 Algorithms Implemented

#### 2.2.1 🚀 Dijkstra's Algorithm
- **Implementation**: Priority queue-based implementation
- **Time Complexity**: O((V + E) log V)
- **Space Complexity**: O(V)
- **Key Features**:
  - Maintains a priority queue of unvisited nodes
  - Uses visited set to prevent cycles
  - Guarantees shortest path in non-negative weighted graphs

#### 2.2.2 ⚡ Bellman-Ford Algorithm
- **Implementation**: Optimized version with early termination
- **Time Complexity**: O(VE)
- **Space Complexity**: O(V)
- **Key Features**:
  - Tracks updated nodes for optimization
  - Early termination when no updates occur
  - Can handle negative edge weights (not applicable in this case)

#### 2.2.3 🎯 A* Algorithm
- **Implementation**: Uses haversine distance as heuristic
- **Time Complexity**: O((V + E) log V)
- **Space Complexity**: O(V)
- **Key Features**:
  - Employs haversine distance heuristic
  - Maintains g-scores and f-scores
  - More efficient than Dijkstra's for targeted search

### 2.3 Distance Calculation
- Implemented using Haversine formula
- Accounts for Earth's spherical shape
- Provides accurate distances between geographical coordinates

## 3. 🖥️ User Interface

### 3.1 ⭐ Features
1. **City Selection**
   - Dropdown menus for source and destination cities
   - Algorithm selection checkboxes
   - Interactive map visualization

2. **Performance Metrics Display**
   - Execution time
   - Total distance
   - Path length
   - Intermediate cities
   - Nodes explored
   - Algorithm steps

3. **Interactive Map**
   - Color-coded markers (green: start, red: end, blue: intermediate)
   - Hover tooltips with city names and distances
   - Path segments with distance information
   - Comprehensive path summary

### 3.2 🛠️ Technology Stack
- Python 3.10
- Streamlit for web interface
- Folium for map visualization
- Pandas for data handling
- Streamlit-folium for map integration

## 4. 📊 Performance Analysis

### 4.1 📈 Algorithm Comparison
1. **Execution Time**
   - A* typically performs fastest for long distances
   - Dijkstra's efficient for shorter paths
   - Bellman-Ford shows improved performance with optimizations

2. **Memory Usage**
   - All algorithms maintain minimal memory footprint
   - Efficient path reconstruction implementation
   - Optimized data structures for city storage

3. **Step Count Analysis**
   - Provides insight into actual algorithm operations
   - Helps understand algorithmic efficiency
   - Validates theoretical complexity analysis

### 4.2 Optimization Techniques
1. **Graph Construction**
   - K-nearest neighbors approach reduces edge count
   - Caching of distance calculations
   - Efficient coordinate lookup system

2. **Algorithm Optimizations**
   - Early termination in Bellman-Ford
   - Efficient path reconstruction
   - Optimized priority queue operations

## 5. 🎯 Conclusions

### 5.1 💡 Key Findings
- A* algorithm generally performs best for geographical pathfinding
- K-nearest neighbors approach effectively reduces graph complexity
- Interactive visualization enhances user understanding
- Performance metrics provide valuable algorithm comparison insights

### 5.2 🚀 Future Improvements
1. **Potential Enhancements**
   - Additional pathfinding algorithms (e.g., Bidirectional search)
   - Dynamic graph updates for real-time traffic conditions
   - Multiple waypoints support
   - Alternative route suggestions

2. **Technical Improvements**
   - Parallel processing for larger datasets
   - Caching mechanism for frequent routes
   - Advanced heuristics for A* algorithm
   - Mobile-friendly interface optimization

## 6. 📚 References

1. Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. Numerische Mathematik, 1(1), 269-271.
2. Bellman, R. (1958). On a routing problem. Quarterly of Applied Mathematics, 16(1), 87-90.
3. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. IEEE Transactions on Systems Science and Cybernetics, 4(2), 100-107.
4. Streamlit Documentation: https://docs.streamlit.io/
5. Folium Documentation: https://python-visualization.github.io/folium/

## 📎 Appendix

### A. 📂 Code Structure
```
project/
├── app.py              # Main application file
├── Data/               # Data directory
│   └── Cities_of_SriLanka.csv
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

### B. ⚙️ Installation and Setup
```bash
# Clone repository
git clone https://github.com/username/shortest-path-map

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```