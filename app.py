import streamlit as st
import pandas as pd
import math, heapq, time
import folium
from streamlit_folium import st_folium

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on Earth using the Haversine formula"""
    earth_radius = 6371  # Earth's r (kms)
    
    # Convert latitude and longitude to radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * earth_radius * math.atan2(math.sqrt(a), math.sqrt(1-a))

@st.cache_resource
def load_data_and_build_graph():
    """Load city data and build a k-nearest neighbors graph for pathfinding"""
  
    cities_df = pd.read_csv("Data/Cities_of_SriLanka.csv")
    city_coords = {row['name_en']: (row['latitude'], row['longitude']) 
                 for _, row in cities_df.iterrows()}
    cities = list(city_coords.keys())
    
    # 10 neighbours graph
    nearest_neighbors = 10
    graph = {city: [] for city in cities}
    
    for city in cities:
        lat1, lon1 = city_coords[city]
        distances = [(haversine(lat1, lon1, city_coords[other][0], city_coords[other][1]), other)
                    for other in cities if other != city]
        
        # Connect each city to its nearest neighbors
        for distance, neighbor in sorted(distances)[:nearest_neighbors]:
            graph[city].append((neighbor, distance))
    
    return cities, city_coords, graph

cities, city_coords, graph = load_data_and_build_graph()

# -------------------------
# 3. Algos
# -------------------------
def dijkstra(start, end):
    pq = [(0, start, [])]
    visited = set()
    steps = 0  # Initialize step counter
    
    while pq:
        steps += 1  # Count each node processing as a step
        (dist, node, path) = heapq.heappop(pq)
        if node in visited:
            continue
        path = path + [node]
        if node == end:
            return dist, path, steps
        visited.add(node)
        for neighbor, d in graph[node]:
            if neighbor not in visited:
                heapq.heappush(pq, (dist + d, neighbor, path))
    return float("inf"), [], steps

def bellman_ford(start, end):
    dist = {c: float("inf") for c in cities}
    prev = {c: None for c in cities}
    dist[start] = 0
    steps = 0  # Initialize step counter
    
    # Keep track of nodes updated in the last iteration
    updated_nodes = {start}
    
    for _ in range(len(cities)-1):
        if not updated_nodes:  # Early termination if no updates
            break
            
        current_updates = set()
        # Only process edges from nodes that were updated
        for u in updated_nodes:
            for v, w in graph[u]:
                steps += 1  # Count each edge relaxation as a step
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    current_updates.add(v)
        
        updated_nodes = current_updates
    
    # More efficient path reconstruction
    if dist[end] == float("inf"):
        return float("inf"), [], steps
        
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    return dist[end], path[::-1], steps  # Reverse path at end

def a_star(start, end):
    def heuristic(city1, city2):
        lat1, lon1 = city_coords[city1]
        lat2, lon2 = city_coords[city2]
        return haversine(lat1, lon1, lat2, lon2)
    
    pq = [(0, start, [])]
    g = {c: float("inf") for c in cities}
    g[start] = 0
    steps = 0  # Initialize step counter
    
    while pq:
        steps += 1  # Count each node processing as a step
        f, node, path = heapq.heappop(pq)
        path = path + [node]
        if node == end:
            return g[node], path, steps
        for neighbor, d in graph[node]:
            tentative = g[node] + d
            if tentative < g[neighbor]:
                g[neighbor] = tentative
                f = tentative + heuristic(neighbor, end)
                heapq.heappush(pq, (f, neighbor, path))
    return float("inf"), [], steps


st.title("Sri Lankan Cities - Pathfinding Explorer")
st.markdown("By FC211004 - A.K.G.N. Deshapriya, FC211036 - S.M.A.S.A. Sewwandi, FC211007 - H.Y.M.T.P. Wickramasinghe")

st.markdown("""
### How to Use This App
1. Type & Select your starting city and destination city from the dropdown menus below
2. Choose which pathfinding algorithms you want to compare using the checkboxes
3. Click 'Find Shortest Paths' to calculate and visualize the routes
""")

# Set up the city selection interface
col1, col2 = st.columns(2)
with col1:
    source = st.selectbox("Starting City", cities, index=0)
with col2:
    destination = st.selectbox("Destination City", cities, index=1)

# Define available pathfinding algorithms
algorithms = {
    "Dijkstra": dijkstra,
    "Bellman-Ford": bellman_ford,
    "A* Search": a_star
}

selected_algos = []
algo_cols = st.columns(len(algorithms))
for i, algo_name in enumerate(algorithms.keys()):
    with algo_cols[i]:
        if st.checkbox(algo_name, value=True):
            selected_algos.append(algo_name)

# Ensure session state exists
if "results" not in st.session_state:
    st.session_state.results = None

# Custom CSS for the button
st.markdown("""
    <style>
        div.stButton > button:first-child {
            background-color: #ff4b4b;
            color: white;
            border: none;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color:#ab0b05 ;
            color: white;
            border: none;
        }
        div.stButton > button:focus {
            background-color: #ff4b4b;
            color: white;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

if st.button("Find Shortest Paths") and selected_algos:
    results = {}
    
    # Run selected algorithms
    for algo_name in selected_algos:
        start_time = time.time()
        dist, path, steps = algorithms[algo_name](source, destination)
        elapsed = time.time() - start_time
        results[algo_name] = {
            "distance": dist,
            "path": path,
            "time": elapsed,
            "steps": steps
        }
    st.session_state.results = results  # save results persistently

# Show results if available
if st.session_state.results:
    st.subheader("Results Comparison")
    
    # Create DataFrame for results
    results_df = pd.DataFrame([
        {
            "Algorithm": algo,
            "Path": " → ".join(res["path"]),
            "Distance (km)": round(res["distance"], 2),
            "Time (s)": round(res["time"], 6)
        }
        for algo, res in st.session_state.results.items()
    ])
    
    # Display results table
    st.dataframe(results_df, use_container_width=True)
    
    # Detailed Algorithm Performance Comparison
    st.subheader("Algorithm Performance Comparison")
    
    # Prepare data for the comparison table
    metrics_data = {}
    
    # Define metrics order and labels
    metrics_order = [
        ("Execution Time", "ms"),
        ("Total Distance", "km"),
        ("Path Length", "cities"),
        ("Intermediate Cities", "count"),
        ("Nodes Explored", "nodes"),
        ("Algorithm Steps", "steps")
    ]
    
    for algo_name, results in st.session_state.results.items():
        visited_count = len(set(results['path']))
        
        metrics_data[algo_name] = {
            "Execution Time": f"{results['time']*1000:.2f}",
            "Total Distance": f"{results['distance']:.2f}",
            "Path Length": str(len(results['path'])),
            "Intermediate Cities": str(len(results['path'])-2),
            "Nodes Explored": str(visited_count),
            "Algorithm Steps": str(results['steps'])
        }
    
    # Create DataFrame with metrics as rows and algorithms as columns
    comparison_df = pd.DataFrame(metrics_data)
    
    # Create index with metric names and units
    index_with_units = [f"{metric} ({unit})" if unit else metric 
                       for metric, unit in metrics_order]
    
    # Reorder rows according to metrics_order and rename index
    comparison_df = comparison_df.reindex([m[0] for m in metrics_order])
    comparison_df.index = index_with_units
    
    # Style the dataframe
    styled_df = comparison_df.style\
        .set_properties(**{
            'text-align': 'center',
            'font-size': '1em',
            'padding': '5px'
        })\
        .set_table_styles([
            {'selector': 'th', 'props': [
                ('text-align', 'left'),
                ('font-weight', 'bold'),
                ('padding', '5px 10px')
            ]},
            {'selector': 'td', 'props': [
                ('text-align', 'center'),
                ('padding', '5px 10px')
            ]},
            {'selector': 'tr:nth-child(odd)', 'props': [
                ('background-color', 'rgba(0,0,0,0.05)')
            ]}
        ])
    
    # Display the table
    st.dataframe(
        styled_df,
        use_container_width=True,
    )

    # Map Visualization (Dijkstra)
    # Map visualization with algorithm selection
    st.subheader("Path Visualization")
    
    # Let user select which algorithm's path to visualize
    if len(st.session_state.results) > 0:
        selected_algo = st.selectbox(
            "Select algorithm to visualize path:",
            list(st.session_state.results.keys())
        )
        
        selected_path = st.session_state.results[selected_algo]["path"]
        if selected_path:
            m = folium.Map(location=city_coords[source], zoom_start=7)
            coords = [(city_coords[c][0], city_coords[c][1]) for c in selected_path]
            
            # Add markers for source and destination with better tooltips
            folium.Marker(
                coords[0], 
                popup=f"<b>Source:</b> {source}",
                tooltip=f"Source: {source}",
                icon=folium.Icon(color="green")
            ).add_to(m)
            
            folium.Marker(
                coords[-1],
                popup=f"<b>Destination:</b> {destination}",
                tooltip=f"Destination: {destination}",
                icon=folium.Icon(color="red")
            ).add_to(m)
            
            # Add intermediate points with tooltips
            for i, coord in enumerate(coords[1:-1], 1):
                city_name = selected_path[i]
                folium.CircleMarker(
                    coord,
                    radius=5,
                    popup=f"<b>Stop {i}:</b> {city_name}",
                    tooltip=city_name,
                    color="blue",
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)
            
            # Create hover tooltips for path segments
            for i in range(len(coords) - 1):
                # Get city names for this segment
                city1 = selected_path[i]
                city2 = selected_path[i + 1]
                # Calculate segment distance
                segment_dist = haversine(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
                
                # Draw path segment with tooltip
                folium.PolyLine(
                    [coords[i], coords[i+1]],
                    color="blue",
                    weight=3,
                    opacity=0.8,
                    tooltip=f"{city1} → {city2} ({segment_dist:.1f} km)"
                ).add_to(m)
            
            # Add a summary popup for the entire path
            total_dist = st.session_state.results[selected_algo]['distance']
            path_summary = f"""
                <b>{selected_algo} Path Summary:</b><br>
                Total Distance: {total_dist:.2f} km<br>
                Cities: {' → '.join(selected_path)}
            """
            folium.Rectangle(
                bounds=[[min(c[0] for c in coords), min(c[1] for c in coords)], 
                       [max(c[0] for c in coords), max(c[1] for c in coords)]],
                popup=folium.Popup(path_summary, max_width=300),
                color="transparent",
                fill=False
            ).add_to(m)
            
            st_folium(m, width=700, height=500)
        else:
            st.error(f"No path found with {selected_algo}.")
