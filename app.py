import streamlit as st
import pandas as pd
import math, heapq, time
import folium
from streamlit_folium import st_folium

# -------------------------
# 1. Distance Calculation
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

# -------------------------
# 2. Cached Graph Building
# -------------------------
@st.cache_resource
def load_data_and_build_graph():
    cities_df = pd.read_csv("Data/Cities_of_SriLanka.csv")
    city_coords = {row['name_en']: (row['latitude'], row['longitude']) for _, row in cities_df.iterrows()}
    cities = list(city_coords.keys())
    
    K = 10
    graph = {city: [] for city in cities}
    for city in cities:
        lat1, lon1 = city_coords[city]
        distances = []
        for other in cities:
            if city == other: continue
            lat2, lon2 = city_coords[other]
            d = haversine(lat1, lon1, lat2, lon2)
            distances.append((d, other))
        distances.sort()
        for d, neighbor in distances[:K]:
            graph[city].append((neighbor, d))
    
    return cities, city_coords, graph

cities, city_coords, graph = load_data_and_build_graph()

# -------------------------
# 3. Algorithms
# -------------------------
def dijkstra(start, end):
    pq = [(0, start, [])]
    visited = set()
    while pq:
        (dist, node, path) = heapq.heappop(pq)
        if node in visited:
            continue
        path = path + [node]
        if node == end:
            return dist, path
        visited.add(node)
        for neighbor, d in graph[node]:
            if neighbor not in visited:
                heapq.heappush(pq, (dist + d, neighbor, path))
    return float("inf"), []

def bellman_ford(start, end):
    dist = {c: float("inf") for c in cities}
    prev = {c: None for c in cities}
    dist[start] = 0
    for _ in range(len(cities)-1):
        for u in graph:
            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
    path, node = [], end
    while node is not None:
        path.insert(0, node)
        node = prev[node]
    return dist[end], path

def a_star(start, end):
    def heuristic(city1, city2):
        lat1, lon1 = city_coords[city1]
        lat2, lon2 = city_coords[city2]
        return haversine(lat1, lon1, lat2, lon2)
    
    pq = [(0, start, [])]
    g = {c: float("inf") for c in cities}
    g[start] = 0
    
    while pq:
        f, node, path = heapq.heappop(pq)
        path = path + [node]
        if node == end:
            return g[node], path
        for neighbor, d in graph[node]:
            tentative = g[node] + d
            if tentative < g[neighbor]:
                g[neighbor] = tentative
                f = tentative + heuristic(neighbor, end)
                heapq.heappush(pq, (f, neighbor, path))
    return float("inf"), []

# -------------------------
# 4. Streamlit UI
# -------------------------
st.title("🇱🇰 Shortest Path Finder - Sri Lankan Cities")

source = st.selectbox("Select Source City", cities, index=0)
destination = st.selectbox("Select Destination City", cities, index=1)

# Ensure session state exists
if "results" not in st.session_state:
    st.session_state.results = None

if st.button("Find Shortest Paths"):
    results = {}
    for algo_name, algo_fn in {
        "Dijkstra": dijkstra,
        "Bellman-Ford": bellman_ford,
        "A* Search": a_star
    }.items():
        start_time = time.time()
        dist, path = algo_fn(source, destination)
        elapsed = time.time() - start_time
        results[algo_name] = {"distance": dist, "path": path, "time": elapsed}
    st.session_state.results = results  # save results persistently

# Show results if available
if st.session_state.results:
    st.subheader("Results Comparison")
    st.dataframe(pd.DataFrame([
        {"Algorithm": algo,
         "Path": " → ".join(res["path"]),
         "Distance (km)": f"{res['distance']:.2f}",
         "Time (s)": f"{res['time']:.4f}"}
        for algo, res in st.session_state.results.items()
    ]), use_container_width=True)

    # Map Visualization (Dijkstra)
    st.subheader("Path Visualization (Dijkstra)")
    d_path = st.session_state.results["Dijkstra"]["path"]
    if d_path:
        m = folium.Map(location=city_coords[source], zoom_start=7)
        coords = [(city_coords[c][0], city_coords[c][1]) for c in d_path]
        folium.Marker(coords[0], tooltip="Source").add_to(m)
        folium.Marker(coords[-1], tooltip="Destination").add_to(m)
        folium.PolyLine(coords, color="blue", weight=3).add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.error("No path found with Dijkstra.")
