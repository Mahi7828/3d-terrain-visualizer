import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
from sklearn.utils import shuffle
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly.express as px

# --- Configuration and Page Setup ---
st.set_page_config(
    page_title="3D Terrain Visualizer",
    page_icon="🗺️",
    layout="wide"
)

# --- Helper Functions ---

# Cache the API fetching function to avoid re-calling on every script run
@st.cache_data(show_spinner=False)
def fetch_elevation(lat, lon):
    """Fetches elevation using Google Elevation API."""
    api_key = st.secrets["google_api_key"]    # 🔐 Replace with your actual key
    url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={api_key}"
    
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            result = r.json()
            if result['status'] == 'OK' and 'results' in result:
                return result['results'][0]['elevation']
            else:
                print(f"Google API error: {result.get('status')}")
    except requests.RequestException as e:
        print(f"Google API request failed for {lat},{lon}: {e}")
    
    return None


# Cache the main data generation and processing function
@st.cache_data(show_spinner="Generating terrain data... This may take a few minutes.")
def generate_terrain(center_lat, center_lon, area_acres, spacing_m, sample_size):
    """
    Generates a terrain grid by fetching elevation data, interpolating it,
    and calculating the slope. Uses a two-pass interpolation for best results.
    """
    # --- 1. Define Grid from Parameters ---
    area_m2 = area_acres * 4046.86
    side_m = np.sqrt(area_m2)

    lat_step = spacing_m / 111000
    lon_step = spacing_m / (111000 * np.cos(np.radians(center_lat)))

    lat_vals = np.arange(center_lat - side_m/2/111000,
                         center_lat + side_m/2/111000, lat_step)
    lon_vals = np.arange(center_lon - side_m/2/(111000*np.cos(np.radians(center_lat))),
                         center_lon + side_m/2/(111000*np.cos(np.radians(center_lat))), lon_step)

    # Check if grid is too large to prevent memory errors
    if len(lat_vals) * len(lon_vals) > 400*400:
        st.error(f"Grid size is too large ({len(lat_vals)}x{len(lon_vals)}). Please increase 'Resolution' or decrease 'Area'.")
        return None, None, None, None, None, None

    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals)
    all_coords = [(round(lat, 6), round(lon, 6)) for lat, lon in zip(lat_grid.ravel(), lon_grid.ravel())]

    # --- 2. Fetch Elevation Data for a Sample ---
    sampled_coords = shuffle(all_coords, random_state=42)[:sample_size]
    elevation_data = []

    st.info(f"Fetching {sample_size} elevation points from the Open-Elevation API...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (lat, lon) in enumerate(sampled_coords):
        elev = fetch_elevation(lat, lon)
        if elev is not None:
            elevation_data.append((lat, lon, elev))
        
        progress_bar.progress((i + 1) / sample_size)
        status_text.text(f"Fetched point {i+1}/{sample_size}. Successful points: {len(elevation_data)}")
        
        time.sleep(1)

    status_text.text(f"✅ Elevation fetch complete. Total points retrieved: {len(elevation_data)}")

    if len(elevation_data) < 10:
        st.error("Could not retrieve enough elevation points. The API might be down or rate-limited. Please try again later or with a smaller sample size.")
        return None, None, None, None, None, None

    # --- 3. Two-Pass Interpolation for Best Quality and Robustness ---
    points = np.array([(lon, lat) for lat, lon, _ in elevation_data])
    values = np.array([elev for _, _, elev in elevation_data])
    interp_points = np.array([(lon, lat) for lat, lon in all_coords])

    zi_cubic = griddata(points, values, interp_points, method='cubic')
    zi_linear = griddata(points, values, interp_points, method='linear')
    zi = np.where(np.isnan(zi_cubic), zi_linear, zi_cubic)
    
    mean_val = np.nanmean(zi)
    zi = np.nan_to_num(zi, nan=mean_val)
    
    zi = zi.reshape(lat_grid.shape)

    # --- 4. Compute Slope ---
    dy, dx = np.gradient(zi, spacing_m, spacing_m)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    df = pd.DataFrame(elevation_data, columns=["Latitude", "Longitude", "Elevation"])

    # Return the 1D axis arrays along with the grids
    return lat_grid, lon_grid, lat_vals, lon_vals, zi, slope, df


# --- Plotting Functions ---

def plot_3d_terrain(lat_grid, lon_grid, elevation, slope):
    """Creates an interactive 3D surface plot."""
    fig = go.Figure(data=[go.Surface(
        x=lon_grid, y=lat_grid, z=elevation,
        surfacecolor=slope,
        colorscale='viridis',
        cmin=np.nanmin(slope), cmax=np.nanmax(slope),
        colorbar=dict(title='Slope (Degrees)'),
        hovertemplate="Lat: %{y:.5f}<br>Lon: %{x:.5f}<br>Elev: %{z:.2f} m<br>Slope: %{surfacecolor:.2f}°"
    )])
    fig.update_layout(
        title="3D Terrain Model",
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Elevation (m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.2)
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def plot_heatmap(grid, title, colorbar_title):
    """Creates an interactive heatmap."""
    fig = px.imshow(grid,
                    labels=dict(x="Longitude Index", y="Latitude Index", color=colorbar_title),
                    title=title,
                    color_continuous_scale='viridis',
                    aspect='auto'
                   )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig

def plot_contour(lon_vals, lat_vals, elevation):
    """Creates an interactive contour plot using 1D axis arrays."""
    fig = go.Figure(data=go.Contour(
        z=elevation,
        x=lon_vals, # Use 1D longitude array
        y=lat_vals, # Use 1D latitude array
        colorscale='viridis',
        contours=dict(
            coloring='lines',
            showlabels=True,
        ),
        colorbar=dict(title='Elevation (m)')
    ))
    fig.update_layout(
        title="Elevation Contour Lines",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        margin=dict(l=0, r=0, t=40, b=0),
        # Ensure aspect ratio is natural for geo plots
        yaxis = dict(scaleanchor = "x")
    )
    return fig


# --- Streamlit App UI ---

st.title("🗺️ Interactive Terrain Visualizer")
st.markdown("Analyze a piece of land by fetching real-world elevation data. Adjust parameters in the sidebar and click 'Generate Analysis'.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Land Parameters")
    center_lat = st.number_input("Center Latitude", value=18.7885, format="%.6f")
    center_lon = st.number_input("Center Longitude", value=73.3395, format="%.6f")
    area_acres = st.slider("Area (Acres)", 1, 100, 50)
    
    st.header("Technical Parameters")
    spacing_m = st.slider("Resolution (meters between points)", 5, 50, 10, 1)
    sample_size = st.slider("API Sample Points", 100, 1000, 250, 50)
    
    generate_button = st.button("🚀 Generate Analysis", use_container_width=True)


# --- Main Content Area ---
if generate_button:
    # Unpack the new 1D arrays
    lat_grid, lon_grid, lat_vals, lon_vals, elevation, slope, df = generate_terrain(
        center_lat, center_lon, area_acres, spacing_m, sample_size
    )

    if elevation is not None and not np.all(np.isnan(elevation)):
        st.success("Analysis Complete! View the results below.")

        # --- Display Plots ---
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_3d_terrain(lat_grid, lon_grid, elevation, slope), use_container_width=True)

        with col2:
            # Pass the 1D arrays to the contour plot
            st.plotly_chart(plot_contour(lon_vals, lat_vals, elevation), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
             st.plotly_chart(plot_heatmap(elevation, "Elevation Heatmap", "Elevation (m)"), use_container_width=True)
        with col4:
             st.plotly_chart(plot_heatmap(slope, "Slope Heatmap", "Slope (Degrees)"), use_container_width=True)

        # --- Data Preview ---
        with st.expander("View Raw Sampled Data"):
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                 label="📥 Download Data as CSV",
                 data=csv,
                 file_name=f'elevation_data_{center_lat}_{center_lon}.csv',
                 mime='text/csv',
             )

    else:
        st.warning("Could not generate the terrain. Please check the error messages above.")

else:
    st.info("Set your parameters in the sidebar and click the 'Generate Analysis' button to begin.")
