import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
from sklearn.utils import shuffle
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly.express as px
import folium # Import folium

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
    api_key = st.secrets["google_api_key"]
    url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={api_key}"
    
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            result = r.json()
            if result['status'] == 'OK' and 'results' in result:
                return result['results'][0]['elevation']
            else:
                print(f"Google API error: {result.get('status')} - {result.get('error_message', 'No error message provided')}")
                st.error(f"Google Elevation API Error: {result.get('status')} - {result.get('error_message', 'Check console logs.')}")
        else:
            print(f"Google API request failed with status code {r.status_code} for {lat},{lon}")
            st.error(f"API Request Failed: Status {r.status_code}. Check console logs.")
    except requests.RequestException as e:
        print(f"Google API request failed for {lat},{lon}: {e}")
        st.error(f"Network Request Error: {e}")
    
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
    if len(lat_vals) * len(lon_vals) > 400*400: # Max grid size 400x400
        st.error(f"Grid size is too large ({len(lat_vals)}x{len(lon_vals)}). Please increase 'Resolution' or decrease 'Area'.")
        return None, None, None, None, None, None, None # Added None for df

    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals)
    all_coords = [(round(lat, 6), round(lon, 6)) for lat, lon in zip(lat_grid.ravel(), lon_grid.ravel())]

    # --- 2. Fetch Elevation Data for a Sample ---
    # Ensure sample_size doesn't exceed the total number of grid points
    actual_sample_size = min(sample_size, len(all_coords))
    sampled_coords = shuffle(all_coords, random_state=42)[:actual_sample_size]
    elevation_data = []

    st.info(f"Fetching {actual_sample_size} elevation points from the Google Elevation API...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (lat, lon) in enumerate(sampled_coords):
        elev = fetch_elevation(lat, lon)
        if elev is not None:
            elevation_data.append((lat, lon, elev))
        
        progress_bar.progress((i + 1) / actual_sample_size)
        status_text.text(f"Fetched point {i+1}/{actual_sample_size}. Successful points: {len(elevation_data)}")
        
        time.sleep(1) # Important for API rate limiting

    status_text.text(f"✅ Elevation fetch complete. Total points retrieved: {len(elevation_data)}")

    if len(elevation_data) < 10: # Minimum points required for interpolation
        st.error("Could not retrieve enough elevation points. The API might be down or rate-limited. Please try again later or with a smaller sample size, or check your Google API key/billing.")
        return None, None, None, None, None, None, None # Added None for df

    # --- 3. Two-Pass Interpolation for Best Quality and Robustness ---
    points = np.array([(lon, lat) for lat, lon, _ in elevation_data])
    values = np.array([elev for _, _, elev in elevation_data])
    interp_points = np.array([(lon, lat) for lat, lon in all_coords])

    # Try cubic, fallback to linear, then fill remaining NaNs with mean
    zi_cubic = griddata(points, values, interp_points, method='cubic')
    zi_linear = griddata(points, values, interp_points, method='linear')
    
    # Combine: prefer cubic, use linear where cubic fails
    zi = np.where(np.isnan(zi_cubic), zi_linear, zi_cubic)
    
    # Fill any remaining NaNs (e.g., outside convex hull of points) with the mean
    mean_val = np.nanmean(zi) if not np.all(np.isnan(zi)) else 0 # Handle case where all are NaN
    zi = np.nan_to_num(zi, nan=mean_val)
    
    zi = zi.reshape(lat_grid.shape)

    # --- 4. Compute Slope ---
    # Ensure spacing_m is not zero to prevent division by zero in gradient
    if spacing_m == 0:
        st.warning("Resolution (meters between points) cannot be zero. Slope calculation skipped.")
        slope = np.zeros_like(zi) # Default to no slope
    else:
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
    )
    ])
    fig.update_layout(
        title="3D Terrain Model",
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Elevation (m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.2) # Adjust Z aspect for better visualization of terrain
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

# --- NEW: Folium Map with Click Coordinates ---
def plot_interactive_map_with_coords(center_lat, center_lon, zoom_start=14):
    """
    Creates an interactive Folium map with a LatLngPopup to display coordinates on click.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        control_scale=True, # Shows a scale bar
        tiles='OpenStreetMap' # Default tiles, can be changed
    )

    # Add a click event to display Lat/Lon in a popup
    m.add_child(folium.LatLngPopup())

    # If you want to use Google Maps tiles with Folium, it's more involved:
    # Requires a separate Google Maps API key for specific tile layers (not the Embed API key directly)
    # folium.TileLayer(
    #     tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', # Satellite
    #     attr='Google Maps',
    #     name='Google Satellite',
    #     overlay=True,
    #     control=True
    # ).add_to(m)
    # folium.LayerControl().add_to(m)

    # Convert the Folium map to HTML
    map_html = m._repr_html_()
    return map_html


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

        # --- Display Folium Map with Clickable Coords ---
        st.subheader("Interactive Location Map (Click for Coordinates)")
        st.components.v1.html(
            plot_interactive_map_with_coords(center_lat, center_lon, zoom_start=14),
            height=450 # Needs to match the iframe height for proper rendering
        )
        st.info("Click anywhere on the map to see its Latitude and Longitude.")
        st.markdown("---") # Add a separator for visual clarity

        # --- Display Existing Plots ---
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_3d_terrain(lat_grid, lon_grid, elevation, slope), use_container_width=True)

        with col2:
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
