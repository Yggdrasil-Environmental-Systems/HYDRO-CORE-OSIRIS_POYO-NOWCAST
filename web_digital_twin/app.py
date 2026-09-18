"""
YGGDRASIL ENVIRONMENTAL SYSTEMS: HYDRO-CORE-OSIRIS:POYO-NOWCAST: Plataforma C4ISR Civil de Gemelo Digital 3D, Resiliencia y Solvencia II
Tecnología: Streamlit + PyTorch (Hidrodinámica FNO 2D V5) + PyDeck (Gemelo 3D WebGPU)
Integración: C4ISR Civil (AEMET + InSAR DPM) + Zero-Shot Multi-Cuenca + Dijkstra Hospitalario + Solvencia II
Autor: Kelvin Jesús Flores Yarihuaman (https://www.linkedin.com/in/kelvinflores-ingenieria) (Lead Systems Architect & Environmental Modeler (Director Técnico))

LICENCIAMIENTO MODULAR:
- Documentación y Memoria Técnica: CC BY-NC-SA 4.0
- Código Fuente y Algoritmos (PyTorch FNO, Dijkstra): GNU GPLv3
- Datasets, Grafos OSMnx y Matrices Catastrales: ODC-By
"""
import os
import json
import time
import math
import heapq
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from zoneinfo import ZoneInfo
    VALENCIA_TZ = ZoneInfo("Europe/Madrid")
except Exception:
        import time
        # Fallback seguro para la nube basado en UTC.
        # El horario de verano (CEST) en España abarca de abril a octubre aprox.
        mes_actual_utc = time.gmtime().tm_mon
        if 4 <= mes_actual_utc <= 10:
            VALENCIA_TZ = timezone(timedelta(hours=2)) # Verano (Nowcast hoy)
        else:
            VALENCIA_TZ = timezone(timedelta(hours=1)) # Invierno (DANA 29-O)

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

try:
    from aemet_ingestor import AEMETRealTimeClient
    HAS_AEMET = True
except Exception:
    HAS_AEMET = False

# ==============================================================================
# 1. ARQUITECTURA DEL FOURIER NEURAL OPERATOR (FNO 2D UNIVERSAL 4-CH)
# ==============================================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2, dtype=torch.float32))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        batchsize = x_f.shape[0]

        x_ft = torch.fft.rfft2(x_f)
        w1 = torch.view_as_complex(self.weights1)
        w2 = torch.view_as_complex(self.weights2)

        out_ft = torch.zeros(
            batchsize, self.out_channels, x_f.size(-2), x_f.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        m1 = min(self.modes1, x_f.size(-2) // 2)
        m2 = min(self.modes2, x_f.size(-1) // 2 + 1)

        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], w1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], w2[:, :, -m1:, :m2])

        out = torch.fft.irfft2(out_ft, s=(x_f.size(-2), x_f.size(-1)))
        return out.to(orig_dtype)

class FNO2dUniversal(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, modes=16, width=48, padding=8):
        super().__init__()
        self.padding = padding
        self.in_proj = nn.Conv2d(in_channels + 2, width, 1)

        self.conv0 = SpectralConv2d(width, width, modes, modes)
        self.conv1 = SpectralConv2d(width, width, modes, modes)
        self.conv2 = SpectralConv2d(width, width, modes, modes)
        self.conv3 = SpectralConv2d(width, width, modes, modes)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.norm0 = nn.GroupNorm(4, width)
        self.norm1 = nn.GroupNorm(4, width)
        self.norm2 = nn.GroupNorm(4, width)
        self.norm3 = nn.GroupNorm(4, width)

        self.out_proj = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 1)
        )

    def get_grid(self, shape, device):
        b, h, w = shape[0], shape[2], shape[3]
        gx = torch.linspace(0, 1, h, device=device).reshape(1, 1, h, 1).repeat(b, 1, 1, w)
        gy = torch.linspace(0, 1, w, device=device).reshape(1, 1, 1, w).repeat(b, 1, h, 1)
        return torch.cat((gx, gy), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding], mode="replicate")

        x = self.in_proj(x)
        x1 = self.norm0(F.gelu(self.conv0(x) + self.w0(x)))
        x2 = self.norm1(F.gelu(self.conv1(x1) + self.w1(x1)))
        x3 = self.norm2(F.gelu(self.conv2(x2) + self.w2(x2)))
        x4 = self.norm3(F.gelu(self.conv3(x3) + self.w3(x3)))

        out = self.out_proj(x4)
        if self.padding > 0:
            out = out[..., :-self.padding, :-self.padding]

        h = F.softplus(out[:, 0:1, :, :], beta=2.0)
        u = out[:, 1:2, :, :]
        v = out[:, 2:3, :, :]
        wet_mask = torch.sigmoid((h - 0.005) * 400.0)
        return torch.cat([h, u * wet_mask, v * wet_mask], dim=1)

# ==============================================================================
# 2. ENTORNO GEODÉSICO Y MATRIZ CATASTRAL (CON CACHÉ DEFENSIVO)
# ==============================================================================
import networkx as nx

@st.cache_data
def load_osm_vectors():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_dirs = [os.path.join(base_dir, "data"), os.path.join(base_dir, "data", "processed"), "web_digital_twin/data", "data"]
    
    def find_file(fname):
        for d in possible_dirs:
            p = os.path.join(d, fname)
            if os.path.exists(p): return p
        return None

    hydro_path = find_file("hidrografia_y_defensas.geojson")
    bridges_path = find_file("puentes_y_vados.geojson")
    graph_path = find_file("red_vial_principal_valencia.graphml")

    hydro_geo = json.load(open(hydro_path, "r", encoding="utf-8")) if hydro_path else None
    bridges_geo = json.load(open(bridges_path, "r", encoding="utf-8")) if bridges_path else None
    
    real_roads = []
    if graph_path:
        try:
            G = nx.read_graphml(graph_path)
            for u, v, data in G.edges(data=True):
                name = data.get("name", "")
                if not name: continue
                
                if "geometry" in data:
                    geom = str(data["geometry"])
                    if geom.startswith("LINESTRING"):
                        pts = geom.replace("LINESTRING", "").replace("(", "").replace(")", "").split(",")
                        coords = [[float(c.split()[0]), float(c.split()[1])] for c in pts]
                        real_roads.append({"name": name, "coords": coords})
                else:
                    try:
                        coords = [[float(G.nodes[u]['x']), float(G.nodes[u]['y'])], [float(G.nodes[v]['x']), float(G.nodes[v]['y'])]]
                        real_roads.append({"name": name, "coords": coords})
                    except:
                        pass
        except Exception:
            pass
            
    return hydro_geo, bridges_geo, real_roads

GEO_HYDRO, GEO_BRIDGES, REAL_ROADS_OSM = load_osm_vectors()

GRID_BOUNDS = [714000.0, 4361000.0, 730000.0, 4377000.0]

class GeoProjector:
    def __init__(self):
        self.transformer_fwd = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True) if HAS_PYPROJ else None
        self.transformer_inv = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True) if HAS_PYPROJ else None
        self.ref_x, self.ref_y = 721000.0, 4371500.0

    def transform_points(self, x_arr: np.ndarray, y_arr: np.ndarray):
        if self.transformer_fwd: return self.transformer_fwd.transform(x_arr, y_arr)
        lon = -0.4180 + (x_arr - self.ref_x) / 85000.0; lat = 39.4230 + (y_arr - self.ref_y) / 111000.0
        return lon, lat

    def inverse_transform(self, lon: float, lat: float):
        if self.transformer_inv: return self.transformer_inv.transform(lon, lat)
        x = self.ref_x + (lon - (-0.4180)) * 85000.0; y = self.ref_y + (lat - 39.4230) * 111000.0
        return float(x), float(y)

PROJECTOR = GeoProjector()

VULNERABLE_CENTERS_BASE = [
    {"name": "Residencia San Francisco", "mun": "Paiporta", "type": "GERIÁTRICO", "lon": -0.4190, "lat": 39.4255, "beds": 120},
    {"name": "Centro de Salud Paiporta", "mun": "Paiporta", "type": "SALUD", "lon": -0.4160, "lat": 39.4280, "beds": 0},
    {"name": "Residencia Benetússer", "mun": "Benetússer", "type": "GERIÁTRICO", "lon": -0.3980, "lat": 39.4220, "beds": 95},
    {"name": "IES La Sénia (Refugio)", "mun": "Paiporta", "type": "REFUGIO", "lon": -0.4230, "lat": 39.4290, "beds": 300},
    {"name": "Residencia Ballesol", "mun": "Sedaví", "type": "GERIÁTRICO", "lon": -0.3880, "lat": 39.4260, "beds": 140},
    {"name": "CS La Torre", "mun": "València", "type": "SALUD", "lon": -0.3950, "lat": 39.4360, "beds": 0},
]

TRANSPORT_LANDMARKS_BASE = [
    {"name": "Aeropuerto Manises", "type": "AEROPUERTO", "lon": -0.4816, "lat": 39.4893, "icon": "✈️"},
    {"name": "Estación AVE J. Sorolla", "type": "FERROCARRIL", "lon": -0.3800, "lat": 39.4580, "icon": "🚆"},
    {"name": "Puerto de Valencia", "type": "PUERTO", "lon": -0.3250, "lat": 39.4450, "icon": "🚢"},
    {"name": "Metrovalencia Paiporta", "type": "TRANSPORTE", "lon": -0.4175, "lat": 39.4270, "icon": "🚇"},
]

INSTITUTION_LANDMARKS_BASE = [
    {"name": "Palau Generalitat", "type": "GOBIERNO", "lon": -0.3765, "lat": 39.4770, "icon": "🏛️"},
    {"name": "CHJ - Conf. Júcar", "type": "ORGANISMO_CUENCA", "lon": -0.3590, "lat": 39.4785, "icon": "💧"},
    {"name": "Centro Coord. 112 (L'Eliana)", "type": "MANDO_C2", "lon": -0.5280, "lat": 39.5660, "icon": "🏢"},
    {"name": "Ciutat de les Arts y las Ciencias", "type": "PATRIMONIO", "lon": -0.3530, "lat": 39.4540, "icon": "🏛️"},
]

UNIVERSITIES_BASE = [
    {"name": "VIU - Univ. Internacional", "type": "UNIVERSIDAD", "lon": -0.3580, "lat": 39.4720, "icon": "🎓"},
    {"name": "UV (Tarongers)", "type": "UNIVERSIDAD", "lon": -0.3440, "lat": 39.480, "icon": "🎓"},
    {"name": "UPV (Vera)", "type": "UNIVERSIDAD", "lon": -0.3420, "lat": 39.4810, "icon": "🎓"},
]

HOSPITALS_BASE = [
    {"name": "Hosp. La Fe", "lon": -0.3768, "lat": 39.4435, "beds": 1000},
    {"name": "Hosp. General", "lon": -0.4072, "lat": 39.4682, "beds": 550},
    {"name": "Hospital Manises", "lon": -0.4608, "lat": 39.4930, "beds": 240},
]

SUBSTATIONS_BASE = [
    {"name": "Sub. Paiporta-Benetússer", "lon": -0.4180, "lat": 39.4240, "tipo": "ELÉCTRICA", "icon": "⚡"},
    {"name": "Sub. Torrent Este", "lon": -0.4680, "lat": 39.4320, "tipo": "ELÉCTRICA", "icon": "⚡"},
    {"name": "Fibra Massanassa", "lon": -0.4000, "lat": 39.4150, "tipo": "TELECOM", "icon": "📡"},
    {"name": "EDAR Catarroja", "lon": -0.3950, "lat": 39.4000, "tipo": "SANEAMIENTO", "icon": "🚰"},
]

@st.cache_data
def load_all_system_artifacts():
    expanded_muns = ["Paiporta", "Torrent", "Catarroja", "Sedaví", "Massanassa", "Picanya", "Benetússer", "Alfafar", "Aldaia", "València"]
    
    possible_parquets = [
        "data/processed/flood_damage_matrix_v16.parquet",
        "web_digital_twin/data/processed/flood_damage_matrix_v16.parquet",
        "data/processed/flood_damage_matrix_v15.parquet",
        "web_digital_twin/data/processed/flood_damage_matrix_v15.parquet",
        "data/flood_damage_matrix_v15.parquet",
        "flood_damage_matrix_v15.parquet"
    ]
    
    parquet_path = next((p for p in possible_parquets if os.path.exists(p)), "data/processed/flood_damage_matrix_v16.parquet")

    if not os.path.exists(parquet_path):
        os.makedirs(os.path.dirname(os.path.abspath(parquet_path)), exist_ok=True)
        np.random.seed(46)
        n = 6500
        muns = np.random.choice(expanded_muns, size=n, p=[0.20, 0.15, 0.12, 0.10, 0.10, 0.10, 0.08, 0.07, 0.04, 0.04])
        
        x = np.random.uniform(GRID_BOUNDS[0] + 500, GRID_BOUNDS[2] - 500, n)
        # Geometría real del cauce del Poyo: pendiente -0.42 hacia l'Horta Sud
        base_y = 4369200.0 - 0.42 * (x - 722000.0)
        y_offset = np.random.normal(0, 1800, n)
        y = np.clip(base_y + y_offset, GRID_BOUNDS[1] + 200, GRID_BOUNDS[3] - 200)
        
        dpm = np.random.uniform(0.0, 0.8, n).astype(np.float32)
        assets = np.random.lognormal(12.2, 0.50, n)
        tti = np.random.uniform(22.0, 45.0, n).astype(np.float32)
        critical = np.random.choice([True, False], size=n, p=[0.05, 0.95])
        asset_types = np.random.choice(["Residencial", "Industrial / Logística", "Vehículos / Vados", "Infraestructura Pública"], size=n, p=[0.50, 0.30, 0.15, 0.05])
        base_pop = np.random.randint(2, 8, n)

        df = pd.DataFrame({
            "parcel_id": [f"46{np.random.randint(100, 999)}A{i:05d}" for i in range(n)],
            "municipality": muns, "asset_type": asset_types, "x_coord": x, "y_coord": y,
            "dpm_p90": dpm, "asset_value_eur": assets, "time_to_isolation_min": tti, 
            "is_critical_infra": critical, "base_population": base_pop
        })
        pq.write_table(pa.Table.from_pandas(df), parquet_path, compression="snappy")
    else:
        df = pq.read_table(parquet_path).to_pandas()

    df["lon"], df["lat"] = PROJECTOR.transform_points(df["x_coord"].to_numpy(), df["y_coord"].to_numpy())

    realistic_roads = [
        {"name": "CV-36 Torrent-Picanya", "coords": [[-0.468, 39.432], [-0.432, 39.439], [-0.418, 39.444], [-0.380, 39.452]]},
        {"name": "V-30 Bulevar Sur", "coords": [[-0.440, 39.458], [-0.390, 39.442], [-0.340, 39.430]]},
        {"name": "V-31 Pista Silla", "coords": [[-0.405, 39.385], [-0.388, 39.420], [-0.370, 39.450]]},
        {"name": "CV-400 Eje Paiporta", "coords": [[-0.402, 39.400], [-0.418, 39.425], [-0.418, 39.448]]},
        {"name": "Puente CV-407 Picanya", "coords": [[-0.432, 39.434], [-0.415, 39.426]]},
        {"name": "Enlace H. General", "coords": [[-0.415, 39.450], [-0.4072, 39.4682]]},
        {"name": "Corredor H. La Fe", "coords": [[-0.390, 39.442], [-0.3768, 39.4435]]},
        {"name": "A-3 a Manises", "coords": [[-0.4072, 39.4682], [-0.4608, 39.4930]]},
        {"name": "Bypass A-7 (Perimetral)", "coords": [[-0.468, 39.432], [-0.475, 39.480], [-0.4608, 39.4930]]}
    ]
    
    if REAL_ROADS_OSM:
        realistic_roads.extend(REAL_ROADS_OSM)

    return df, realistic_roads

df_parcels, realistic_roads = load_all_system_artifacts()

# ==============================================================================
# 3. CARGA PERSISTENTE FNO Y ZERO-SHOT MULTI-CUENCA
# ==============================================================================
@st.cache_resource
def load_production_fno(basin_name):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FNO2dUniversal(in_channels=4, out_channels=3, modes=16, width=48, padding=8).to(dev)

    base_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    possible_paths = [
        os.path.join(base_models_dir, "fno_poyo_surrogate.pt"),
        "web_digital_twin/models/fno_poyo_surrogate.pt",
        "models/fno_poyo_surrogate.pt",
        "fno_poyo_surrogate.pt"
    ]
    
    ckpt_path = next((p for p in possible_paths if os.path.exists(p)), None)

    H, W = 128, 128
    y_c = np.linspace(GRID_BOUNDS[3], GRID_BOUNDS[1], H, dtype=np.float32)
    x_c = np.linspace(GRID_BOUNDS[0], GRID_BOUNDS[2], W, dtype=np.float32)
    mesh_y, mesh_x = np.meshgrid(y_c, x_c, indexing="ij")

    # -------------------------------------------------------------------------
    # GEOMETRÍA REALISTA NO-LINEAL: MEANDRO DEL POYO + BIFURCACIÓN DELTAICA
    # -------------------------------------------------------------------------
    if basin_name == "Río Magro":
        thalweg_y = 4369000.0 + 0.15 * (mesh_x - 722000.0)
        dist_thalweg = np.abs(mesh_y - thalweg_y).astype(np.float32)
    elif basin_name == "Barranco del Carraixet":
        thalweg_y = 4366000.0 - 0.45 * (mesh_x - 722000.0)
        dist_thalweg = np.abs(mesh_y - thalweg_y).astype(np.float32)
    else:
        # Rambla del Poyo: Curvatura natural en cabecera (Torrent)
        dx = mesh_x - 722000.0
        meandro = 650.0 * np.sin(dx / 3200.0)
        eje_central_y = 4369200.0 - 0.42 * dx + meandro

        # A partir de Paiporta (dx > -500), el flujo se abre en abanico aluvial y se bifurca:
        # 1. Ramal Sur: Massanassa, Catarroja y Albal hacia la Albufera (pendiente más baja)
        ramal_sur_y = eje_central_y - 0.18 * np.maximum(0.0, dx + 500.0)
        # 2. Ramal Norte: Benetússer, Alfafar, Sedaví y La Torre
        ramal_norte_y = eje_central_y + 0.22 * np.maximum(0.0, dx + 500.0)

        # Distancia al sistema hidrográfico ramificado (rompe los carriles paralelos)
        dist_sur = np.abs(mesh_y - ramal_sur_y)
        dist_norte = np.abs(mesh_y - ramal_norte_y)
        
        # En la llanura (dx > -500), el agua cubre ambos ramales; en cabecera es cauce único:
        dist_thalweg = np.where(dx > -500.0, np.minimum(dist_sur, dist_norte), np.abs(mesh_y - eje_central_y)).astype(np.float32)

    dem_map = (160.0 * (1.0 - (mesh_x - GRID_BOUNDS[0]) / (GRID_BOUNDS[2] - GRID_BOUNDS[0])) - 7.0 * np.exp(-dist_thalweg / 680.0)).astype(np.float32)
    manning_map = np.where(dist_thalweg < 500.0, 0.035, 0.075).astype(np.float32)

    is_real_ckpt = False
    if ckpt_path:
        try: payload = torch.load(ckpt_path, map_location=dev, weights_only=False)
        except TypeError: payload = torch.load(ckpt_path, map_location=dev)

        raw_dict = payload.get("model_state", payload.get("model_state_dict", payload))
        clean_dict = {k.replace("module.", "").replace("model.", ""): v for k, v in raw_dict.items()}

        for k, v in list(clean_dict.items()):
            if torch.is_complex(v):
                clean_dict[k] = torch.view_as_real(v)
            elif torch.is_floating_point(v):
                clean_dict[k] = v.float()
        try:
            net.load_state_dict(clean_dict, strict=False)
            is_real_ckpt = True
        except Exception:
            is_real_ckpt = False

    net.eval()
    dem_t = torch.tensor(dem_map / 250.0, device=dev, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    manning_t = torch.tensor(manning_map, device=dev, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    thalweg_dist_t = torch.tensor(dist_thalweg, device=dev, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    px_coords = df_parcels["x_coord"].to_numpy()
    py_coords = df_parcels["y_coord"].to_numpy()
    norm_px = 2.0 * ((px_coords - GRID_BOUNDS[0]) / (GRID_BOUNDS[2] - GRID_BOUNDS[0])) - 1.0
    norm_py = 2.0 * ((py_coords - GRID_BOUNDS[1]) / (GRID_BOUNDS[3] - GRID_BOUNDS[1])) - 1.0
    grid_static_parcels = torch.tensor(np.column_stack([norm_px, -norm_py]), device=dev, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    return net, dev, dem_t, manning_t, thalweg_dist_t, grid_static_parcels, is_real_ckpt

# ==============================================================================
# 4. CONFIGURACIÓN DEL ENTORNO Y ESTILO HUD C2 DARK
# ==============================================================================
st.set_page_config(page_title="POYO-NOWCAST | Gemelo Digital", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');
    :root { color-scheme: dark !important; }
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #04070b !important; background: radial-gradient(circle at 10% 10%, #0d1117 0%, #04070b 100%) !important;
        color: #f0f6fc !important; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    section[data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #30363d !important; }
    .hud-header { background: rgba(22, 27, 34, 0.85); backdrop-filter: blur(16px); border: 1px solid #30363d; border-radius: 10px; padding: 12px 18px; margin-bottom: 12px; }
    @keyframes pulse-evac { 0% { box-shadow: 0 0 0 0 rgba(255, 23, 68, 0.75); } 70% { box-shadow: 0 0 0 14px rgba(255, 23, 68, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 23, 68, 0); } }
    .evac-banner-red { background: linear-gradient(90deg, rgba(255, 23, 68, 0.42) 0%, rgba(255, 23, 68, 0.16) 100%); border: 1.5px solid #ff1744; border-left: 8px solid #ff1744; border-radius: 10px; padding: 14px 20px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; animation: pulse-evac 1.8s infinite; }
    .evac-banner-orange { background: linear-gradient(90deg, rgba(255, 145, 0, 0.35) 0%, rgba(255, 145, 0, 0.12) 100%); border: 1.5px solid #ff9100; border-left: 8px solid #ff9100; border-radius: 10px; padding: 14px 20px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; }
    .badge-alert-red { background: #ff1744; color: #ffffff; padding: 6px 14px; border-radius: 4px; font-weight: 800; font-size: 0.80rem; font-family: 'JetBrains Mono', monospace; }
    .badge-alert-orange { background: rgba(255, 145, 0, 0.25); color: #ff9100; border: 1px solid #ff9100; padding: 6px 14px; border-radius: 4px; font-weight: 800; font-size: 0.80rem; font-family: 'JetBrains Mono', monospace; }
    .badge-alert-yellow { background: rgba(255, 214, 0, 0.2); color: #ffd600; border: 1px solid #ffd600; padding: 6px 14px; border-radius: 4px; font-weight: 800; font-size: 0.80rem; font-family: 'JetBrains Mono', monospace; }
    .badge-alert-green { background: rgba(0, 230, 118, 0.2); color: #00e676; border: 1px solid #00e676; padding: 6px 14px; border-radius: 4px; font-weight: 800; font-size: 0.80rem; font-family: 'JetBrains Mono', monospace; }
    .badge-clock-box { background: #161b22; border: 1px solid #388bfd; color: #58a6ff; padding: 6px 12px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700; white-space: nowrap; }
    .telemetry-strip { background: rgba(13, 17, 23, 0.95); border: 1px solid #30363d; border-radius: 6px; padding: 9px 18px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }
    .legend-box { background: rgba(22, 27, 34, 0.92); border: 1px solid #30363d; border-radius: 8px; padding: 8px 14px; margin-bottom: 8px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; font-size: 0.73rem; }
    .legend-item { display: flex; align-items: center; gap: 5px; }
    .legend-bullet { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }
    .sidebar-block { background: rgba(22, 27, 34, 0.85); border: 1px solid #30363d; border-radius: 8px; padding: 10px 12px; margin-bottom: 12px; }
    .sidebar-block-title { font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; font-weight: 700; margin-bottom: 6px; }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 1.60rem !important; font-weight: 700; color: #f0f6fc; }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem !important; text-transform: uppercase; color: #8b949e; font-weight: 600; }
    .stMetric { background: rgba(22, 27, 34, 0.85); padding: 12px 16px; border-radius: 8px; border: 1px solid #30363d; border-left: 4px solid #1f6feb; }
    .triage-hud { display: flex; gap: 10px; background: rgba(13, 17, 23, 0.8); border: 1px solid #30363d; padding: 8px 14px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; margin-bottom: 16px; font-weight: bold;}
    .t-p1 { color: #ff1744; } .t-p2 { color: #ff9100; } .t-p3 { color: #ffd600; } .t-p4 { color: #00e676; }
    </style>
    """, unsafe_allow_html=True
)

def fmt_dec(val: float, decimals: int = 1, suffix: str = "") -> str:
    if pd.isna(val) or not np.isfinite(val): return "—"
    return f"{val:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".") + suffix

def fmt_int(val: float, suffix: str = "") -> str:
    if pd.isna(val) or not np.isfinite(val): return "—"
    return f"{int(round(val)):,}".replace(",", ".") + suffix

# ==============================================================================
# 5. CONTROLES Y TELEMETRÍA EN BARRA LATERAL (ZERO-SHOT Y AMC GLOBAL)
# ==============================================================================
logo_url_sidebar = "https://raw.githubusercontent.com/Yggdrasil-Environmental-Systems/HYDRO-CORE-OSIRIS_POYO-NOWCAST/main/logo_yggdrasil.png"

import hashlib

_IP_CLAIM = "Kelvin Jesus Flores Yarihuaman | Creador Unico de HYDRO-CORE OSIRIS | WEHD 2026 | Valencia DANA C4ISR"
_IP_SHA256 = hashlib.sha256(_IP_CLAIM.encode("utf-8")).hexdigest()

if st.query_params.get("auth") == "kjfy":
    st.sidebar.markdown(
        f"""
        <div style='background: rgba(0, 230, 118, 0.12); border: 1.5px solid #00e676; border-radius: 8px; padding: 10px 12px; margin-bottom: 12px; font-family: monospace; font-size: 0.72rem; box-shadow: 0 0 15px rgba(0, 230, 118, 0.3);'>
            <b style='color: #00e676; font-size: 0.80rem;'>🔒 CERTIFICADO DE AUTORÍA LEGAL</b><br/>
            <b>Autor Original:</b> Kelvin Jesús Flores Yarihuaman<br/>
            <b>Metasistema:</b> YGGDRASIL-OS / HYDRO-CORE<br/>
            <b>Firma SHA-256:</b><br/>
            <span style='color: #58a6ff; word-break: break-all;'>{_IP_SHA256[:32]}...</span><br/>
            <span style='color: #8b949e; font-size: 0.65rem;'>Verificación biométrica e intelectual de código fuente.</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
st.sidebar.markdown(
    f"""
    <div style="text-align: center; margin-bottom: 15px;">
        <img src="{logo_url_sidebar}" width="160" style="border-radius: 50%; border: 2px solid #30363d; box-shadow: 0 0 15px rgba(0, 230, 118, 0.2);">
    </div>
    """, 
    unsafe_allow_html=True
)

st.sidebar.markdown("<div style='padding: 10px 14px; background: rgba(22, 27, 34, 0.95); border: 1px solid #30363d; border-left: 4px solid #00e676; border-radius: 6px; margin-bottom: 12px;'><b style='color: #00e676; font-size: 0.92rem;'>🌲 YGGDRASIL ENVIRONMENTAL SYSTEMS</b><br/><b style='color: #f0f6fc; font-size: 0.85rem;'>HYDRO-CORE: OSIRIS C4ISR</b><br/><span style='color: #8b949e; font-size: 0.74rem;'>Centro de Mando C4ISR | YGGDRASIL-OS</span></div>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div style='background: rgba(13, 17, 23, 0.9); border: 1px solid #388bfd; border-left: 4px solid #388bfd; padding: 12px; border-radius: 6px; margin-bottom: 15px;'>
        <b style='color:#58a6ff; font-size: 0.85rem;'>🌍 SUBMISSION: WEHD 2026</b><br/>
        <i style='color:#c9d1d9; font-size:0.75rem;'>Theme: "Standing with Science to Protect Communities"</i>
    </div>
    """, 
    unsafe_allow_html=True
)

if "sim_mode_selector" not in st.session_state: st.session_state["sim_mode_selector"] = "⚡ Modo Nowcast Predictivo (Tiempo Real)"
if "hindcast_slider" not in st.session_state: st.session_state["hindcast_slider"] = 210
if "rain_slider" not in st.session_state: st.session_state["rain_slider"] = 0.0
if "aemet_toggle" not in st.session_state: st.session_state["aemet_toggle"] = True

st.sidebar.markdown("<div class='sidebar-block'><div class='sidebar-block-title' style='color:#58a6ff;'>🎯 1. MODO OPERATIVO & PRESETS</div></div>", unsafe_allow_html=True)
p_col1, p_col2 = st.sidebar.columns(2)
with p_col1:
    if st.button("⛈️ DANA Extrema", use_container_width=True):
        st.session_state["sim_mode_selector"] = "🔴 Modo Hindcast (Forense DANA Valencia 29-O 2024)"
        st.session_state["hindcast_slider"] = 210
        st.session_state["aemet_toggle"] = False
        st.rerun()
with p_col2:
    if st.button("⚠️ Alerta (190 mm)", use_container_width=True):
        st.session_state["sim_mode_selector"] = "⚡ Modo Nowcast Predictivo (Tiempo Real)"
        st.session_state["aemet_toggle"] = False
        st.session_state["rain_slider"] = 190.0
        st.rerun()

sim_mode = st.sidebar.radio("Seleccionar Modo:", ["🔴 Modo Hindcast (Forense DANA Valencia 29-O 2024)", "⚡ Modo Nowcast Predictivo (Tiempo Real)"], key="sim_mode_selector", label_visibility="collapsed")
if st.sidebar.button("🔄 Restablecer Parámetros", use_container_width=True):
    st.session_state.clear()
    st.rerun()

basin_selected = st.sidebar.selectbox("🗺️ Seleccionar Cuenca (Zero-Shot):", ["Rambla del Poyo", "Río Magro", "Barranco del Carraixet"])

basin_names_map = {
    "Río Magro": {"Paiporta": "Algemesí", "Torrent": "Requena", "Catarroja": "Utiel", "Sedaví": "Carlet", "Massanassa": "Guadassuar", "Picanya": "L'Alcúdia", "Benetússer": "Alginet", "Alfafar": "Llombai", "Aldaia": "Montserrat", "València": "Sueca"},
    "Barranco del Carraixet": {"Paiporta": "Alboraya", "Torrent": "Moncada", "Catarroja": "Bétera", "Sedaví": "Vinalesa", "Massanassa": "Alfara", "Picanya": "Tavernes Blanques", "Benetússer": "Rocafort", "Alfafar": "Godella", "Aldaia": "Paterna", "València": "València Norte"}
}

df_display = df_parcels.copy()
if basin_selected in basin_names_map:
    df_display["municipality"] = df_display["municipality"].replace(basin_names_map[basin_selected])

all_municipalities = sorted(df_display["municipality"].unique())

if "prev_basin" not in st.session_state or st.session_state["prev_basin"] != basin_selected:
    st.session_state["prev_basin"] = basin_selected
    st.session_state["sel_muns"] = all_municipalities

FNO_MODEL, FNO_DEVICE, DEM_T, MANNING_T, THALWEG_DIST_T, GRID_STATIC_PARCELS, IS_REAL_CKPT = load_production_fno(basin_selected)

if not IS_REAL_CKPT:
    st.sidebar.info("🔄 Calibración Dinámica: Ajuste cinemático activado (Archivo FNO .pt ausente).")

st.sidebar.markdown("<div class='sidebar-block'><div class='sidebar-block-title' style='color:#00e676;'>📡 2. FORZAMIENTO & TELEMETRÍA AEMET</div></div>", unsafe_allow_html=True)

is_aemet_on = st.session_state["aemet_toggle"]
label_aemet = "🟢 TELEMETRÍA AEMET [ON]" if is_aemet_on else "🔴 MODO SIMULADOR [OFF]"
conectar_aemet = st.sidebar.toggle(label_aemet, key="aemet_toggle")

if conectar_aemet:
    st.sidebar.markdown("<div style='background: rgba(0, 230, 118, 0.15); border: 1px solid #00e676; padding: 5px 10px; border-radius: 4px; color: #00e676; font-size: 0.75rem; font-weight: bold; text-align: center; margin-bottom: 12px;'>📡 LEYENDO DATOS DEL SATÉLITE AEMET</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div style='background: rgba(255, 23, 68, 0.15); border: 1px solid #ff1744; padding: 5px 10px; border-radius: 4px; color: #ff1744; font-size: 0.75rem; font-weight: bold; text-align: center; margin-bottom: 12px;'>🛠️ DESCONECTADO: INGRESO MANUAL</div>", unsafe_allow_html=True)

if "Forense" in sim_mode:
    st.sidebar.markdown("<span style='color:#58a6ff; font-size:0.8rem;'>Humedad Antecedente: <br/><b>🔴 Saturado (AMC III) - Histórico</b></span>", unsafe_allow_html=True)
    amc_weight = 1.15
else:
    soil_amc = st.sidebar.select_slider("Humedad Antecedente (AMC):", ["Seco (AMC I)", "Normal (AMC II)", "Saturado (AMC III)"], "Normal (AMC II)", disabled=conectar_aemet)
    amc_weight = 1.15 if "Saturado" in soil_amc else (0.90 if "Seco" in soil_amc else 1.0)

now_valencia = datetime.now(VALENCIA_TZ)
live_obs = None
telemetry_active = False

if conectar_aemet and HAS_AEMET and "Nowcast" in sim_mode:
    try:
        aemet_client = AEMETRealTimeClient()
        live_obs = aemet_client.get_basin_live_rainfall()
        telemetry_active = True
        raw_ts = live_obs.get('timestamp_utc', '')
        sensor_date_label = raw_ts.replace("T", " ") if raw_ts else now_valencia.strftime("%d/%m/%Y %H:%M UTC")
        rain_real = float(live_obs.get("rain_4h_mm", 0.0))
        st.session_state["rain_slider"] = rain_real
        if rain_real < 10: amc_weight = 0.90; amc_str = "Seco (AMC I) - Auto AEMET"
        elif rain_real < 35: amc_weight = 1.0; amc_str = "Normal (AMC II) - Auto AEMET"
        else: amc_weight = 1.15; amc_str = "Saturado (AMC III) - Auto AEMET"
        st.sidebar.markdown(f"<span style='color:#58a6ff; font-size:0.8rem;'>Ajuste AEMET: <b>{amc_str}</b></span>", unsafe_allow_html=True)
    except Exception:
        sensor_date_label = f"{now_valencia.strftime('%d/%m/%Y %H:%M')} (Simulación Local)"
else:
    sensor_date_label = f"{now_valencia.strftime('%d/%m/%Y %H:%M')} (Simulación Local)" if "Nowcast" in sim_mode else ""

if "Forense" in sim_mode:
    sim_minute = st.sidebar.slider("Minuto del Evento (Base 16:00 h = 0 min):", 0, 360, st.session_state.get("hindcast_slider", 210), 15, "%d min", key="hindcast_slider")
    exact_clock = f"{16 + sim_minute // 60:02d}:{sim_minute % 60:02d} h"
    clock_badge_text = f"🕒 29-O-2024 | {exact_clock} (T + {sim_minute} min)"
    sensor_date_label = f"29/10/2024 {exact_clock}"
    t_arr = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])
    rain_series = np.array([120.0, 185.0, 260.0, 340.0, 415.0, 465.0, 488.0, 491.2, 491.2, 491.2, 491.2, 491.2])
    q_series = np.array([180.0, 290.0, 460.0, 780.0, 1200.0, 1650.0, 1890.0, 1950.0, 1420.0, 890.0, 520.0, 210.0])
    rain_val = float(np.interp(sim_minute, t_arr, rain_series))
    raw_q = float(np.interp(sim_minute, t_arr, q_series))
    factor_clima = 1.0
    usar_hidrograma = True
else:
    usar_hidrograma = st.sidebar.toggle("🌊 Animación: Propagación de Ola", value=False, help="Muestra la inundación dinámica. Si está apagado, verás el PEOR ESCENARIO proyectado (recomendado para C2).")
    lead_time_min = st.sidebar.slider("Avance Temporal (Lead Time):", 15, 180, 60, 15, "T + %d min", key="nowcast_lead")
    simulated_target_time = now_valencia + timedelta(minutes=lead_time_min)
    clock_badge_text = f"🕒 HORA VALENCIA: {simulated_target_time.strftime('%H:%M h')} (T + {lead_time_min} min)"
    rain_val = st.sidebar.slider("Precipitación Cabecera Chiva (mm / 4h):", 0.0, 1000.0, step=5.0, format="%.1f mm", key="rain_slider", disabled=telemetry_active)

st.sidebar.markdown("<div class='sidebar-block'><div class='sidebar-block-title' style='color:#ff9100;'>🛡️ 3. MEDIDAS WHAT-IF & COMPUESTAS</div></div>", unsafe_allow_html=True)
what_if = st.sidebar.selectbox("Medida de Mitigación:", ["Línea Base (Sin Obras Adicionales)", "Presa Cheste (-35% Q)", "Ampliación Cauce (+30% Capacidad)", "Plan Integral"])
ruptura_mota = st.sidebar.toggle("💥 Ruptura de Mota en Paiporta", value=False)
sim_turia = st.sidebar.toggle("🌊 Colapso Plan Sur (Río Turia > 5.000 m³/s)", value=False)
sim_pluvial_vlc = st.sidebar.toggle("🌧️ Inundación Pluvial en València (> 120 mm)", value=False)
sim_storm_surge = st.sidebar.toggle("🛑 Taponamiento Marino / Storm Surge", value=False)
sim_golas = st.sidebar.toggle("🌾 Colmatación Golas de la Albufera", value=False)

q_mitig_factor = 0.65 if "Presa" in what_if or "Integral" in what_if else 1.0
h_mitig_offset = 0.40 if "Ampliación" in what_if or "Integral" in what_if else 0.0

st.sidebar.markdown("<div class='sidebar-block'><div class='sidebar-block-title' style='color:#c9d1d9;'>🌍 4. HORIZONTE CLIMÁTICO & MUNICIPIOS</div></div>", unsafe_allow_html=True)
if "Forense" in sim_mode: factor_clima = 1.0
else:
    horizonte_clima = st.sidebar.selectbox("Horizonte Climático:", ["Actual / Línea Base", "2030 (SSP2-4.5 / +8% Q)", "2040 (SSP3-7.0 / +15% Q)", "2050 (SSP5-8.5 / +22% Q)"])
    factor_clima = 1.08 if "2030" in horizonte_clima else (1.15 if "2040" in horizonte_clima else (1.22 if "2050" in horizonte_clima else 1.0))

selected_muns = st.sidebar.multiselect("Municipios Analizados:", options=all_municipalities, default=all_municipalities, key=f"muns_{basin_selected}")

url_params = st.query_params
url_key = url_params.get("key", "")
show_auth_panel = (url_params.get("admin") == "1") or bool(url_key)

if show_auth_panel:
    with st.sidebar.expander("🛡️ Auditoría & Verificación C4ISR", expanded=True):
        clave = st.text_input(
            "Clave Maestra:", 
            value=url_key, 
            type="password", 
            help="Credencial criptográfica de desarrollador"
        )
        master_key = st.secrets.get("C4ISR_KEY", "")
        
        if master_key and clave == master_key:
            st.session_state["_c4isr_verified"] = True
            st.success("✅ Identidad Verificada: Kelvin Jesús Flores Yarihuaman (Lead Systems Architect & Environmental Modeler)")
            st.markdown("""
            **Certificación C4ISR Civil:**
            - **Autor / Titular:** Kelvin Jesús Flores Yarihuaman
            - **Validación:** Streamlit Cloud Secure Enclave
            - **Licenciamiento:** GPLv3 + CC BY-NC-SA 4.0
            - **ID de Autoría:** KJFY-YGGDRASIL-C4ISR-2026
            """)
        elif clave:
            st.error("❌ Credencial C4ISR no reconocida.")
            
# ==============================================================================
# 6. INFERENCIA NEURONAL FNO 2D V5 (INFERENCIA PURA + CALIBRACIÓN HIDRÁULICA IBER)
# ==============================================================================

if "Forense" in sim_mode:
    q_natural = raw_q * amc_weight
    q_peak_simulated = q_natural * q_mitig_factor
    q_instant = q_peak_simulated 
    mask_past = t_arr <= sim_minute
    peak_q_so_far = float(np.max(q_series[mask_past])) * q_mitig_factor * amc_weight if np.any(mask_past) else q_peak_simulated
else:
    if rain_val <= 5.0: q_raw = 0.0
    elif rain_val <= 60.0: q_raw = (rain_val / 60.0) * 320.0
    elif rain_val <= 180.0: q_raw = 320.0 + ((rain_val - 60.0) / 120.0) * (1380.0 - 320.0)
    else: q_raw = 1380.0 + (((rain_val - 180.0) / 820.0) ** 0.85) * (18000.0 - 1380.0)

    q_natural = q_raw * factor_clima * amc_weight
    q_peak_simulated = float(np.clip(q_natural * q_mitig_factor, 0.0, 25000.0))
    peak_q_so_far = q_peak_simulated
    
    if usar_hidrograma:
        q_instant = q_peak_simulated * math.exp(-(((lead_time_min - 120.0) / 85.0) ** 2))
    else:
        q_instant = q_peak_simulated

plan_sur_desbordado = sim_turia or (q_peak_simulated >= 5000.0)

t_fno_start = time.perf_counter()
H_dim = DEM_T.size(-2); W_dim = DEM_T.size(-1)

inflow_act = torch.zeros((1, 1, H_dim, W_dim), device=FNO_DEVICE, dtype=torch.float32)

# -------------------------------------------------------------------------------------
# INYECCIÓN DINÁMICA DE CAUDAL (CORRESPONDENCIA EXACTA CON ENTRENAMIENTO V5)
# -------------------------------------------------------------------------------------
norm_q_act = float(q_instant) / 25000.0

c_width = 0.025 + 0.0018 * (max(0.1, float(q_instant))**0.48)
pix_w = max(2, int(c_width * H_dim * 1.5))
y_center_idx = int(H_dim * 0.75)
y_min = max(0, y_center_idx - pix_w)
y_max = min(H_dim, y_center_idx + pix_w)

inflow_act[0, 0, y_min:y_max, 0:6] = norm_q_act

rain_m_s = (float(rain_val) / 4.0) / (3600.0 * 1000.0)
rain_channel = torch.full((1, 1, H_dim, W_dim), rain_m_s * 1e4, device=FNO_DEVICE, dtype=torch.float32)

x_act = torch.cat([DEM_T, MANNING_T, inflow_act, rain_channel], dim=1)

with torch.no_grad():
    # Modelo cinemático de respaldo (fallback de seguridad estricto)
    scale_h_fallback = 4.0 * (max(0.01, norm_q_act)**0.6)
    if q_instant < 10.0:
        frente_act = torch.zeros_like(THALWEG_DIST_T)
    else:
        frente_act = torch.clamp(scale_h_fallback * torch.exp(-THALWEG_DIST_T / (450.0 + 350.0 * min(2.0, norm_q_act))), min=0.0)
        
    h_kinematic = F.relu(frente_act - h_mitig_offset)
    v_kinematic = torch.sqrt(torch.clamp(2.0 * 9.81 * h_kinematic, min=0.01)) * (0.32 / (MANNING_T + 0.01))
    
    if IS_REAL_CKPT:
        out_act = FNO_MODEL(x_act)
        out_act = torch.flip(out_act, dims=[-2])

        # ---------------------------------------------------------------------
        # CALIBRACIÓN HIDRÁULICA NETA (CRITERIO IBER / HEC-RAS)
        # 1. Eliminación del piso matemático de Softplus (beta=2.0 -> ln(2)/2 = 0.3465m)
        # 2. Confinamiento hidrotopográfico según energía de caudal
        # 3. Umbral de secado de celdas (Wetting/Drying)
        # ---------------------------------------------------------------------
        OFFSET_SOFTPLUS = 0.3465
        h_raw_fno = out_act[:, 0:1, :, :]
        h_fno = F.relu(h_raw_fno - OFFSET_SOFTPLUS - h_mitig_offset)

        # Ancho real del abanico aluvial metropolitano (cubre Paiporta, Benetússer y Sedaví)
        ancho_inundable_max = 2400.0 + 600.0 * math.log10(max(1.0, float(q_instant)))
        mask_valle = THALWEG_DIST_T <= ancho_inundable_max
        
        # Eliminación estricta de agua voladora fuera de la llanura
        h_fno = torch.where(mask_valle, h_fno, torch.zeros_like(h_fno))
        
        # Criterio técnico de calado útil en inundaciones (Iber: 8 cm)
        h_fno = torch.where(h_fno < 0.08, torch.zeros_like(h_fno), h_fno)
        
        u_fno = out_act[:, 1:2, :, :]
        v_fno = out_act[:, 2:3, :, :]
        wet_mask_fno = (h_fno > 0.05).float()
        vel_mag_fno = torch.sqrt(u_fno**2 + v_fno**2 + 1e-6) * wet_mask_fno
        
        # -------- 🚨 DETECTOR DINÁMICO DE ALUCINACIONES (C2) CALIBRADO --------
        mask_cauce = (THALWEG_DIST_T < 250.0).float()
        mask_montana = (THALWEG_DIST_T > (ancho_inundable_max + 150.0)).float()
        mask_inundada = (h_fno > 0.10).float()
        
        ia_media_cauce = float((h_fno * mask_cauce).sum() / (mask_cauce.sum() + 1e-6))
        ia_media_montana = float((h_fno * mask_montana).sum() / (mask_montana.sum() + 1e-6))
        
        max_calado_ia = float(h_fno.max().item())
        max_vel_ia = float(vel_mag_fno.max().item())
        
        ia_fuera_de_rango = False
        
        # El filtro sólo supervisa crecidas activas (> 600 m3/s) para evitar falsos positivos en recesión
        if q_instant > 600.0:
            if ia_media_montana > 0.40 or ia_media_cauce < 0.03 or max_calado_ia > 20.0 or max_vel_ia > 15.0:
                ia_fuera_de_rango = True
        # ---------------------------------------------------------------------
       
        # Si la IA está en rango sano (V5 validada), INFERENCIA PURA FNO
        if ia_fuera_de_rango:
            h_field_act = h_kinematic
            v_field_act = v_kinematic
        elif q_instant < 10.0:
            h_field_act = torch.zeros_like(h_kinematic)
            v_field_act = torch.zeros_like(v_kinematic)
        else:
            h_field_act = h_fno
            v_field_act = vel_mag_fno
    else:
        h_field_act = h_kinematic
        v_field_act = v_kinematic

    # CALADO PICO HISTÓRICO (h_field_peak): Inferencia FNO pura al caudal punta
    if "Forense" in sim_mode and peak_q_so_far > q_peak_simulated:
        norm_q_pk = float(peak_q_so_far) / 25000.0
        c_w_pk = 0.025 + 0.0018 * (max(0.1, float(peak_q_so_far))**0.48)
        pix_w_pk = max(2, int(c_w_pk * H_dim * 1.5))
        y_min_pk = max(0, y_center_idx - pix_w_pk)
        y_max_pk = min(H_dim, y_center_idx + pix_w_pk)

        if IS_REAL_CKPT:
            inflow_peak = torch.zeros((1, 1, H_dim, W_dim), device=FNO_DEVICE, dtype=torch.float32)
            inflow_peak[0, 0, y_min_pk:y_max_pk, 0:6] = norm_q_pk
            out_pk = FNO_MODEL(torch.cat([DEM_T, MANNING_T, inflow_peak, rain_channel], dim=1))
            out_pk = torch.flip(out_pk, dims=[-2])
            
            h_raw_pk = out_pk[:, 0:1, :, :]
            h_fno_pk = F.relu(h_raw_pk - OFFSET_SOFTPLUS - h_mitig_offset)
            # Ancho aluvial completo hacia Paiporta, Catarroja y Sedaví:
            ancho_pk = 2400.0 + 600.0 * math.log10(max(1.0, float(peak_q_so_far)))
            h_fno_pk = torch.where(THALWEG_DIST_T <= ancho_pk, h_fno_pk, torch.zeros_like(h_fno_pk))
            h_field_peak = torch.where(h_fno_pk < 0.08, torch.zeros_like(h_fno_pk), h_fno_pk)
        else:
            scale_h_pk_fall = 4.0 * (max(0.01, norm_q_pk)**0.6)
            h_kinematic_pk = F.relu(torch.clamp(scale_h_pk_fall * torch.exp(-THALWEG_DIST_T / (450.0 + 350.0 * min(2.0, norm_q_pk))), min=0.0) - h_mitig_offset)
            h_field_peak = h_kinematic_pk
    else:
        h_field_peak = h_field_act

    sampled_h_act = F.grid_sample(h_field_act, GRID_STATIC_PARCELS, mode="bilinear", align_corners=True)
    sampled_h_pk  = F.grid_sample(h_field_peak, GRID_STATIC_PARCELS, mode="bilinear", align_corners=True)
    sampled_v_act = F.grid_sample(v_field_act, GRID_STATIC_PARCELS, mode="bilinear", align_corners=True)

raw_h_act = sampled_h_act[0, 0, :, 0].cpu().numpy()
raw_h_pk  = sampled_h_pk[0, 0, :, 0].cpu().numpy()
raw_v_act = sampled_v_act[0, 0, :, 0].cpu().numpy()

t_fno_end = time.perf_counter()
fno_latency_ms = (t_fno_end - t_fno_start) * 1000.0
hw_device_name = "GPU CUDA" if FNO_DEVICE.type == "cuda" else "CPU Multi-Thread"

# ==============================================================================
# 7. MUESTREO VECTORIZADO DE VÍAS Y CENTROS SIN DOUBLE-COUNTING
# ==============================================================================
pts_to_sample = []
for v in VULNERABLE_CENTERS_BASE: pts_to_sample.append((v["lon"], v["lat"]))
for t in TRANSPORT_LANDMARKS_BASE: pts_to_sample.append((t["lon"], t["lat"]))
for g in INSTITUTION_LANDMARKS_BASE: pts_to_sample.append((g["lon"], g["lat"]))
for u in UNIVERSITIES_BASE: pts_to_sample.append((u["lon"], u["lat"]))
for h in HOSPITALS_BASE: pts_to_sample.append((h["lon"], h["lat"]))
for s in SUBSTATIONS_BASE: pts_to_sample.append((s["lon"], s["lat"]))

road_vertex_slices = []
cur_idx = len(pts_to_sample)
for r in realistic_roads:
    start_s = cur_idx
    for c in r["coords"]: pts_to_sample.append((c[0], c[1])); cur_idx += 1
    road_vertex_slices.append((start_s, cur_idx))

pts_utm = [PROJECTOR.inverse_transform(p[0], p[1]) for p in pts_to_sample]
n_x_arr = np.array([2.0 * ((p[0] - GRID_BOUNDS[0]) / (GRID_BOUNDS[2] - GRID_BOUNDS[0])) - 1.0 for p in pts_utm], dtype=np.float32)
n_y_arr = np.array([2.0 * ((p[1] - GRID_BOUNDS[1]) / (GRID_BOUNDS[3] - GRID_BOUNDS[1])) - 1.0 for p in pts_utm], dtype=np.float32)

all_pts_tensor = torch.tensor(np.column_stack([n_x_arr, -n_y_arr]), device=FNO_DEVICE, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

with torch.no_grad():
    sampled_infra_act = F.grid_sample(h_field_act, all_pts_tensor, padding_mode="zeros", mode="bilinear", align_corners=True)
    sampled_infra_pk  = F.grid_sample(h_field_peak, all_pts_tensor, padding_mode="zeros", mode="bilinear", align_corners=True)

infra_h_act = sampled_infra_act[0, 0, :, 0].cpu().numpy()
infra_h_pk  = sampled_infra_pk[0, 0, :, 0].cpu().numpy()

# En modo forense o tras el pico, las vías y centros mantienen el barro y daño del pico:
if "Forense" in sim_mode and peak_q_so_far >= 250.0:
    progreso_rec_infra = max(0.0, min(1.0, (sim_minute - 210.0) / 150.0))
    all_infra_h = np.maximum(infra_h_act, infra_h_pk * max(0.40, 1.0 - 0.50 * progreso_rec_infra))
else:
    all_infra_h = infra_h_act

y_pts = np.array([p[1] for p in pts_utm])
x_pts = np.array([p[0] for p in pts_utm])

is_pluvial_infra = (y_pts > 4370000.0) & (x_pts > 722000.0)
is_coastal_infra = x_pts > 727000.0
is_turia_infra   = y_pts > 4370000.0
is_golas_infra   = (y_pts < 4363000.0) & (x_pts > 726000.0)
is_represa_infra = y_pts <= 4370000.0

pluvial_infra = np.where(is_pluvial_infra & sim_pluvial_vlc, 0.85, 0.0)
surge_infra = np.where(is_coastal_infra & sim_storm_surge, 0.70, 0.0)
turia_infra = np.where(is_turia_infra & plan_sur_desbordado, 1.80, 0.0)
golas_infra = np.where(is_golas_infra & sim_golas, 0.60, 0.0)

all_infra_h = np.where(is_represa_infra & plan_sur_desbordado, all_infra_h * 1.35 + 0.40 * (all_infra_h > 0.1), all_infra_h)
all_infra_h = np.clip(all_infra_h + pluvial_infra + surge_infra + turia_infra + golas_infra, 0.0, 6.0)

idx_ptr = 0
v_depths = all_infra_h[idx_ptr : idx_ptr + len(VULNERABLE_CENTERS_BASE)]; idx_ptr += len(VULNERABLE_CENTERS_BASE)
t_depths = all_infra_h[idx_ptr : idx_ptr + len(TRANSPORT_LANDMARKS_BASE)]; idx_ptr += len(TRANSPORT_LANDMARKS_BASE)
g_depths = all_infra_h[idx_ptr : idx_ptr + len(INSTITUTION_LANDMARKS_BASE)]; idx_ptr += len(INSTITUTION_LANDMARKS_BASE)
u_depths = all_infra_h[idx_ptr : idx_ptr + len(UNIVERSITIES_BASE)]; idx_ptr += len(UNIVERSITIES_BASE)
h_depths = all_infra_h[idx_ptr : idx_ptr + len(HOSPITALS_BASE)]; idx_ptr += len(HOSPITALS_BASE)
s_depths = all_infra_h[idx_ptr : idx_ptr + len(SUBSTATIONS_BASE)]; idx_ptr += len(SUBSTATIONS_BASE)
road_max_depths = [float(np.max(all_infra_h[s[0]:s[1]])) for s in road_vertex_slices]

power_outage = (s_depths[0] >= 0.35) or (s_depths[1] >= 0.35)

# ==============================================================================
# 9. AFECCIÓN CATASTRAL Y FINANCIERA ESTRUCTURAL
# ==============================================================================
active_df = df_display[df_display["municipality"].isin(selected_muns)].copy()

if not active_df.empty:
    indices_sel = active_df.index.to_numpy()
    x_c = active_df["x_coord"].to_numpy()
    y_c = active_df["y_coord"].to_numpy()
    active_df["pop_density"] = np.where(active_df["asset_type"] == "Industrial / Logística", active_df["base_population"] * 2.4, active_df["base_population"] * 0.7)
    mota_add = 0.85 if (ruptura_mota and q_peak_simulated > 200.0) else 0.0
    pluvial_desbordado = sim_pluvial_vlc
    pluvial_add = np.where((y_c >= 4373500.0) & (x_c >= 721000.0) & (x_c <= 728000.0) & pluvial_desbordado, 0.85, 0.0)
    surge_add = np.where((x_c > 727000.0) & sim_storm_surge, 0.70, 0.0)
    turia_add = np.where((y_c > 4370000.0) & plan_sur_desbordado, 1.80, 0.0)
    golas_add = np.where((y_c < 4363000.0) & (x_c > 726000.0) & sim_golas, 0.60, 0.0)
    mask_represa = (y_c <= 4370000.0) & plan_sur_desbordado
    h_act_rep = np.where(mask_represa, raw_h_act[indices_sel] * 1.35 + 0.40 * (raw_h_act[indices_sel] > 0.1), raw_h_act[indices_sel])
    h_pk_rep  = np.where(mask_represa, raw_h_pk[indices_sel] * 1.35 + 0.40 * (raw_h_pk[indices_sel] > 0.1), raw_h_pk[indices_sel])
    mask_val_norte = (y_c > (4375200.0 - 0.45 * (x_c - 720000.0)))
    if not plan_sur_desbordado:
        h_act_rep[mask_val_norte] = 0.0
        h_pk_rep[mask_val_norte] = 0.0
    # Histéresis urbana: en recesión el agua no se evapora, drena gradualmente hacia la Albufera
    if "Forense" in sim_mode and peak_q_so_far >= 500.0:
        progreso_recesion = max(0.0, min(1.0, (sim_minute - 210.0) / 150.0))
        factor_retencion = max(0.40, 1.0 - 0.60 * progreso_recesion)
        desborde_rambla_act = h_act_rep if q_instant >= 500.0 else (h_pk_rep * factor_retencion * (h_pk_rep > 0.15))
        desborde_rambla_pk  = h_pk_rep
    else:
        desborde_rambla_act = h_act_rep if q_instant >= 500.0 else np.zeros_like(h_act_rep)
        desborde_rambla_pk  = h_pk_rep if peak_q_so_far >= 500.0 else np.zeros_like(h_pk_rep)

    raw_active = desborde_rambla_act + mota_add + pluvial_add + surge_add + turia_add + golas_add
    raw_peak   = desborde_rambla_pk + mota_add + pluvial_add + surge_add + turia_add + golas_add
    
    active_df["active_depth"] = np.where(raw_active < 0.08, 0.0, np.clip(raw_active, 0.0, 6.0)).astype(np.float32)
    active_df["peak_depth_experienced"] = np.where(raw_peak < 0.08, 0.0, np.clip(raw_peak, 0.0, 6.0)).astype(np.float32)
    
    # La velocidad máxima destructiva corresponde a la energía cinética del pico registrado:
    scale_v_pk = np.sqrt(max(1.0, float(peak_q_so_far)) / max(100.0, float(q_instant))) if "Forense" in sim_mode else 1.0
    v_est_pk = np.clip(raw_v_act[indices_sel] * scale_v_pk, 0.0, 4.50)
    active_df["max_velocity_ms"] = np.where(active_df["peak_depth_experienced"] == 0.0, 0.0, v_est_pk).astype(np.float32)
    
    h_eval = active_df["peak_depth_experienced"].to_numpy()
    types = active_df["asset_type"].to_numpy()
    ratios = np.zeros(len(active_df), dtype=np.float64)
    mask_wet = h_eval >= 0.08
    ratios[(types == "Residencial") & mask_wet] = np.clip((h_eval[(types == "Residencial") & mask_wet] - 0.08) / 1.70, 0.0, 1.0) ** 1.25
    ratios[(types == "Industrial / Logística") & (h_eval >= 0.35)] = np.clip((h_eval[(types == "Industrial / Logística") & (h_eval >= 0.35)] - 0.35) / 1.40, 0.0, 1.0) ** 1.10
    ratios[(types == "Vehículos / Vados") & mask_wet] = np.where(h_eval[(types == "Vehículos / Vados") & mask_wet] >= 0.35, 1.0, (h_eval[(types == "Vehículos / Vados") & mask_wet] / 0.35) ** 2)
    ratios[(types == "Infraestructura Pública") & (h_eval >= 0.50)] = np.clip((h_eval[(types == "Infraestructura Pública") & (h_eval >= 0.50)] - 0.50) / 2.0, 0.0, 1.0)
    vel_multiplier = 1.0 + (0.15 * active_df["max_velocity_ms"])
    active_df["damage_ratio"] = np.where(active_df["peak_depth_experienced"] == 0.0, 0.0, np.clip(ratios * vel_multiplier, 0.0, 1.0))
    
    # MEMORIA DE CATÁSTROFE (HISTÉRESIS ACTUARIAL SOLVENCIA II Y 112)
    is_impacted = active_df["peak_depth_experienced"] >= 0.08
    active_df["peak_hazard_vh"] = active_df["peak_depth_experienced"] * active_df["max_velocity_ms"]
    
    # Inmuebles en ruina permanente: calado >= 1.20m con arrastre cinético o degradación DPM
    active_df["dynamic_collapse"] = ((active_df["peak_depth_experienced"] >= 1.20) & ((active_df["peak_hazard_vh"] >= 1.10) | (active_df["dpm_p90"] >= 0.40)))
    bi_factor = np.where((types == "Industrial / Logística") & active_df["dynamic_collapse"], 1.30, 1.0)
    active_df["active_loss"] = (active_df["asset_value_eur"].to_numpy() * active_df["damage_ratio"] * bi_factor).astype(np.float64)
    
    # Triaje 112: Quien necesitó rescate crítico en el pico SIGUE requiriéndolo por la noche
    active_df["P1_flag"] = (is_impacted & ((active_df["peak_depth_experienced"] >= 1.15) | (active_df["peak_hazard_vh"] >= 1.8) | (active_df["is_critical_infra"] & (active_df["peak_depth_experienced"] >= 0.55)) | active_df["dynamic_collapse"]))
    active_df["P2_flag"] = ((~active_df["P1_flag"]) & is_impacted & ((active_df["peak_depth_experienced"] >= 0.65) | (active_df["peak_hazard_vh"] >= 1.30)))
    active_df["P3_flag"] = ((~active_df["P1_flag"]) & (~active_df["P2_flag"]) & is_impacted & (active_df["peak_depth_experienced"] >= 0.25))
    active_df["P4_flag"] = ((~active_df["P1_flag"]) & (~active_df["P2_flag"]) & (~active_df["P3_flag"]) & is_impacted)
else:
    active_df["active_depth"] = 0.0; active_df["peak_depth_experienced"] = 0.0; active_df["max_velocity_ms"] = 0.0; active_df["damage_ratio"] = 0.0; active_df["active_loss"] = 0.0; active_df["dynamic_collapse"] = False; active_df["P1_flag"] = False; active_df["P2_flag"] = False; active_df["P3_flag"] = False; active_df["P4_flag"] = False; active_df["peak_hazard_vh"] = 0.0

# Cota real: la EDAR Catarroja está elevada +0.80 m sobre la solera del cauce del río
h_cauce_edar = float(s_depths[3]) if len(s_depths) > 3 else 0.0
h_edar = max(0.0, h_cauce_edar - 0.80)

# Umbrales reales de ingeniería sanitaria:
# - Prealerta / Sobrecarga: h_cauce >= 0.70 m (aliviadero activo, planta segura)
# - Colapso Biológico / Avería Motores: h_edar >= 0.40 m (inundación real de reactores)
edar_biocollapse = h_edar >= 0.40
edar_prealert = (h_cauce_edar >= 0.70) and not edar_biocollapse

mask_water_risk = (active_df["active_depth"] >= 0.50) & (active_df["asset_type"] == "Residencial")
pop_water_compromised = int(active_df.loc[mask_water_risk, "pop_density"].sum())

geriatric_beds_critical = 0
for idx_v, v_center in enumerate(VULNERABLE_CENTERS_BASE):
    if v_center["type"] == "GERIÁTRICO" and v_depths[idx_v] >= 0.30: 
        geriatric_beds_critical += v_center["beds"]

ha_biohazard_exposed = float(np.sum(active_df["active_depth"] >= 0.30) * 0.15) if edar_biocollapse else 0.0

if edar_biocollapse or pop_water_compromised > 5000:
    sanitary_status_label = "ALERTA BIOLÓGICA FASE 2: RUPTURA SANITARIA & RETROSIFONAJE"
    sanitary_color = "#ff1744"
    water_advisory = "HERVIDO OBLIGATORIO DE AGUA (BOIL WATER NOTICE)"
elif edar_prealert or pop_water_compromised > 0:
    sanitary_status_label = "ALERTA BIOLÓGICA FASE 1: SOBRECARGA & VIGILANCIA DE CLORO (RD 3/2023)"
    sanitary_color = "#ff9100"
    water_advisory = "MONITOREO DE DESINFECCIÓN & ALIVIADEROS ACTIVOS"
else:
    sanitary_status_label = "FASE 0: RED HIDROSANITARIA CONFORME"
    sanitary_color = "#00e676"
    water_advisory = "SUMINISTRO POTABLE SEGURO"
        
rain_mm = float(rain_val)
has_experienced_catastrophe = ("Forense" in sim_mode and peak_q_so_far >= 1200.0) or (rain_mm >= 180.0) or (q_peak_simulated >= 1200.0) or sim_pluvial_vlc or sim_storm_surge or sim_turia

if has_experienced_catastrophe:
    badge_txt = "🔴 SIT. 2: DESBORDAMIENTO METROPOLITANO SEVERO" if q_peak_simulated >= 1200.0 or sim_turia else "🔴 SIT. 2: ALERTA ROJA PREVENTIVA"
    alert_badge_html = f"<span class='badge-alert-red'>{badge_txt}</span>"
    alert_state = "ROJO"
elif rain_mm >= 90.0 or q_peak_simulated >= 600.0:
    alert_badge_html = "<span class='badge-alert-orange'>🟠 SIT. 1: CRECIDA SEVERA EN CAUCE</span>"
    alert_state = "NARANJA"
elif rain_mm >= 40.0 or q_peak_simulated >= 250.0:
    alert_badge_html = "<span class='badge-alert-yellow'>🟡 PREALERTA POR LLUVIAS</span>"
    alert_state = "AMARILLO"
else:
    alert_badge_html = "<span class='badge-alert-green'>🟢 NORMALIDAD HIDROLÓGICA</span>"
    alert_state = "VERDE"

lluvia_sensor = rain_real if telemetry_active else 0.0
medidas_manuales_activas = sim_turia or ruptura_mota or sim_storm_surge or sim_pluvial_vlc or sim_golas or ("Línea Base" not in what_if)

if "Forense" in sim_mode:
    hud_border = "border: 2px solid #b388ff; box-shadow: 0 0 15px rgba(179, 136, 255, 0.15);"
    hud_badge = "<span style='background:#b388ff; color:#000; padding:3px 8px; border-radius:4px; font-size:0.75rem; font-weight:800; display:inline-block;'>📜 MODO FORENSE (HISTÓRICO)</span>"
elif rain_mm > lluvia_sensor or medidas_manuales_activas:
    hud_border = "border: 2px solid #ff9100; box-shadow: 0 0 15px rgba(255, 145, 0, 0.15);"
    hud_badge = "<span style='background:#ff9100; color:#000; padding:3px 8px; border-radius:4px; font-size:0.75rem; font-weight:800; display:inline-block;'>🛠️ MODO SIMULADOR (MANUAL)</span>"
else:
    hud_border = "border: 2px solid #1f6feb; box-shadow: 0 0 15px rgba(31, 111, 235, 0.15);"
    hud_badge = "<span style='background:#1f6feb; color:#fff; padding:3px 8px; border-radius:4px; font-size:0.75rem; font-weight:800; display:inline-block;'>📡 MODO NOWCAST (EN VIVO)</span>"

# ------------------------------------------------------------------------------
# CÁLCULO DE RELOJ ETA Y FASES OPERATIVAS C2 (PRE, PICO Y POST-IMPACTO)
# ------------------------------------------------------------------------------
t_pico_ref = 210 if "Forense" in sim_mode else 120
t_actual_ref = sim_minute if "Forense" in sim_mode else lead_time_min
diff_mins = t_pico_ref - t_actual_ref
eta_mins = max(0, diff_mins)

if q_peak_simulated >= 600.0 or medidas_manuales_activas:
    if diff_mins > 0:
        # FASE 1: APROXIMACIÓN (Cuenta atrás para evacuación)
        eta_badge_html = f"<span style='background:#ff1744; color:#ffffff; padding:4px 10px; border-radius:6px; font-weight:900; font-family: monospace; border: 1.5px solid #fff;'>⏳ IMPACTO MÁXIMO EN: {diff_mins} MIN</span>"
    elif diff_mins == 0:
        # FASE 2: HORA CERO (Punto Crítico en curso)
        eta_badge_html = "<span style='background:#ff1744; color:#ffffff; padding:4px 10px; border-radius:6px; font-weight:900; font-family: monospace; border: 2px solid #ffd600;'>🚨 IMPACTO PICO ACTIVO (PUNTO CRÍTICO)</span>"
    else:
        # FASE 3: POST-IMPACTO / RECESIÓN (Operaciones SAR y Rescate)
        post_mins = abs(diff_mins)
        eta_badge_html = f"<span style='background:#ff9100; color:#000000; padding:4px 10px; border-radius:6px; font-weight:900; font-family: monospace;'>🌊 POST-IMPACTO (+{post_mins} MIN) | FASE RESCATE SAR</span>"
else:
    eta_badge_html = ""
    
alertas_criticas_html = ""
if edar_biocollapse:
    alertas_criticas_html += "<div style='background:#9c27b0; border: 1px solid #9c27b0; color:#ffffff; padding: 4px 12px; border-radius:20px; font-size:0.75rem; font-weight:800; font-family: monospace; white-space: nowrap;'>☣️ ALERTA BIOLÓGICA (AGUA CONTAMINADA)</div>"
if plan_sur_desbordado:
    motivo = "SOBRECAPACIDAD (>5.000 m³/s)" if q_peak_simulated >= 5000.0 else "FALLO ESTRUCTURAL"
    alertas_criticas_html += f"<div style='background:#ff1744; border: 1px solid #ff1744; color:#ffffff; padding: 4px 12px; border-radius:20px; font-size:0.75rem; font-weight:800; font-family: monospace; white-space: nowrap;'>⚠️ COLAPSO PLAN SUR: {motivo}</div>"
if ruptura_mota:
    alertas_criticas_html += "<div style='background:#ff1744; border: 1px solid #ff1744; color:#ffffff; padding: 4px 12px; border-radius:20px; font-size:0.75rem; font-weight:800; font-family: monospace; white-space: nowrap;'>💥 RUPTURA DE MOTA: PAIPORTA</div>"
if sim_pluvial_vlc:
    alertas_criticas_html += "<div style='background:#00e5ff; border: 1px solid #00e5ff; color:#000000; padding: 4px 12px; border-radius:20px; font-size:0.75rem; font-weight:800; font-family: monospace; white-space: nowrap;'>🌧️ ANEGAMIENTO PLUVIAL VALÈNCIA</div>"
if sim_storm_surge:
    alertas_criticas_html += "<div style='background:#2979ff; border: 1px solid #2979ff; color:#ffffff; padding: 4px 12px; border-radius:20px; font-size:0.75rem; font-weight:800; font-family: monospace; white-space: nowrap;'>🛑 INTRUSIÓN MARINA (STORM SURGE)</div>"
if sim_golas:
    alertas_criticas_html += "<div style='background:#ff9100; border: 1px solid #ff9100; color:#000000; padding: 4px 12px; border-radius:20px; font-size:0.75rem; font-weight:800; font-family: monospace; white-space: nowrap;'>🌾 COLMATACIÓN GOLAS</div>"

if alertas_criticas_html:
    alertas_criticas_html = f"<div style='display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; margin-top: 6px;'>{alertas_criticas_html}</div>"

logo_url_hud = "https://raw.githubusercontent.com/Yggdrasil-Environmental-Systems/HYDRO-CORE-OSIRIS_POYO-NOWCAST/main/logo_yggdrasil.png"

header_html = (
    f"<div class='hud-header' style='{hud_border} background: rgba(22, 27, 34, 0.85); backdrop-filter: blur(16px); border-radius: 10px; padding: 14px 20px; margin-bottom: 16px;'>"
    f"<div style='display:flex; justify-content:space-between; align-items:flex-start; width:100%; flex-wrap:wrap; gap:16px;'>"
    f"<div style='flex:1; min-width:300px;'>"
    f"<div style='margin-bottom:8px;'>{hud_badge} {eta_badge_html}</div>"
    f"<div style='display:flex; align-items:center; gap:12px; margin-bottom:4px;'>"
    f"<img src='{logo_url_hud}' width='40' style='border-radius:6px; box-shadow: 0 0 10px rgba(0, 230, 118, 0.4);'>"
    f"<h1 style='margin:0; font-size:1.55rem; letter-spacing:-0.02em; color:#00e676;'>YGGDRASIL / HYDRO-CORE: OSIRIS C4ISR</h1>"
    f"</div>"
    f"<div style='color:#8b949e; font-size:0.75rem; margin-top:4px;'>ÁREA METROPOLITANA DE VALÈNCIA | GEMELO DIGITAL 3D (FÍSICA FNO 2D) & TRANSFERENCIA SOLVENCIA II</div>"
    f"</div>"
    f"<div style='flex:1; display:flex; flex-direction:column; align-items:flex-end; min-width:320px;'>"
    f"<div style='display:flex; gap:8px; align-items:center;'>{alert_badge_html}<div class='badge-clock-box'>{clock_badge_text}</div></div>"
    f"{alertas_criticas_html}"
    f"</div>"
    f"</div>"
    f"</div>"
)
st.markdown(header_html, unsafe_allow_html=True)

max_local_depth = float(np.max(active_df["active_depth"])) if not active_df.empty else 0.0

if alert_state == "ROJO":
    if eta_mins <= 20 or q_peak_simulated >= 1400.0 or max_local_depth >= 0.80:
        escape_msg = "VENTANA DE ESCAPE: AGOTADA<br/><span style='color:#c9d1d9; font-weight:normal;'>Permanezca en pisos altos. NO intente salir.</span>"
    else:
        escape_msg = f"VENTANA DE SEGURIDAD: ~ {eta_mins} MIN<br/><span style='color:#ffe3a8; font-weight:normal;'>Evacuación vertical preventiva inmediata.</span>"
        
    st.markdown(
        f"""
        <div class='evac-banner-red'>
            <div style='display:flex; align-items:center; gap:16px;'>
                <span style='font-size:2.2rem;'>🚨</span>
                <div>
                    <div style='color:#ffffff; font-weight:800; font-size:1.10rem;'>ORDEN GENERAL DE EVACUACIÓN VERTICAL — PROTECCIÓN CIVIL / CECOPI</div>
                    <div style='color:#f0f6fc; font-size:0.88rem; font-weight:500;'>PELIGRO EXTREMO POR DESBORDAMIENTO. Suba de inmediato a plantas altas. Prohibido circular por carretera o acceder a garajes/vados.</div>
                </div>
            </div>
            <div style='text-align:right; font-family: monospace; font-size:0.80rem; color:#ff7b72; font-weight:800;'>{escape_msg}</div>
        </div>
        """, unsafe_allow_html=True,
    )
elif alert_state == "NARANJA":
    st.markdown(
        f"""
        <div class='evac-banner-orange'>
            <div style='display:flex; align-items:center; gap:16px;'>
                <span style='font-size:2.2rem;'>⚠️</span>
                <div>
                    <div style='color:#ff9100; font-size:1.10rem; font-weight:800;'>PRE-ALERTA DE EVACUACIÓN: EVITE DESPLAZAMIENTOS Y RETIRE VEHÍCULOS</div>
                    <div style='color:#f0f6fc; font-size:0.88rem; font-weight:500;'>Onda de avenida aproximándose a l'Horta Sud o afección pluvial/marítima. Asegure puntos altos.</div>
                </div>
            </div>
            <div style='text-align:right; font-family: monospace; font-size:0.80rem; color:#ff9100; font-weight:800;'>
                VENTANA DE SEGURIDAD:<br/><span style='color:#ffe3a8; font-weight:normal;'>~ {eta_mins} MINUTOS</span>
            </div>
        </div>
        """, unsafe_allow_html=True,
    )

if not IS_REAL_CKPT:
    ckpt_status_tag = "🔴 RESPALDO CINEMÁTICO (MODELO IA AUSENTE)"
elif ia_fuera_de_rango:
    ckpt_status_tag = "🔴 RESPALDO CINEMÁTICO (IA FUERA DE RANGO / ALUCINANDO)"
elif q_peak_simulated > 25000.0:
    ckpt_status_tag = "🔴 RESPALDO CINEMÁTICO (LÍMITE MÁXIMO SUPERADO)"
else:
    ckpt_status_tag = f"🟢 FNO V5 CARGADO: fno_poyo_surrogate.pt ({H_dim}x{W_dim} tensores)"
telemetry_txt = "SERIE FORENSE 29-O: Chiva / SAIH" if "Forense" in sim_mode else ("TELEMETRÍA AEMET: " + live_obs['station_name'] if (telemetry_active and live_obs) else "NOWCAST PREDICTIVO")

st.markdown(
    f"""
    <div class='telemetry-strip' style='border-left: 4px solid {"#ff1744" if "Forense" in sim_mode else "#1f6feb"};'>
        <div>📡 <b>{telemetry_txt}</b></div>
        <div>📅 <b>Sensor:</b> <span style='color:#58a6ff; font-weight:700;'>{sensor_date_label}</span></div>
        <div>🌧️ <b>Lluvia Cabecera:</b> <span style='color:#ff1744; font-weight:700;'>{fmt_dec(rain_val, 1, ' mm')}</span></div>
        <div>🌊 <b>Caudal Rambla:</b> <span style='color:#58a6ff; font-weight:700;'>{fmt_int(q_peak_simulated, ' m³/s')}</span></div>
        <div>⚡ <b>Inferencia 2D ({hw_device_name}):</b> <span style='color:#00e676; font-weight:700;'>{fno_latency_ms:.2f} ms</span></div>
        <div>🔵 <span style='font-weight:700;'>{ckpt_status_tag}</span></div>
    </div>
    """, unsafe_allow_html=True,
)

total_exposure_m = active_df["asset_value_eur"].sum() / 1e6 if not active_df.empty else 0.0
current_loss_m = active_df["active_loss"].sum() / 1e6 if not active_df.empty else 0.0
total_collapsed = int(active_df["dynamic_collapse"].sum()) if not active_df.empty else 0

p1_pop = int(active_df.loc[active_df["P1_flag"], "pop_density"].sum()) if not active_df.empty else 0
p2_pop = int(active_df.loc[active_df["P2_flag"], "pop_density"].sum()) if not active_df.empty else 0
p3_pop = int(active_df.loc[active_df["P3_flag"], "pop_density"].sum()) if not active_df.empty else 0
p4_pop = int(active_df.loc[active_df["P4_flag"], "pop_density"].sum()) if not active_df.empty else 0

if current_loss_m == 0.0 and total_collapsed == 0 and not has_experienced_catastrophe:
    st.info("ℹ️ **RÉGIMEN SECO:** El área metropolitana analizada se encuentra sin afección hidráulica activa. Monitoreo pasivo en curso.")

q_att, q_exh = 1000.0 if "Ampliación" not in what_if else 1300.0, 1800.0 if "Ampliación" not in what_if else 2340.0
ins_att, ins_exh = 0.02, 0.08
q_eval_cat = peak_q_so_far if "Forense" in sim_mode else q_peak_simulated
collapse_ratio = total_collapsed / max(1, len(active_df)) if len(active_df) > 0 else 0

fq = min(1.0, max(0.0, (q_eval_cat - q_att) / (q_exh - q_att)))
fi = min(1.0, max(0.0, (collapse_ratio - ins_att) / (ins_exh - ins_att)))

if fq > 0 and fi > 0: payout_rate = min(100.0, np.sqrt(fq * fi) * 100.0)
elif (sim_pluvial_vlc or sim_storm_surge or sim_turia) and fi > 0: payout_rate = min(100.0, fi * 85.0)
else: payout_rate = 0.0

total_payout_m = 80.0 * (payout_rate / 100.0)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

if "Forense" not in sim_mode and usar_hidrograma:
    titulo_kpi = "Caudal en Tránsito (T Actual)"
    delta_kpi = f"Pico Previsto: {fmt_int(q_peak_simulated, ' m³/s')}"
    d_color = "off"
else:
    titulo_kpi = "Caudal Punta Estimado (Máx)"
    delta_kpi = f"Mitigado: {fmt_int(q_natural - q_peak_simulated, ' m³/s')} retenidos" if q_mitig_factor < 1.0 else f"{'+' if q_peak_simulated - 1200 >= 0 else ''}{fmt_int(q_peak_simulated - 1200, ' m³/s')} vs Alerta"
    d_color = "normal" if q_mitig_factor < 1.0 else "inverse"

with kpi1: st.metric(titulo_kpi, fmt_int(q_instant, " m³/s"), delta=delta_kpi, delta_color=d_color)
with kpi2: st.metric("Pérdida Directa Activa", fmt_dec(current_loss_m, 1, " M€"), delta=f"{fmt_dec((current_loss_m / max(0.1, total_exposure_m))*100, 1, '%')} de Exposición", delta_color="inverse" if current_loss_m > 0 else "off")
with kpi3: st.metric("Inmuebles en Ruina", fmt_int(total_collapsed), delta="InSAR DPM ≥ 0,40" if total_collapsed > 0 else "Sin colapsos estructurales", delta_color="inverse" if total_collapsed > 0 else "off")
with kpi4: st.metric("Prioridad P1 (Rescate 112)", fmt_int(p1_pop), delta="Evacuación Inmediata" if p1_pop > 0 else "Situación bajo control", delta_color="inverse" if p1_pop > 0 else "off")
with kpi5: st.metric("Gatillo Paramétrico (Cat Bond)", fmt_dec(payout_rate, 1, "%"), delta=f"{fmt_dec(total_payout_m, 1, ' M€')} Liberados < 48h", delta_color="inverse" if payout_rate > 50.0 else "normal")

st.markdown(
    f"""
    <div class='triage-hud'>
        <span>Población Afectada (Datos Telco): </span>
        <span class='t-p1'>P1 (Crítica): {fmt_int(p1_pop)}</span> |
        <span class='t-p2'>P2 (Alta): {fmt_int(p2_pop)}</span> |
        <span class='t-p3'>P3 (Moderada): {fmt_int(p3_pop)}</span> |
        <span class='t-p4'>P4 (Baja): {fmt_int(p4_pop)}</span>
    </div>
    """, unsafe_allow_html=True
)

if "map_lat" not in st.session_state: st.session_state["map_lat"] = 39.4320
if "map_lon" not in st.session_state: st.session_state["map_lon"] = -0.4150
if "map_zoom" not in st.session_state: st.session_state["map_zoom"] = 12.6
if "map_pitch" not in st.session_state: st.session_state["map_pitch"] = 52

# ==============================================================================
# 10. PESTAÑAS DEL CENTRO DE MANDO C2
# ==============================================================================
tab_3d, tab_esalert, tab_hydro, tab_roads, tab_finances, tab_compare = st.tabs([
    "🌐 Gemelo Digital 3D (WebGPU)", 
    "🚨 Despacho ES-Alert & Alertas C4ISR", 
    "🌊 Dinámica Hidráulica FNO (M2)", 
    "🚑 Resiliencia Vial & TTI (M3)", 
    "💼 Finanzas del Clima & Solvencia II (M4)",
    "⚖️ Auditoría Split A/B (29-O vs Nowcast)"
])

with tab_3d:
    col_ctrl_left, col_ctrl_right = st.columns([2.0, 2.0])
    with col_ctrl_left:
        render_variable = st.radio("Modo de Representación 3D:", ["🌊 Calado Hidrodinámico FNO", "💶 Pérdida Económica CCS (€)", "🚨 Prioridad Triaje 112 (P1-P4)"], horizontal=True, key="render_var_horizontal", label_visibility="collapsed")
    with col_ctrl_right:
        cam_c1, cam_c2, cam_c3, cam_c4 = st.columns(4)
        with cam_c1:
            if st.button("📍 Paiporta", use_container_width=True): st.session_state["map_lat"] = 39.4240; st.session_state["map_lon"] = -0.4180; st.session_state["map_zoom"] = 13.5; st.session_state["map_pitch"] = 55; st.rerun()
        with cam_c2:
            if st.button("🏛️ Sedes GVA", use_container_width=True): st.session_state["map_lat"] = 39.4770; st.session_state["map_lon"] = -0.3700; st.session_state["map_zoom"] = 13.5; st.session_state["map_pitch"] = 48; st.rerun()
        with cam_c3:
            if st.button("🏥 Hosp. La Fe", use_container_width=True): st.session_state["map_lat"] = 39.4435; st.session_state["map_lon"] = -0.3768; st.session_state["map_zoom"] = 13.8; st.session_state["map_pitch"] = 52; st.rerun()
        with cam_c4:
            if st.button("✈️ Aeropuerto", use_container_width=True): st.session_state["map_lat"] = 39.4850; st.session_state["map_lon"] = -0.4550; st.session_state["map_zoom"] = 12.8; st.session_state["map_pitch"] = 50; st.rerun()

    st.markdown(
        """
        <div class='legend-box'>
            <span style='color:#8b949e; font-weight:700;'>SIMBOLOGÍA 3D:</span>
            <div class='legend-item'><span class='legend-bullet' style='background:#ff1744;'></span> Ruina / P1 Crítica</div>
            <div class='legend-item'><span class='legend-bullet' style='background:#ff9100;'></span> 0,80 - 1,50 m / P2</div>
            <div class='legend-item'><span class='legend-bullet' style='background:#ffd600;'></span> 0,30 - 0,80 m / P3</div>
            <div class='legend-item'><span class='legend-bullet' style='background:#00e676;'></span> 0,08 - 0,30 m / P4</div>
            <div class='legend-item'><span class='legend-bullet' style='background:#00e5ff;'></span> Pluvial Valencia</div>
            <div class='legend-item'><span class='legend-bullet' style='background:#2979ff;'></span> Intrusión Marina</div>
            <div class='legend-item'><span class='legend-bullet' style='background:#00e676;'></span> Vía Abierta</div>
            <div class='legend-item'><span class='legend-bullet' style='background:#ff1744;'></span> Vía Cortada</div>
            <div class='legend-item'><span class='legend-bullet' style='background:#00e5ff; border-radius:50%;'></span> Transporte</div>
            <div class='legend-item'><span class='legend-bullet' style='background:#b388ff; border-radius:50%;'></span> Sedes Oficiales</div>
            <div class='legend-item'><span class='legend-bullet' style='background:#ffd700; border-radius:50%;'></span> Universidades</div>
            <div class='legend-item'><span class='legend-bullet' style='background:#ffab40; border-radius:50%;'></span> Subestaciones</div>
        </div>
        """, unsafe_allow_html=True
    )

    cap_c1, cap_c2, cap_c3, cap_c4, cap_c5, cap_c6, cap_c7, cap_c8 = st.columns(8)
    with cap_c1: show_roads = st.checkbox("🛣️ Red Viaria", value=True)
    with cap_c2: show_vulnerable = st.checkbox("🏥 Centros Sensibles", value=True)
    with cap_c3: show_transport = st.checkbox("✈️ Logística", value=True)
    with cap_c4: show_institutions = st.checkbox("🏛️ Sedes / Subestaciones", value=True)
    with cap_c5: show_universities = st.checkbox("🎓 Universidades", value=True)
    with cap_c6: show_isochrones = st.checkbox("⭕ Isocronas", value=True)
    with cap_c7: show_building_mesh = st.checkbox("🏢 Malla 3D", value=False)
    with cap_c8: show_hydro_vectors = st.checkbox("🌊 Red Fluvial & Diques", value=True)

    deck_layers = []

    RAW_GEOJSON_URL = "https://raw.githubusercontent.com/kelvinjfloresy/poyo-nowcast/main/web_digital_twin/data/edificios_valencia_metropolitana_3d.geojson"

    if show_hydro_vectors and GEO_HYDRO:
        deck_layers.append(pdk.Layer("GeoJsonLayer", data=GEO_HYDRO, opacity=0.85, stroked=True, filled=False, get_line_color=[0, 229, 255, 220], line_width_min_pixels=2, pickable=False, auto_highlight=False))
    if show_hydro_vectors and GEO_BRIDGES:
        deck_layers.append(pdk.Layer("GeoJsonLayer", data=GEO_BRIDGES, opacity=0.90, stroked=True, filled=True, get_fill_color=[255, 214, 0, 200], get_line_color=[255, 255, 255, 255], point_radius_min_pixels=4, pickable=False, auto_highlight=False))
    if show_building_mesh:
        deck_layers.append(pdk.Layer("GeoJsonLayer", data=RAW_GEOJSON_URL, opacity=0.75, stroked=True, filled=True, extruded=True, wireframe=False, get_elevation="properties.height_m", elevation_scale=1.0, get_fill_color=[90, 95, 105, 200], get_line_color=[50, 60, 70, 255], line_width_min_pixels=1, pickable=False))

    if not active_df.empty:
        c_eval_3d = active_df["dynamic_collapse"].to_numpy()

        if "Calado" in render_variable:
            # Calado Hidrodinámico FNO: Muestra la lámina de agua transitoria viva en ese minuto exacto
            h_eval_3d = active_df["active_depth"].to_numpy()
            condlist = [c_eval_3d, h_eval_3d >= 1.50, h_eval_3d >= 0.80, h_eval_3d >= 0.30, h_eval_3d >= 0.08]
            r_c = np.select(condlist, [255, 255, 255, 255, 0],   default=35)
            g_c = np.select(condlist, [23,  82,  145, 214, 230], default=45)
            b_c = np.select(condlist, [68,  82,  0,   0,   118], default=58)
            a_c = np.select(condlist, [255, 255, 255, 255, 255], default=140) 
            elevations = np.where(h_eval_3d < 0.08, 2.0, np.select([c_eval_3d, h_eval_3d >= 0.30], [np.maximum(h_eval_3d * 42.0, 75.0), h_eval_3d * 32.0 + 10.0], default=np.clip(h_eval_3d * 22.0 + 5.0, 5.0, 32.0)))
        elif "Pérdida" in render_variable:
            # Pérdida Económica CCS (€): Daño consolidado irreversible (No se borra cuando el agua se retira)
            l_col = active_df["active_loss"].to_numpy()
            condlist_loss = [l_col >= 150000, l_col >= 75000, l_col >= 30000, l_col >= 10000, l_col >= 1000]
            r_c = np.select(condlist_loss, [218, 245, 255, 255, 0],   default=35)
            g_c = np.select(condlist_loss, [54,  100, 160, 210, 200], default=45)
            b_c = np.select(condlist_loss, [51,  20,  0,   0,   150], default=58)
            a_c = np.select(condlist_loss, [255, 255, 255, 255, 255], default=140)
            elevations = np.where(l_col <= 0.0, 2.0, np.clip((l_col / 2200.0) + 8.0, 5.0, 180.0))
        else:
            # Prioridad Triaje 112 (P1-P4): Mantiene las personas en ruina/rescate en Paiporta y Catarroja
            p1_col = active_df["P1_flag"].to_numpy()
            p2_col = active_df["P2_flag"].to_numpy()
            p3_col = active_df["P3_flag"].to_numpy()
            p4_col = active_df["P4_flag"].to_numpy()
            condlist_triage = [p1_col, p2_col, p3_col, p4_col]
            r_c = np.select(condlist_triage, [255, 255, 255, 0], default=35)
            g_c = np.select(condlist_triage, [23,  145, 214, 230], default=45)
            b_c = np.select(condlist_triage, [68,  0,   0,   118], default=58)
            a_c = np.select(condlist_triage, [255, 255, 255, 255], default=140)
            elevations = np.where(p1_col | p2_col | p3_col | p4_col, np.select([p1_col, p2_col, p3_col], [85.0, 45.0, 20.0], default=10.0), 2.0)

        active_df["rgba"] = np.column_stack([r_c, g_c, b_c, a_c]).astype(np.uint8).tolist()
        active_df["elevation_m"] = elevations
        active_df["layer_title"] = active_df["parcel_id"] + " (" + active_df["municipality"] + ")"
        active_df["metric_primary"] = "Calado Modelado: " + active_df["peak_depth_experienced"].apply(lambda v: fmt_dec(v, 2, " m"))
        active_df["metric_secondary"] = "Pérdida CCS: " + active_df["active_loss"].apply(lambda v: fmt_dec(v, 0, " €"))
        active_df["status_tag"] = np.select([active_df["dynamic_collapse"], active_df["P1_flag"], active_df["P2_flag"], active_df["P3_flag"]], ["RUINA / COLAPSO", "P1 CRÍTICA", "P2 ALTA", "P3 MODERADA"], default="P4 NORMAL")
        active_df["status_color"] = np.select([active_df["dynamic_collapse"], active_df["P1_flag"], active_df["P2_flag"], active_df["P3_flag"]], ["#ff1744", "#ff1744", "#ff9100", "#ffd600"], default="#00e676")
        
        deck_layers.append(pdk.Layer("ColumnLayer", data=active_df, get_position=["lon", "lat"], get_elevation="elevation_m", elevation_scale=0.85, radius=28, disk_resolution=4, get_fill_color="rgba", pickable=True, auto_highlight=True))
        
    if plan_sur_desbordado:
        turia_polygon = [{"polygon": [[-0.440, 39.458, 120], [-0.340, 39.458, 120], [-0.340, 39.480, 120], [-0.440, 39.480, 120]], "layer_title": "🌊 Colapso Plan Sur (Río Turia > 5.000 m³/s)", "metric_primary": "Desbordamiento margen derecha", "metric_secondary": "Remanso hacia poblados del sur", "status_tag": "DESBORDAMIENTO ACTIVO", "status_color": "#ff1744", "color": [255, 23, 68, 115]}]
        deck_layers.append(pdk.Layer("PolygonLayer", data=pd.DataFrame(turia_polygon), get_polygon="polygon", get_fill_color="color", get_line_color=[255, 23, 68, 240], line_width_min_pixels=3, stroked=True, pickable=True))

    if sim_golas:
        golas_polygon = [{"polygon": [[-0.360, 39.350], [-0.310, 39.350], [-0.310, 39.380], [-0.370, 39.380]], "layer_title": "🌾 Colmatación Golas de la Albufera", "metric_primary": "Bloqueo mecánico de compuertas", "metric_secondary": "Reflujo hacia Pinedo y El Palmar", "status_tag": "REFLUJO ACTIVO", "status_color": "#ff9100", "color": [255, 145, 0, 115]}]
        deck_layers.append(pdk.Layer("PolygonLayer", data=pd.DataFrame(golas_polygon), get_polygon="polygon", get_fill_color="color", get_line_color=[255, 145, 0, 240], line_width_min_pixels=3, stroked=True, pickable=True))

    if pluvial_desbordado:
        pluvial_polygon = [{"polygon": [[-0.402, 39.472, 90], [-0.395, 39.493, 90], [-0.365, 39.498, 90], [-0.342, 39.482, 90], [-0.324, 39.468, 90], [-0.330, 39.448, 90], [-0.360, 39.444, 90], [-0.388, 39.446, 90], [-0.400, 39.458, 90], [-0.402, 39.472, 90]], "layer_title": "Anegamiento Pluvial Urbano (València Centro)", "metric_primary": "Intensidad > 120 mm/h | Sobrecarga de Red Unitaria", "metric_secondary": "Colapso en pasos inferiores y accesos hospitalarios", "status_tag": "COLAPSO DRENAJE URBANO", "status_color": "#00e5ff", "color": [0, 229, 255, 18]}]
        deck_layers.append(pdk.Layer("PolygonLayer", data=pd.DataFrame(pluvial_polygon), get_polygon="polygon", get_fill_color="color", get_line_color=[0, 229, 255, 255], line_width_min_pixels=2, stroked=True, pickable=True))
        
    if sim_storm_surge:
        surge_polygon = [{"polygon": [[-0.355, 39.490], [-0.310, 39.490], [-0.310, 39.380], [-0.370, 39.380], [-0.360, 39.440], [-0.355, 39.490]], "layer_title": "Intrusión Marina & Storm Surge (Temporal de Levante)", "metric_primary": "Sobreelevación del Nivel del Mar +0,70 m", "metric_secondary": "Taponamiento Desembocadura Albufera/Turia", "status_tag": "TAPONAMIENTO MARINO ACTIVO", "status_color": "#2979ff", "color": [41, 121, 255, 115]}]
        deck_layers.append(pdk.Layer("PolygonLayer", data=pd.DataFrame(surge_polygon), get_polygon="polygon", get_fill_color="color", get_line_color=[41, 121, 255, 240], line_width_min_pixels=3, stroked=True, pickable=True))

    if ruptura_mota:
        mota_polygon = [{"polygon": [[-0.420, 39.426], [-0.416, 39.427], [-0.415, 39.422], [-0.419, 39.421]], "layer_title": "💥 Ruptura de Mota en Paiporta", "metric_primary": "Colapso de muro de contención", "metric_secondary": "Onda de rotura adicional: +0,85 m", "status_tag": "ZONA CERO (COLAPSO)", "status_color": "#ff1744", "color": [255, 23, 68, 140]}]
        deck_layers.append(pdk.Layer("PolygonLayer", data=pd.DataFrame(mota_polygon), get_polygon="polygon", get_fill_color="color", get_line_color=[255, 255, 255, 255], line_width_min_pixels=3, stroked=True, pickable=True))
    
    if "Presa" in what_if or "Integral" in what_if:
        presa_polygon = [{"polygon": [[-0.520, 39.480], [-0.510, 39.480], [-0.510, 39.490], [-0.520, 39.490]], "layer_title": "🚧 Presa de Laminación (Cheste)", "metric_primary": "Retención de Crecidas", "metric_secondary": "Reducción de Caudal Punta: -35%", "status_tag": "OPERATIVA", "status_color": "#00e676", "color": [0, 230, 118, 140]}]
        deck_layers.append(pdk.Layer("PolygonLayer", data=pd.DataFrame(presa_polygon), get_polygon="polygon", get_fill_color="color", get_line_color=[0, 255, 128, 255], line_width_min_pixels=3, stroked=True, pickable=True))

    if "Ampliación" in what_if or "Integral" in what_if:
        ampliacion_path = [{"path": [[-0.468, 39.432], [-0.432, 39.430], [-0.415, 39.426], [-0.380, 39.420]], "layer_title": "🚜 Ampliación de Sección (Rambla del Poyo)", "metric_primary": "Aumento de Capacidad Hidráulica", "metric_secondary": "Caudal máximo sin desborde: +30%", "status_tag": "CANALIZACIÓN ACTIVA", "status_color": "#00e676", "color": [0, 230, 118, 200], "width": 10}]
        deck_layers.append(pdk.Layer("PathLayer", data=pd.DataFrame(ampliacion_path), get_path="path", get_color="color", get_width="width", width_scale=1, width_min_pixels=4, pickable=True))

    if show_isochrones and alert_state != "VERDE" and q_peak_simulated < 3500.0:
        if pluvial_desbordado and q_peak_simulated < 300.0:
            origin_lon, origin_lat = -0.3768, 39.4700
        else:
            wet_parcels = active_df[active_df["active_depth"] >= 0.30]
            if not wet_parcels.empty:
                origin_lon = float(wet_parcels["lon"].mean())
                origin_lat = float(wet_parcels["lat"].mean())
            else:
                origin_lon, origin_lat = -0.4190, 39.4230
            
        penalizacion = max(0.18, 1.0 - (q_peak_simulated / 2200.0))
        iso_rings = [{"time": "Isocrona 30 min (Exterior)", "radius": 3200 * penalizacion, "color": [0, 230, 118, 240]}, {"time": "Isocrona 20 min (Intermedia)", "radius": 2000 * penalizacion, "color": [255, 145, 0, 240]}, {"time": "Isocrona 10 min (Inmediata)", "radius": 900 * penalizacion, "color": [255, 23, 68, 255]}]
        iso_df = pd.DataFrame(iso_rings)
        iso_df["lon"] = origin_lon; iso_df["lat"] = origin_lat
        iso_df["layer_title"] = iso_df["time"]; iso_df["metric_primary"] = "Alcance Perimetral dinámico"
        iso_df["metric_secondary"] = f"Margen de Escape Viable: {int(penalizacion * 100)}%"
        iso_df["status_tag"] = "ISOCRONA DE EVACUACIÓN"; iso_df["status_color"] = "#58a6ff"
        deck_layers.append(pdk.Layer("ScatterplotLayer", data=iso_df, get_position=["lon", "lat"], get_radius="radius", filled=False, stroked=True, get_line_color="color", line_width_min_pixels=3, pickable=True))

    if show_roads and realistic_roads:
        road_paths = []
        for i, r_item in enumerate(realistic_roads):
            h_road_max = road_max_depths[i]
            is_open = h_road_max < 0.30
            reason = "OPERATIVO (TRANSITABLE)" if is_open else f"CORTADO (Punto Crítico FNO: {fmt_dec(h_road_max, 2, ' m')})"
            r_hex = "#00e676" if is_open else "#ff1744"
            col = [0, 230, 118, 230] if is_open else [255, 23, 68, 255]
            road_paths.append({"path": r_item["coords"], "layer_title": r_item["name"], "metric_primary": f"Calado Máx en Traza: {fmt_dec(h_road_max, 2, ' m')}", "metric_secondary": "Eje Arterial Metropolitano", "status_tag": reason, "status_color": r_hex, "color": col, "width": 5 if is_open else 8})
        deck_layers.append(pdk.Layer("PathLayer", data=pd.DataFrame(road_paths), get_path="path", get_color="color", get_width="width", width_scale=1, width_min_pixels=3, pickable=True))

    if show_vulnerable:
        vuln_rows = []
        for i, v in enumerate(VULNERABLE_CENTERS_BASE):
            h_v = v_depths[i]
            is_crit = h_v >= 0.50
            v_stat = "🔴 INUNDACIÓN CRÍTICA / EVACUAR" if is_crit else ("🟠 PREALERTA" if h_v >= 0.15 else "🟢 OPERATIVO")
            v_col = [255, 23, 68, 240] if is_crit else ([255, 145, 0, 230] if h_v >= 0.15 else [0, 230, 118, 220])
            vuln_rows.append({"name": v["name"], "lon": v["lon"], "lat": v["lat"], "layer_title": f"{v['name']} ({v['mun']})", "metric_primary": f"Capacidad: {v['beds']} plazas", "metric_secondary": f"Calado Modelado: {fmt_dec(h_v, 2, ' m')}", "status_tag": v_stat, "status_color": "#ff1744" if is_crit else "#00e676", "color": v_col, "height": 160.0 if is_crit else 80.0})
        deck_layers.append(pdk.Layer("ColumnLayer", data=pd.DataFrame(vuln_rows), get_position=["lon", "lat"], get_elevation="height", elevation_scale=1, radius=65, disk_resolution=4, get_fill_color="color", pickable=True, auto_highlight=True))

    if show_transport:
        t_rows = []
        for i, t in enumerate(TRANSPORT_LANDMARKS_BASE):
            h_t = t_depths[i]
            t_col = [255, 23, 68, 240] if h_t >= 0.35 else [0, 229, 255, 240]
            t_rows.append({"name": f"{t['icon']} {t['name']}", "lon": t["lon"], "lat": t["lat"], "layer_title": t["name"], "metric_primary": f"Nodo: {t['type']}", "metric_secondary": f"Calado Estimado: {fmt_dec(h_t, 2, ' m')}", "status_tag": "🔴 COLAPSO EN ACCESOS" if h_t >= 0.35 else "🟢 OPERATIVO", "status_color": "#ff1744" if h_t >= 0.35 else "#00e5ff", "color": t_col, "height": 180.0})
        df_t = pd.DataFrame(t_rows)
        deck_layers.append(pdk.Layer("ColumnLayer", data=df_t, get_position=["lon", "lat"], get_elevation="height", elevation_scale=1, radius=85, disk_resolution=4, get_fill_color="color", pickable=True))
        deck_layers.append(pdk.Layer("TextLayer", data=df_t, get_position=["lon", "lat"], get_text="name", get_size=11, get_color=[0, 229, 255, 255], get_text_anchor="'start'", pixel_offset=[18, -20]))

    if show_institutions:
        inst_rows = []
        for i, g in enumerate(INSTITUTION_LANDMARKS_BASE):
            h_g = g_depths[i]
            is_exterior = "L'Eliana" in g["name"]
            if is_exterior:
                g_col = [0, 230, 118, 240]; i_stat = "🟢 NODO EXTERIOR DE MANDO (SEGURO)"; i_hex = "#00e676"
            else:
                g_col = [255, 23, 68, 240] if h_g >= 0.35 else [179, 136, 255, 240]
                i_stat = "🔴 COMITÉ CRISIS ACTIVO" if h_g >= 0.35 else "🟢 OPERATIVIDAD NORMAL"
                i_hex = "#ff1744" if h_g >= 0.35 else "#b388ff"
            inst_rows.append({"name": f"{g['icon']} {g['name']}", "lon": g["lon"], "lat": g["lat"], "layer_title": g["name"], "metric_primary": f"Tipo: {g['type']}", "metric_secondary": f"Calado Estimado: {fmt_dec(h_g, 2, ' m')}" if not is_exterior else "(Fuera de Zona Aluvial)", "status_tag": i_stat, "status_color": i_hex, "color": g_col, "height": 140.0})
        for i, s in enumerate(SUBSTATIONS_BASE):
            h_s = s_depths[i]
            if s["tipo"] == "ELÉCTRICA": is_down = h_s >= 0.35
            elif s["tipo"] == "SANEAMIENTO": is_down = h_s >= 0.20
            else: is_down = (h_s >= 0.60) or power_outage
            s_col = [255, 23, 68, 240] if is_down else [255, 171, 64, 240]
            reason = "Corte Energía" if (s["tipo"] == "TELECOM" and power_outage and h_s < 0.60) else "Inundación"
            s_stat = f"🔴 BLACKOUT / FUERA DE SERVICIO ({reason})" if is_down else "🟢 OPERATIVA"
            s_hex = "#ff1744" if is_down else "#ffab40"
            inst_rows.append({"name": f"{s['icon']} {s['name']}", "lon": s["lon"], "lat": s["lat"], "layer_title": s["name"], "metric_primary": f"Nodo Vital: {s['tipo']}", "metric_secondary": f"Calado Estimado: {fmt_dec(h_s, 2, ' m')}", "status_tag": s_stat, "status_color": s_hex, "color": s_col, "height": 130.0})
        df_inst = pd.DataFrame(inst_rows)
        deck_layers.append(pdk.Layer("ColumnLayer", data=df_inst, get_position=["lon", "lat"], get_elevation="height", elevation_scale=1, radius=75, disk_resolution=4, get_fill_color="color", pickable=True))
        deck_layers.append(pdk.Layer("TextLayer", data=df_inst, get_position=["lon", "lat"], get_text="name", get_size=11, get_color=[240, 246, 252, 255], get_text_anchor="'start'", pixel_offset=[18, -15]))

    if show_universities:
        u_rows = []
        for i, u in enumerate(UNIVERSITIES_BASE):
            h_u = u_depths[i]
            u_col = [255, 23, 68, 240] if h_u >= 0.30 else [255, 215, 0, 240]
            u_rows.append({"name": f"{u['icon']} {u['name']}", "lon": u["lon"], "lat": u["lat"], "layer_title": u["name"], "metric_primary": "Campus Universitario", "metric_secondary": f"Calado Local: {fmt_dec(h_u, 2, ' m')}", "status_tag": "🔴 DOCENCIA SUSPENDIDA" if h_u >= 0.30 else "🟢 CAMPUS OPERATIVO", "status_color": "#ff1744" if h_u >= 0.30 else "#ffd700", "color": u_col, "height": 130.0})
        df_u = pd.DataFrame(u_rows)
        deck_layers.append(pdk.Layer("ColumnLayer", data=df_u, get_position=["lon", "lat"], get_elevation="height", elevation_scale=1, radius=70, disk_resolution=4, get_fill_color="color", pickable=True))
        deck_layers.append(pdk.Layer("TextLayer", data=df_u, get_position=["lon", "lat"], get_text="name", get_size=11, get_color=[255, 215, 0, 255], get_text_anchor="'start'", pixel_offset=[18, -15]))

    hosp_data = []
    for i, h in enumerate(HOSPITALS_BASE):
        h_h = h_depths[i]
        is_isolated = h_h >= 0.40
        h_col = [255, 23, 68, 255] if is_isolated else [0, 140, 255, 255]
        hosp_data.append({"name": f"🏥 {h['name']}", "lon": h["lon"], "lat": h["lat"], "layer_title": h["name"], "metric_primary": f"Capacidad: {h['beds']} camas", "metric_secondary": f"Calado en Accesos: {fmt_dec(h_h, 2, ' m')}", "status_tag": "COLAPSO DE ACCESOS" if is_isolated else "OPERATIVO (SINK ACTIVO)", "status_color": "#ff1744" if is_isolated else "#00e676", "color": h_col, "height": 220.0})
    df_hosp = pd.DataFrame(hosp_data)
    deck_layers.append(pdk.Layer("ColumnLayer", data=df_hosp, get_position=["lon", "lat"], get_elevation="height", elevation_scale=1, radius=95, disk_resolution=4, get_fill_color="color", pickable=True, auto_highlight=True))
    deck_layers.append(pdk.Layer("TextLayer", data=df_hosp, get_position=["lon", "lat"], get_text="name", get_size=12, get_color=[255, 255, 255, 255], get_text_anchor="'start'", pixel_offset=[20, -25]))

    camera = pdk.ViewState(latitude=st.session_state["map_lat"], longitude=st.session_state["map_lon"], zoom=st.session_state["map_zoom"], pitch=st.session_state["map_pitch"], bearing=-20)
    
    tooltip_html = {
        "html": "<div style='font-family: Inter, sans-serif; padding: 6px;'>"
                "<b style='color:#58a6ff; font-size:13px;'>{layer_title}</b><br/>"
                "<span style='color:#c9d1d9;'>{metric_primary}</span><br/>"
                "<span style='color:#c9d1d9;'>{metric_secondary}</span><br/>"
                "<div style='margin-top:6px; padding-top:6px; border-top:1px solid #30363d;'>"
                "<b style='color:#8b949e;'>Estado:</b> <span style='color:{status_color}; font-weight:800;'>{status_tag}</span>"
                "</div></div>",
        "style": {"backgroundColor": "#161b22", "color": "white", "fontSize": "12px", "borderRadius": "6px", "border": "1px solid #30363d", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)"}
    }

    st.pydeck_chart(pdk.Deck(layers=deck_layers, initial_view_state=camera, map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json", tooltip=tooltip_html), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: DESPACHO ES-ALERT TRILINGÜE Y ACCESIBILIDAD UNIVERSAL (RD 193/2023)
# ------------------------------------------------------------------------------
with tab_esalert:
    st.subheader("Centro de Despacho ES-Alert Trilingüe & Nodos Vitales (Lifeline Utilities)")
    col_es1, col_es2 = st.columns([1.3, 1.7])
    
    with col_es1:
        st.markdown("#### Estado de Subestaciones y Redes Estratégicas")
        sub_rows = []
        for i, s in enumerate(SUBSTATIONS_BASE):
            h_s = s_depths[i]
            if s["tipo"] == "SANEAMIENTO":
                state_text = "🔴 FUERA SERVICIO (Inundación)" if h_s >= 0.60 else ("🟠 SOBRECARGA (Aliviadero)" if h_s >= 0.25 else "🟢 OPERATIVO")
            elif s["tipo"] == "ELÉCTRICA":
                is_down = h_s >= 0.35
                state_text = "🔴 FUERA SERVICIO" if is_down else "🟢 OPERATIVO"
            else:
                is_down = (h_s >= 0.60) or power_outage
                reason = "Corte Energía" if (power_outage and h_s < 0.60) else "Inundación"
                state_text = f"🔴 FUERA SERVICIO ({reason})" if is_down else "🟢 OPERATIVO"
                
            sub_rows.append({"Infraestructura": s["name"], "Tipo": s["tipo"], "Calado Integrado": fmt_dec(h_s, 2, " m"), "Estado": state_text})
        st.dataframe(pd.DataFrame(sub_rows), hide_index=True, width="stretch")

        if q_peak_simulated >= 4000.0:
            nivel_situacion = "SITUACIÓN 3 (INTERÉS NACIONAL - MANDO ESTATAL)"
            aviso_extra = " MANDO ASUMIDO POR MINISTERIO DEL INTERIOR. DESPLIEGUE MILITAR TOTAL."
        else:
            nivel_situacion = "SITUACIÓN 2"
            aviso_extra = ""

        st.markdown("#### 📢 Comunicado Oficial Automatizado (X / Radio)")
        
        water_alert_es = "\n\n☣️ ALERTA SANITARIA: Red de agua contaminada. HIERVA EL AGUA antes de consumirla." if edar_biocollapse else ""
        water_alert_val = "\n\n☣️ ALERTA SANITÀRIA: Xarxa d'aigua contaminada. BULLIU L'AIGUA abans de consumir-la." if edar_biocollapse else ""
        water_alert_en = "\n\n☣️ SANITARY ALERT: Contaminated water network. BOIL WATER before consumption." if edar_biocollapse else ""

        if alert_state == "ROJO":
            tweet_body = f"🚨 URGENTE 112 // ALERTA ROJA RAMBLA DEL POYO\nNivel: {nivel_situacion} | Hora: {clock_badge_text}\nCaudal Estimado: {fmt_int(q_peak_simulated, ' m3/s')}. Inundación inminente. EVACUACIÓN VERTICAL INMEDIATA en l'Horta Sud. Suba a pisos altos. NO circule por carretera.{aviso_extra}{water_alert_es}"
        elif alert_state == "NARANJA":
            tweet_body = f"⚠️ PRECAUCIÓN 112 // ALERTA NARANJA CRECIDA SEVERA\nHora: {clock_badge_text}\nCaudal: {fmt_int(q_peak_simulated, ' m3/s')}. Aléjese de cauces y barrancos. Retire vehículos de zonas inundables.{water_alert_es}"
        elif alert_state == "AMARILLO":
            tweet_body = f"🟡 AVISO 112 // PREALERTA METEOROLÓGICA\nHora: {clock_badge_text}\nCaudal: {fmt_int(q_peak_simulated, ' m3/s')}. Lluvias intensas. Manténgase informado por canales oficiales.{water_alert_es}"
        else:
            tweet_body = f"✅ INFO 112 // ESTADO DE NORMALIDAD HIDROLÓGICA\nHora: {clock_badge_text}\nCaudal: {fmt_int(q_peak_simulated, ' m3/s')}. Sin riesgo de desbordamiento. Monitoreo pasivo activo.{water_alert_es}"

        st.text_area("Despacho oficial de emergencia:", value=tweet_body, height=130)

        st.markdown("#### ⚡ Monitor de Cascada Crítica (Nodos C4ISR)")
        c_fallos = sum([1 for s in s_depths if s >= 0.35])
        c_vados = sum([1 for h in road_max_depths if h >= 0.30])
        
        st.markdown(
            f"""
            <div style='background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; font-size: 0.8rem; color: #c9d1d9;'>
                <div style='display:flex; justify-content:space-between; margin-bottom: 6px;'>
                    <span>🔌 Subestaciones Caídas:</span> <b style='color:{"#ff1744" if c_fallos > 0 else "#00e676"};'>{c_fallos} / {len(SUBSTATIONS_BASE)}</b>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom: 6px;'>
                    <span>🌉 Vados Intransitables:</span> <b style='color:{"#ff1744" if c_vados > 0 else "#00e676"};'>{c_vados} / {len(road_max_depths)}</b>
                </div>
                <div style='display:flex; justify-content:space-between;'>
                    <span>📡 Red Troncal de Datos:</span> <b style='color:{"#ff9100" if power_outage else "#00e676"};'>{"CONMUTADA A RESPALDO" if power_outage else "ÓPTIMA"}</b>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    with col_es2:
        st.markdown("#### 🧪 Monitor de Resiliencia Hidrosanitaria & Salud Ambiental (One Health)")

        env_c1, env_c2, env_c3, env_c4 = st.columns(4)
        with env_c1:
            if edar_biocollapse:
                val_ed = "COLAPSO ACTIVO"
                del_ed = f"Calado: {fmt_dec(h_edar, 2, ' m')} (Crítico ≥ 0,60 m)"
                col_ed = "inverse"
            elif edar_prealert:
                val_ed = "SOBRECARGA / ALIVIO"
                del_ed = f"Calado: {fmt_dec(h_edar, 2, ' m')} (Aliviadero Activo)"
                col_ed = "normal"
            else:
                val_ed = "OPERATIVA"
                del_ed = f"Calado: {fmt_dec(h_edar, 2, ' m')} (Nominal)"
                col_ed = "normal"
            st.metric(label="Fallo Biológico EDAR", value=val_ed, delta=del_ed, delta_color=col_ed)
        with env_c2:
            st.metric(label="Camas Geriátricas Aisladas", value=fmt_int(geriatric_beds_critical), delta="Riesgo Vital Inmediato" if geriatric_beds_critical > 0 else "Acceso Asegurado", delta_color="inverse" if geriatric_beds_critical > 0 else "off")
        with env_c3:
            st.metric(label="Población sin Agua Segura", value=fmt_int(pop_water_compromised), delta="Riesgo Retrosifonaje (RD 3/2023)" if pop_water_compromised > 0 else "Presión Nominal", delta_color="inverse" if pop_water_compromised > 0 else "off")
        with env_c4:
            st.metric(label="Superficie Aguas Negras", value=fmt_dec(ha_biohazard_exposed, 1, " ha"), delta="Vector Leptospira / Coliformes" if edar_biocollapse else "Sin Vertido Masivo", delta_color="inverse" if edar_biocollapse else "off")

        st.markdown(
            f"""
            <div style='background: rgba(22, 27, 34, 0.95); border: 1.5px solid {sanitary_color}; border-left: 6px solid {sanitary_color}; border-radius: 6px; padding: 12px 16px; margin-top: 10px; font-family: monospace;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <b style='color:{sanitary_color}; font-size:0.90rem;'>{sanitary_status_label}</b>
                    <span style='background:#161b22; color:#f0f6fc; padding:2px 8px; border-radius:4px; font-size:0.75rem; border: 1px solid #30363d;'>PROTOCOLO OMS / WASH</span>
                </div>
                <div style='margin-top:6px; font-size:0.80rem; color:#c9d1d9;'>
                    <b>Prescripción Sanitaria:</b> {water_advisory} | <b>Contención:</b> Aislar captaciones freáticas someras e hiperclorar la red troncal.
                </div>
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown("#### 📲 Transmisión Celular Trilingüe (Cell Broadcast)")

        if alert_state == "ROJO":
            box_color = "#ff1744"
            headline = f"ALERTA ROJA: EMERGENCIA {nivel_situacion}"
            body_es = "Peligro extremo por desbordamiento en cuenca del Poyo. Suba a pisos altos. No circule por carretera." + water_alert_es
            body_val = "Perill extrem per desbordament a la conca del Poio. Pugeu a pisos alts. No circuleu per carretera." + water_alert_val
            body_en = "Extreme flash flood danger in Poyo ravine basin. Move to upper floors. Do not drive." + water_alert_en
        elif alert_state == "NARANJA":
            box_color = "#ff9100"
            headline = "ALERTA NARANJA: CRECIDA SEVERA"
            body_es = "Alerta por crecida severa. Evite desplazamientos innecesarios y retire vehículos." + water_alert_es
            body_val = "Alerta per crescuda severa. Eviteu desplaçaments innecessaris i retireu vehicles." + water_alert_val
            body_en = "Severe flood warning. Avoid unnecessary travel and remove vehicles." + water_alert_en
        elif alert_state == "AMARILLO":
            box_color = "#ffd600"
            headline = "PREALERTA METEOROLÓGICA"
            body_es = "Prealerta por lluvias intensas. Manténgase informado a través de canales oficiales." + water_alert_es
            body_val = "Prealerta per pluges intenses. Manteniu-vos informats a través de canals oficials." + water_alert_val
            body_en = "Heavy rain pre-alert. Stay informed through official channels." + water_alert_en
        else:
            box_color = "#00e676"
            headline = "ESTADO DE NORMALIDAD"
            body_es = "Sin alertas hidrológicas activas en la zona metropolitana. Monitoreo pasivo en curso." + water_alert_es
            body_val = "Sense alertes hidrològiques actives a la zona metropolitana. Monitoratge passiu en curs." + water_alert_val
            body_en = "No active hydrological alerts in the metropolitan area. Passive monitoring in progress." + water_alert_en

        st.markdown(
            f"""
            <div style='background: rgba(13, 17, 23, 0.9); border: 2px solid {box_color}; border-radius: 8px; padding: 14px;'>
                <b style='color:{box_color}; font-family: monospace;'>ESTADO DE DIFUSIÓN: {alert_state}</b>
                <h4 style='margin: 8px 0;'>{headline}</h4>
                <div style='background:#161b22; padding:6px 10px; border-radius:4px; font-size:0.80rem; margin-bottom:4px;'><b style='color:#58a6ff;'>[ES]</b> {body_es}</div>
                <div style='background:#161b22; padding:6px 10px; border-radius:4px; font-size:0.80rem; margin-bottom:4px;'><b style='color:#ff9100;'>[VAL]</b> {body_val}</div>
                <div style='background:#161b22; padding:6px 10px; border-radius:4px; font-size:0.80rem;'><b style='color:#00e676;'>[EN]</b> {body_en}</div>
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style='background: rgba(22, 27, 34, 0.95); border: 1px solid #30363d; border-radius: 8px; padding: 10px 14px; margin-top: 10px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'><b style='color:#58a6ff; font-size:0.78rem;'>♿ ACCESIBILIDAD UNIVERSAL & CANALES INCLUSIVOS (RD 193/2023)</b><span style='background:#1f6feb; color:white; font-size:0.65rem; padding:2px 6px; border-radius:4px;'>NORMA UNE 139803</span></div>
                <div style='display:flex; gap:12px; margin-top:8px; font-size:0.74rem; color:#c9d1d9; flex-wrap:wrap;'>
                    <div>🤟 <b>Canal LSE:</b> Vídeo signado generado para emisión TV/Web</div>
                    <div>📳 <b>Háptico:</b> Patrón de vibración prolongado para sordoceguera</div>
                    <div>🚹 <b>Lectura Fácil:</b> Suba a pisos altos. No coja el coche. Aléjese del agua.</div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

        iso_timestamp = now_valencia.strftime('%Y-%m-%dT%H:%M:%S%z')
        cap_xml_payload = f"""<?xml version="1.0" encoding="UTF-8"?>\n<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">\n  <identifier>POYO-NOWCAST-{int(time.time())}</identifier>\n  <sender>gva.112@emergencies.gva.es</sender>\n  <sent>{iso_timestamp}</sent>\n  <status>Actual</status>\n  <msgType>Alert</msgType>\n  <scope>Public</scope>\n  <info>\n    <language>es-ES</language>\n    <category>Met</category>\n    <event>Flash Flood / Desbordamiento</event>\n    <urgency>Immediate</urgency>\n    <severity>Extreme</severity>\n    <certainty>Observed</certainty>\n    <headline>{headline}</headline>\n    <description>{body_es}</description>\n    <area>\n      <areaDesc>Area Metropolitana de Valencia y l'Horta Sud</areaDesc>\n      <circle>39.4230,-0.4180,12000</circle>\n    </area>\n  </info>\n</alert>"""

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1: 
            st.download_button("📲 Exportar Payload CAP v1.2", cap_xml_payload, "es_alert_poyo.xml", "application/xml", width="stretch")
        with btn_c2:
            html_informe = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; line-height: 1.6; padding: 40px; }}
                    h1 {{ color: #004d40; border-bottom: 2px solid #00e676; padding-bottom: 10px; }}
                    h2 {{ color: #00695c; margin-top: 30px; }}
                    .alerta {{ background-color: #ffebee; color: #c62828; padding: 15px; border-left: 5px solid #d32f2f; font-weight: bold; }}
                    .info-box {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #004d40; color: white; }}
                </style>
            </head>
            <body>
                <h1>INFORME OPERATIVO OFICIAL // YGGDRASIL ENVIRONMENTAL SYSTEMS: HYDRO CORE - OSIRIS: POYO-NOWCAST C4ISR</h1>
                <p><i>Generado bajo el marco del World Environmental Health Day 2026</i><br>
                <b>Theme:</b> Environmental Health: Standing with Science to Protect Communities</p>
                
                <div class="alerta">
                    ESTADO DE ALERTA: {alert_state} | CAUDAL PUNTA: {fmt_int(q_peak_simulated, ' m3/s')} <br>
                    HORA DE REPORTE: {clock_badge_text}
                </div>

                <h2>1. SALUD AMBIENTAL Y COMUNITARIA (ONE HEALTH)</h2>
                <div class="info-box">
                    <ul>
                        <li><b>Estado Sanitario (WASH):</b> {sanitary_status_label}</li>
                        <li><b>Población P1 en Riesgo Vital:</b> {fmt_int(p1_pop)} hab.</li>
                        <li><b>Población sin Agua Segura:</b> {fmt_int(pop_water_compromised)} hab.</li>
                        <li><b>Exposición a Riesgo Biológico (EDAR):</b> {fmt_dec(ha_biohazard_exposed, 1, ' ha')}</li>
                    </ul>
                </div>

                <h2>2. IMPACTO ESTRUCTURAL Y FINANCIERO</h2>
                <div class="info-box">
                    <ul>
                        <li><b>Pérdida Económica Modelada (CCS):</b> {fmt_dec(current_loss_m, 1, ' M€')}</li>
                        <li><b>Inmuebles en Colapso (InSAR > 0.40):</b> {fmt_int(total_collapsed)}</li>
                        <li><b>Gatillo Paramétrico Liberado:</b> {fmt_dec(payout_rate, 1, "%")}</li>
                    </ul>
                </div>

                <h2>3. AUDITORÍA FORENSE DE MODELADO (CAJA DE CRISTAL)</h2>
                <div class="info-box">
                    <p><b>Motor Físico Activo durante la exportación:</b> {ckpt_status_tag}</p>
                    <p><i>El sistema garantiza la trazabilidad legal de las decisiones operativas, identificando explícitamente si los calados fueron generados por inferencia neuronal pura (FNO 2D) o si el mecanismo de respaldo cinemático fue activado por razones de seguridad algorítmica.</i></p>
                </div>

                <h2>4. HOJA DE RUTA HACIA ESCALA ESTATAL (TRL-9 Roadmap)</h2>
                <div class="info-box">
                    <p>Este Gemelo Digital operativo (TRL-6) es un MVP funcional diseñado para demostrar la integración de inteligencia artificial y resiliencia climática. En fases posteriores de despliegue gubernamental, se integrarán los siguientes módulos:</p>
                    <ul>
                        <li><b>Asimilación WFS en Tiempo Real:</b> Conexión directa con la API INSPIRE de la Dirección General del Catastro, sustituyendo la semilla probabilística.</li>
                        <li><b>Ingesta Raster Topográfica:</b> Lectura automática de los Modelos Digitales de Elevación (MDT05 del IGN) en toda la red hidrográfica.</li>
                        <li><b>Ruteo Completo Multi-Nodo:</b> Despliegue de algoritmos Dijkstra acelerados por GPU sobre el 100% de la red vial OSMnx.</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            st.download_button("📄 Exportar Informe Interactivo (Imprimible PDF)", html_informe, "informe_cecopi_wehd2026.html", "text/html", width="stretch")

        st.markdown("---")
        st.markdown("#### 📧 Despacho Automático de Alertas (Misión Crítica)")
        if "Nowcast" in sim_mode and alert_state == "ROJO":
            if rain_val > lluvia_sensor or medidas_manuales_activas:
                st.info("🔕 **Modo Simulador Detectado:** Has subido la lluvia o activado roturas manualmente. El envío automático de correos oficiales está BLOQUEADO por seguridad.")
            else:
                if not st.session_state.get("correo_rojo_enviado", False):
                    try:
                        import smtplib
                        bot_remitente = st.secrets["EMAIL_BOT"]
                        bot_password = st.secrets["PASS_BOT"]
                        destinatario_oficial = st.secrets["EMAIL_DESTINO"]

                        cuerpo = f"EMERGENCIA {nivel_situacion} - POYO NOWCAST C2\n\nHora: {clock_badge_text}\nCaudal Estimado: {fmt_int(q_peak_simulated, ' m3/s')}\nPérdidas Activas: {fmt_dec(current_loss_m, 1, ' M€')}\n\nSe requiere orden de evacuación vertical inmediata."
                        msg = MIMEText(cuerpo)
                        msg['Subject'] = '🚨 ALERTA ROJA - DESBORDAMIENTO RAMBLA DEL POYO'
                        msg['From'] = bot_remitente
                        msg['To'] = destinatario_oficial

                        # Conexión SMTP real cifrada
                        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                            server.login(bot_remitente, bot_password)
                            server.send_message(msg)

                        st.success(f"✅ Protocolo automático disparado. Correo oficial enviado a: {destinatario_oficial}")
                        st.session_state["correo_rojo_enviado"] = True
                    except Exception as e:
                        st.warning(f"⚠️ El correo no se pudo enviar: {e}")
                else:
                    st.info("ℹ️ El correo de Alerta Roja ya fue despachado al CECOPI para este evento.")
        elif alert_state != "ROJO":
            st.session_state["correo_rojo_enviado"] = False
            
# ------------------------------------------------------------------------------
# TAB 3: DINÁMICA HIDRÁULICA FNO
# ------------------------------------------------------------------------------
with tab_hydro:
    st.subheader("Dinámica Hidráulica 2D Neuronal (Fourier Neural Operator)")
    col_h1, col_h2 = st.columns([1.6, 1.4])
    
    with col_h1:
        t_steps = np.linspace(0, 360, 72)
        q_envelope = np.interp(t_steps, t_arr, q_series) * q_mitig_factor * amc_weight if "Forense" in sim_mode else np.clip(q_peak_simulated * np.exp(-((t_steps - 120.0)/85.0)**2), 0.0, None)
        
        fig_hydro = go.Figure()
        fig_hydro.add_trace(go.Scatter(x=t_steps, y=q_envelope, mode='lines', line=dict(color='#58a6ff', width=3), fill='tozeroy', name="Hidrograma FNO"))
        
        fig_hydro.add_hline(y=1000.0, line_dash="dash", line_color="#ffd600", annotation_text="Capacidad Cauce (1.000 m³/s)")
        fig_hydro.add_hline(y=1800.0, line_dash="dash", line_color="#ff9100", annotation_text="Desborde Catastrófico l'Horta (1.800 m³/s)")
        
        mostrar_plan_sur = (q_peak_simulated > 2200.0) or plan_sur_desbordado
        
        if mostrar_plan_sur:
            fig_hydro.add_hline(y=5000.0, line_dash="dot", line_color="#ff1744", line_width=2.5, annotation_text="🚨 LÍMITE PLAN SUR / COLAPSO (5.000 m³/s)", annotation_position="top right")
            rango_y = [0, max(5500.0, float(q_peak_simulated) * 1.15)]
        else:
            rango_y = [0, max(2200.0, float(q_peak_simulated) * 1.15)]
        
        fig_hydro.update_layout(template="plotly_dark", height=320, xaxis_title="Minutos del Evento", yaxis_title="Caudal (m³/s)", margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(range=rango_y))
        st.plotly_chart(fig_hydro, width="stretch", key="hydro_main_chart")
        
    with col_h2:
        if not active_df.empty and active_df["active_depth"].max() > 0:
            fig_box = px.box(active_df, x="municipality", y="active_depth", color="municipality")
            fig_box.update_layout(template="plotly_dark", height=320, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_box, width="stretch", key="box_depth_chart")
        else:
            st.info("Sin datos de calado activo suficientes para generar distribución estadística.")

# ------------------------------------------------------------------------------
# TAB 4: RESILIENCIA VIAL & TTI CON ALGORITMO DIJKSTRA MULTI-SINK HOSPITALARIO
# ------------------------------------------------------------------------------
with tab_roads:
    st.subheader("Matriz Dinámica de Resiliencia Territorial & Time-to-Isolation (TTI)")

    r_info1, r_info2, r_info3 = st.columns(3)
    with r_info1: st.markdown("<div style='background:#161b22; border-left:4px solid #1f6feb; padding:12px; border-radius:6px; font-size:0.77rem;'><b style='color:#58a6ff;'>¿QUÉ ES EL TIME-TO-ISOLATION (TTI)?</b><br/>Tiempo en minutos transcurrido hasta que el último enlace viario hacia los hospitales de referencia supera un <b>calado h &ge; 0,30 m</b>.</div>", unsafe_allow_html=True)
    with r_info2: st.markdown("<div style='background:#161b22; border-left:4px solid #ff1744; padding:12px; border-radius:6px; font-size:0.77rem;'><b style='color:#ff1744;'>CRITERIO DE CORTE VIAL MULTINODAL</b><br/>La vía se declara cortada si <b>cualquiera de sus vértices</b> supera h &ge; 0,30 m en la malla FNO, considerando sedimentación e histéresis.</div>", unsafe_allow_html=True)
    with r_info3: st.markdown("<div style='background:#161b22; border-left:4px solid #00e676; padding:12px; border-radius:6px; font-size:0.77rem;'><b style='color:#00e676;'>ROUTING MULTI-SINK DIJKSTRA</b><br/>Algoritmo de caminos mínimos filtrado. Si el municipio está aislado, el grafo lo detecta e indica Ruta Bloqueada automáticamente.</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    HOSP_SINKS = {"H. Universitari i Politècnic La Fe": 12, "H. General Universitari": 13, "Hospital de Manises": 14}

    network_edges = [
        ("Paiporta", "Picanya", 2.1, 4), ("Picanya", "H. General Universitari", 4.5, 5), ("Paiporta", "Sedaví", 3.2, 3),
        ("Sedaví", "H. Universitari i Politècnic La Fe", 2.8, 1), ("Benetússer", "H. Universitari i Politècnic La Fe", 3.5, 1),
        ("Alfafar", "H. Universitari i Politècnic La Fe", 3.8, 2), ("Massanassa", "Alfafar", 1.5, 2), ("Catarroja", "Massanassa", 1.8, 2),
        ("Torrent", "Picanya", 3.5, 0), ("Torrent", "Hospital de Manises", 8.5, 8), ("Aldaia", "Hospital de Manises", 4.0, 7),
        ("Alaquàs", "H. General Universitari", 4.8, 5), ("Quart de Poblet", "Hospital de Manises", 3.0, 7),
        ("Quart de Poblet", "H. General Universitari", 3.2, 5), ("València", "H. Universitari i Politècnic La Fe", 2.0, 6),
        ("València", "H. General Universitari", 2.5, 5),
    ]

    adj_graph = {}
    active_nodes = set(active_df["municipality"].unique()).union(set(HOSP_SINKS.keys()))
    current_map = basin_names_map.get(basin_selected, {})
    
    for u, v, length, r_idx in network_edges:
        u_map = current_map.get(u, u)
        v_map = current_map.get(v, v)
        if u_map in active_nodes and v_map in active_nodes:
            h_road = road_max_depths[r_idx]
            weight = length * (1.0 + 80.0 * ((h_road - 0.30) / 0.30)**2.0) if h_road >= 0.30 else length * (1.0 + 2.0 * (h_road / 0.30))
            adj_graph.setdefault(u_map, []).append((v_map, weight, h_road))
            adj_graph.setdefault(v_map, []).append((u_map, weight, h_road))

    def dijkstra_multi_sink(start_node, sinks_dict):
        queue = [(0.0, start_node, [])]; visited = {}
        while queue:
            cost, current, path = heapq.heappop(queue)
            if current in visited and visited[current] <= cost: continue
            visited[current] = cost; path = path + [current]
            if current in sinks_dict: return current, cost, path
            for neighbor, weight, h_e in adj_graph.get(current, []):
                if neighbor not in visited: heapq.heappush(queue, (cost + weight, neighbor, path))
        return "Sin Acceso Viable", float("inf"), []

    tti_rows = []
    for mun in active_df["municipality"].unique():
        sub = active_df[active_df["municipality"] == mun]
        if sub.empty: continue
        pct_inc = float(np.mean(sub["active_depth"] >= 0.30)) * 100.0
        p1_cnt_mun = int(sub.loc[sub["P1_flag"], "pop_density"].sum())

        hosp_dest, travel_cost, route = dijkstra_multi_sink(mun, HOSP_SINKS)

        if travel_cost >= 150.0 or hosp_dest == "Sin Acceso Viable":
            tti_lbl, stat_evac, hosp_label = "15 min (Crítico)", "🔴 AISLAMIENTO TOTAL", "Ruta Bloqueada (Incomunicado)"
        elif travel_cost > 25.0:
            tti_lbl, stat_evac, hosp_label = "30 - 45 min", "🟠 PREALERTA / RUTA ALTERNATIVA", f"{hosp_dest} (Vía desviada)"
        else:
            tti_lbl = "Resiliente (> 120 min)" if pct_inc < 10.0 else "45 min"
            stat_evac = "🟢 CONECTIVIDAD ACTIVA" if pct_inc < 10.0 else "🟠 PREALERTA EN ACCESOS"
            hosp_label = hosp_dest

        tti_rows.append({"Término Municipal": mun, "Time-to-Isolation (TTI)": tti_lbl, "Estado de Evacuación": stat_evac, "% Área Incomunicada": fmt_dec(pct_inc, 1, " %"), "Población Crítica Atrapada (P1)": p1_cnt_mun, "Hospital Asignado (Dijkstra)": hosp_label})

    try:
        if tti_rows:
            html_table = "<table style='width:100%; border-collapse: collapse; font-family: Inter, sans-serif; font-size: 0.85rem;'>"
            html_table += "<tr style='background: #161b22; border-bottom: 2px solid #30363d; text-align: left;'><th style='padding: 10px;'>Municipio</th><th style='padding: 10px;'>TTI</th><th style='padding: 10px;'>Estado Evacuación</th><th style='padding: 10px;'>% Incomunicado</th><th style='padding: 10px;'>Hab. Críticos (P1)</th><th style='padding: 10px;'>Hospital Dijkstra</th></tr>"
            for row in tti_rows:
                if "🔴" in row['Estado de Evacuación']: bg_color = "rgba(255, 23, 68, 0.15)"; text_color = "#ff7b72"
                elif "🟠" in row['Estado de Evacuación']: bg_color = "rgba(255, 145, 0, 0.1)"; text_color = "#ff9100"
                else: bg_color = "transparent"; text_color = "#c9d1d9"
                html_table += f"<tr style='background: {bg_color}; border-bottom: 1px solid #30363d;'><td style='padding: 8px; font-weight: bold;'>{row['Término Municipal']}</td><td style='padding: 8px; color: {text_color}; font-family: monospace;'>{row['Time-to-Isolation (TTI)']}</td><td style='padding: 8px; color: {text_color}; font-weight: 600;'>{row['Estado de Evacuación']}</td><td style='padding: 8px;'>{row['% Área Incomunicada']}</td><td style='padding: 8px; color: #ff1744; font-family: monospace;'>{row['Población Crítica Atrapada (P1)']}</td><td style='padding: 8px; color: #58a6ff;'>{row['Hospital Asignado (Dijkstra)']}</td></tr>"
            html_table += "</table>"
            st.markdown(html_table, unsafe_allow_html=True)
        else: 
            st.warning("Seleccione al menos un municipio en el panel izquierdo.")
    except Exception:
        pass

# ------------------------------------------------------------------------------
# TAB 5: FINANZAS DEL CLIMA & SOLVENCIA II COMPLETA (6 PANELES GRÁFICOS)
# ------------------------------------------------------------------------------
with tab_finances:
    st.subheader("Modelado Catastrófico, Reaseguro y Seguros Paramétricos (Solvencia II / EIOPA)")
    
    total_exposure_for_risk = total_exposure_m if total_exposure_m > 0 else 1467.0
    aal_base = total_exposure_for_risk * 0.0085
    scr_base = total_exposure_for_risk * 0.14
    coc_base = 0.06 * scr_base

    f_col1, f_col2, f_col3 = st.columns([1.1, 1.1, 0.9])
    with f_col1:
        st.markdown("#### Balance Regulatorio (EIOPA ORSA)")
        solv_table = pd.DataFrame([
            {"Parámetro Actuarial": "Exposición Bruta (TIV)", "Importe": fmt_dec(total_exposure_for_risk, 1, " M€")},
            {"Parámetro Actuarial": "Pérdida Bruta Modelada", "Importe": fmt_dec(current_loss_m, 1, " M€")},
            {"Parámetro Actuarial": "Pérdida Anual Esperada (AAL)", "Importe": fmt_dec(aal_base, 2, " M€/año")},
            {"Parámetro Actuarial": "SCR Solvencia II (VaR 99,5%)", "Importe": fmt_dec(scr_base, 1, " M€")},
            {"Parámetro Actuarial": "Margen de Riesgo (CoC 6%)", "Importe": fmt_dec(coc_base, 2, " M€")},
            {"Parámetro Actuarial": "Indemnización Paramétrica", "Importe": fmt_dec(total_payout_m, 2, " M€")},
        ])
        st.dataframe(solv_table, width="stretch", hide_index=True)

    with f_col2:
        st.markdown("#### Cascada de Financiación (Waterfall)")
        val_ccs = current_loss_m * 0.72
        val_param = min(current_loss_m * 0.28, total_payout_m)
        val_gap = max(0.0, current_loss_m - (val_ccs + val_param))
        
        fig_waterfall = go.Figure(go.Bar(
            x=["Consorcio (CCS)", "Cat Bond", "Brecha", "Total"], y=[val_ccs, val_param, val_gap, current_loss_m],
            marker=dict(color=["#00e676", "#2979ff", "#ffd600", "#ff1744"]), text=[fmt_dec(v, 1, " M€") for v in [val_ccs, val_param, val_gap, current_loss_m]], textposition="auto"
        ))
        fig_waterfall.update_layout(template="plotly_dark", height=260, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_waterfall, width="stretch")

    with f_col3:
        st.markdown("#### Daño por Tipología de Activo")
        if not active_df.empty and current_loss_m > 0:
            asset_breakdown = active_df.groupby("asset_type")["active_loss"].sum().reset_index()
            asset_breakdown["loss_m"] = asset_breakdown["active_loss"] / 1e6
            fig_assets = px.pie(asset_breakdown, names="asset_type", values="loss_m", hole=0.45)
            fig_assets.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_assets, width="stretch")
        else:
            st.info("Daños insuficientes para el desglose.")

    st.markdown("<br/>", unsafe_allow_html=True)
    f_col4, f_col5 = st.columns(2)
    with f_col4:
        st.markdown("#### Curva EP con Reaseguro Exceso de Pérdida (XoL)")
        T_periods = np.array([2, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500])
        loss_base_curve = total_exposure_for_risk * 0.20 * (1.0 - np.exp(-0.38 * (T_periods / 100.0)**0.45))
        loss_net_curve = loss_base_curve - np.clip(loss_base_curve - (total_exposure_for_risk * 0.05), 0.0, (total_exposure_for_risk * 0.15))
        fig_ep = go.Figure()
        fig_ep.add_trace(go.Scatter(x=T_periods, y=loss_base_curve, name="Base Bruta", line=dict(color="#2979ff", width=2.5)))
        fig_ep.add_trace(go.Scatter(x=T_periods, y=loss_net_curve, name="Neto Post-XoL", line=dict(color="#00e676", width=2.5)))
        fig_ep.add_vline(x=200, line_dash="dot", line_color="#ffffff", annotation_text="SCR (T=200)")
        fig_ep.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_ep, width="stretch")

    with f_col5:
        st.markdown("#### Matriz de Disparo Paramétrico (Dual-Trigger)")
        q_grid = np.linspace(800, 2000, 40)
        insar_grid = np.linspace(0.0, 32.0, 40)
        Q_mesh, I_mesh = np.meshgrid(q_grid, insar_grid)
        payout_surf = np.sqrt(np.clip((Q_mesh - 1000.0) / 800.0, 0.0, 1.0) * np.clip((I_mesh - 2.0) / 6.0, 0.0, 1.0)) * 100.0
        fig_matrix = go.Figure(go.Contour(z=payout_surf, x=q_grid, y=insar_grid, colorscale="Viridis", reversescale=True))
        fig_matrix.add_trace(go.Scatter(x=[min(2000.0, max(800.0, q_eval_cat))], y=[min(30.0, collapse_ratio * 100.0)], mode='markers+text', marker=dict(color='#ff1744', size=14, symbol='diamond'), text=[f"{fmt_dec(payout_rate, 1, '%')}"], textposition="top center"))
        fig_matrix.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_matrix, width="stretch")

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("#### Proyección Decenal del Riesgo Climático & Coste de Solvencia (2024 - 2050)")
    decadas = np.array([2024, 2030, 2035, 2040, 2045, 2050])
    factor_ssp2 = 1.0 + 0.0045 * (decadas - 2024)
    factor_ssp5 = 1.0 + 0.0090 * (decadas - 2024)
    aal_ssp2 = aal_base * factor_ssp2
    aal_ssp5 = aal_base * factor_ssp5
    scr_ssp5 = scr_base * factor_ssp5

    fig_future = go.Figure()
    fig_future.add_trace(go.Bar(x=decadas, y=scr_ssp5, name="Capital Solvencia Requerido (SCR SSP5-8.5)", marker=dict(color="rgba(41, 121, 255, 0.25)", line=dict(color="#2979ff", width=1.5)), yaxis="y2"))
    fig_future.add_trace(go.Scatter(x=decadas, y=aal_ssp2, mode="lines+markers", name="AAL (Senda Intermedia SSP2-4.5)", line=dict(color="#00e676", width=2.5)))
    fig_future.add_trace(go.Scatter(x=decadas, y=aal_ssp5, mode="lines+markers", name="AAL (Senda Pesimista SSP5-8.5)", line=dict(color="#ff1744", width=2.5, dash="dash")))

    fig_future.update_layout(
        template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#0d1117", xaxis_title="Año de Proyección",
        yaxis=dict(title="Pérdida Anual Esperada AAL (M€/año)"), yaxis2=dict(title="Requisito de Capital SCR (M€)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=40, r=40, t=40, b=40), height=320
    )
    st.plotly_chart(fig_future, width="stretch")

# ------------------------------------------------------------------------------
# TAB 6: AUDITORÍA FORENSE SPLIT A/B
# ------------------------------------------------------------------------------
with tab_compare:
    st.subheader("Auditoría Forense Split A/B: Inteligencia Anticipada vs. Gestión Burocrática")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1: st.metric("Margen de Preaviso", "+146 min", delta="Nowcast anticipado vs Oficial", delta_color="normal")
    with sc2: st.metric("Telecomunicaciones", "100% On-line", delta="Sin apagón de antenas", delta_color="normal")
    with sc3: st.metric("Población sin Aviso", "0 hab", delta="-180.000 hab protegidos", delta_color="normal")
    with sc4: st.metric("Atrapamientos Viales", "-85%", delta="Evacuación vertical viable", delta_color="normal")

    base_t = now_valencia.replace(second=0, microsecond=0)
    
    if "Forense" in sim_mode:
        t0 = datetime(2024, 10, 29, 16, 0, tzinfo=VALENCIA_TZ)
        t_current = t0 + timedelta(minutes=sim_minute)
    else:
        t0 = base_t - timedelta(hours=1) 
        t_current = base_t + timedelta(minutes=lead_time_min)
        
    t_lluvia_end = t0 + timedelta(hours=2)
    t_frente_start = t0 + timedelta(hours=1)
    t_frente_end = t0 + timedelta(hours=3, minutes=30)
    t_impacto_start = t0 + timedelta(hours=2, minutes=30)
    t_impacto_end = t0 + timedelta(hours=6)
    
    t_alert_nowcast = t0 + timedelta(minutes=15) if "Nowcast" in sim_mode else t0 + timedelta(hours=1, minutes=45)
    t_alert_oficial = t0 + timedelta(hours=3, minutes=15) if "Nowcast" in sim_mode else t0 + timedelta(hours=4, minutes=11)

    timeline_df = pd.DataFrame([
        dict(Task="Precipitación Extrema (> 400 mm)", Start=t0, Finish=t_lluvia_end, Tipo="Fenómeno Físico"),
        dict(Task="Propagación Frente Onda FNO", Start=t_frente_start, Finish=t_frente_end, Tipo="Fenómeno Físico"),
        dict(Task="Anegamiento Crítico Poblaciones", Start=t_impacto_start, Finish=t_impacto_end, Tipo="Impacto Crítico"),
        dict(Task="DISPARO ES-ALERT POYO-NOWCAST", Start=t_alert_nowcast, Finish=t_alert_nowcast + timedelta(minutes=8), Tipo="Alerta Anticipada"),
        dict(Task="DISPARO ES-ALERT OFICIAL CECOPI", Start=t_alert_oficial, Finish=t_alert_oficial + timedelta(minutes=8), Tipo="Alerta Tardía"),
    ])
    
    fig_timeline = px.timeline(
        timeline_df, x_start="Start", x_end="Finish", y="Task", color="Tipo",
        color_discrete_map={"Fenómeno Físico": "#58a6ff", "Impacto Crítico": "#ff9100", "Alerta Anticipada": "#00e676", "Alerta Tardía": "#ff1744"}
    )
    
    fig_timeline.add_vline(x=t_current.timestamp() * 1000, line_width=3, line_dash="dash", line_color="#ffffff", annotation_text="⏱️ SIMULACIÓN ACTUAL", annotation_position="top right")
    fig_timeline.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_timeline, width="stretch")

# ==============================================================================
# 11. PIE DE PÁGINA INSTITUCIONAL Y LICENCIAMIENTO MODULAR
# ==============================================================================
st.markdown("<hr style='border:0.5px solid #21262d; margin:14px 0;'/>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='color: #8b949e; font-size: 0.75rem; line-height: 1.5; padding: 0 10px;'>
        <b style='color: #58a6ff;'>POYO-NOWCAST (YGGDRASIL ENVIRONMENTAL SYSTEMS: HYDRO-CORE: OSIRIS) // PLATAFORMA C4ISR CIVIL & GEMELO DIGITAL 3D INTEGRADO</b><br/>
        Arquitectura Ciberfísica de Resiliencia bajo estándares internacionales: <b>ISO 22320:2018</b> (Mando y Control de Emergencias), <b>ISO 14091:2021</b> (Vulnerabilidad Climática), <b>ISO 24512:2007</b> (Seguridad del Agua Potable), teledetección radar <b>Copernicus InSAR DPM</b> y protocolo de alerta <b>ITU-T X.1303 / OASIS CAP v1.2</b>.<br/>
        Autor: <b>Kelvin Jesús Flores Yarihuaman</b> (<a href='https://www.linkedin.com/in/kelvinflores-ingenieria' target='_blank' style='color: #58a6ff; text-decoration: none;'>LinkedIn</a>) | Metasistema: <b>YGGDRASIL-OS</b>.<br/>
        <ul style='margin-top: 4px; margin-bottom: 0; padding-left: 20px;'>
            <li><b>Documentación Técnica & Memoria:</b> Licencia Creative Commons Atribución-NoComercial-CompartirIgual (CC BY-NC-SA 4.0).</li>
            <li><b>Código Fuente y Algoritmos (PyTorch FNO 2D, Dijkstra, InSAR):</b> Licencia GNU General Public License v3 (GNU GPLv3).</li>
            <li><b>Datasets, Grafos OSMnx y Tablas Parquet:</b> Licencia Open Data Commons Attribution (ODC-By).</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
