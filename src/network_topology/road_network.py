"""
POYO-NOWCAST: Módulo 3 - Redes Críticas y Resiliencia Territorial Operativa
Tecnología: Python (NetworkX, NumPy, PyArrow, JSON) + Multi-Sink Dijkstra
Arquitectura: Hidráulica de Intradós -> Intersección Geoespacial InSAR LOD1 ->
              Cálculo TTI (Time-to-Isolation) -> Exportación GeoJSON (CECOPI/112).

Autor: Kelvin Jesus Flores Yarihuaman (https://www.linkedin.com/in/kelvinflores-ingenieria)
Metasistema: YGGDRASIL-OS | Framework de Resiliencia Climática

LICENCIAMIENTO MODULAR:
- Documentación y Memoria Técnica: CC BY-NC-SA 4.0
- Código Fuente y Algoritmos: GNU GPLv3
- Datasets y Tablas Parquet: ODC-By
"""

from typing import Dict, List, Tuple, Optional, Any
import json
import networkx as nx
import numpy as np
import pyarrow.parquet as pq


class MissionCriticalRoadNetwork:
    """
    Grafo vial dinámico multihospital acoplado al tensor bidimensional del FNO
    y a las cicatrices de ruina estructural del procesador InSAR DPM.
    """

    DEFAULT_RESGUARDO_M = 0.50  # Margen de seguridad contra flotantes, cañas y oleaje

    def __init__(self, parquet_dpm_path: Optional[str] = None, insar_buffer_m: float = 75.0):
        self.graph = nx.DiGraph()
        self.permanently_destroyed_edges = set()
        self.hospitals = {
            "HOSPITAL_LA_FE": {"name": "H. Universitari i Politècnic La Fe", "capacity": "ALTA"},
            "HOSPITAL_GENERAL": {"name": "H. General Universitari de València", "capacity": "MEDIA"},
            "HOSPITAL_MANISES": {"name": "Hospital de Manises", "capacity": "MEDIA"},
        }
        self._initialize_topology()

        if parquet_dpm_path:
            self._apply_insar_spatial_damage(parquet_dpm_path, buffer_m=insar_buffer_m)

    def _initialize_topology(self) -> None:
        """Inicializa la red con coordenadas métricas UTM 30N y perfiles de rasante."""
        nodes = [
            ("N_CHIVA", {"mun": "Chiva", "x": 699500.0, "y": 4377000.0}),
            ("N_CHESTE", {"mun": "Cheste", "x": 703200.0, "y": 4374500.0}),
            ("N_TORRENT", {"mun": "Torrent", "x": 720500.0, "y": 4368500.0}),
            ("N_PICANYA", {"mun": "Picanya", "x": 722000.0, "y": 4369500.0}),
            ("N_PAIPORTA", {"mun": "Paiporta", "x": 723500.0, "y": 4368200.0}),
            ("N_SEDAVI", {"mun": "Sedaví", "x": 725800.0, "y": 4368900.0}),
            ("N_CATARROJA", {"mun": "Catarroja", "x": 724800.0, "y": 4365800.0}),
            ("N_ALBAL", {"mun": "Albal", "x": 724200.0, "y": 4364500.0}),
            ("N_V30_SUR", {"mun": "Valencia_Sur", "x": 725500.0, "y": 4371500.0}),
            ("N_V30_OESTE", {"mun": "Valencia_Oeste", "x": 722800.0, "y": 4374100.0}),
            ("HOSPITAL_LA_FE", {"mun": "Valencia", "x": 726400.0, "y": 4372100.0}),
            ("HOSPITAL_GENERAL", {"mun": "Valencia", "x": 723500.0, "y": 4373800.0}),
            ("HOSPITAL_MANISES", {"mun": "Manises", "x": 718200.0, "y": 4377200.0}),
        ]
        for n_id, attrs in nodes:
            self.graph.add_node(n_id, **attrs)

        # edges: (u, v, length_m, name, z_deck, is_bridge, nominal_speed_kmh)
        edges = [
            ("N_CHIVA", "N_CHESTE", 6500.0, "A-3", 15.0, False, 80.0),
            ("N_CHESTE", "N_TORRENT", 14000.0, "CV-411", 8.0, False, 70.0),
            ("N_TORRENT", "N_PICANYA", 2200.0, "CV-366", 4.5, False, 50.0),
            ("N_PICANYA", "N_PAIPORTA", 1800.0, "Puente_Picanya_CV-407", 2.2, True, 45.0),
            ("N_PAIPORTA", "N_SEDAVI", 2500.0, "Camí_Compartit", 1.8, True, 40.0),
            ("N_PAIPORTA", "N_CATARROJA", 2800.0, "CV-400", 3.0, False, 50.0),
            ("N_CATARROJA", "N_ALBAL", 1500.0, "V-31_Enlace", 4.0, False, 60.0),
            ("N_SEDAVI", "N_V30_SUR", 2800.0, "V-31_Pista_Silla", 3.5, False, 70.0),
            ("N_PAIPORTA", "N_V30_SUR", 3400.0, "CV-36_Eje_Sur", 2.0, True, 60.0),
            ("N_TORRENT", "N_V30_OESTE", 4800.0, "CV-36_Norte", 5.0, False, 70.0),
            ("N_V30_SUR", "HOSPITAL_LA_FE", 1500.0, "Bulevar_Sur", 10.0, False, 50.0),
            ("N_V30_OESTE", "HOSPITAL_GENERAL", 1800.0, "Av_Tres_Cruces", 10.0, False, 50.0),
            ("N_TORRENT", "HOSPITAL_MANISES", 7200.0, "CV-405_CV-370", 8.0, False, 60.0),
        ]

        for u, v, length, name, z_deck, is_bridge, speed_kmh in edges:
            t_base = (length / 1000.0) / speed_kmh * 60.0
            edge_attrs = {
                "length_m": length,
                "name": name,
                "z_deck": z_deck,
                "is_bridge": is_bridge,
                "nominal_speed_kmh": speed_kmh,
                "base_time_min": t_base,
                "travel_time_min": t_base,
                "is_active": True,
                "failure_reason": "NINGUNO",
                "h_water_m": 0.0,
                "vel_water_ms": 0.0,
            }
            self.graph.add_edge(u, v, **edge_attrs)
            self.graph.add_edge(v, u, **edge_attrs)

    def _point_to_segment_distance(
        self, px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """Calcula la distancia euclídea mínima entre un punto catastral y el segmento vial."""
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        if l2 == 0.0:
            return float(np.hypot(px - x1, py - y1))

        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return float(np.hypot(px - proj_x, py - proj_y))

    def _apply_insar_irreversible_damage(self, parquet_path: str, buffer_m: float = 75.0) -> None:
        """
        Cruza parcelas con colapso estructural InSAR (DPM >= 0.40) con la infraestructura vial.
        Inhabilita permanentemente los puentes que presenten ruina estructural en su entorno.
        """
        table = pq.read_table(parquet_path, columns=["x_coord", "y_coord", "structural_collapse"])
        df = table.to_pandas()
        collapsed = df[df["structural_collapse"] == True]

        if collapsed.empty:
            return

        c_coords = collapsed[["x_coord", "y_coord"]].to_numpy()

        for u, v, d in list(self.graph.edges(data=True)):
            if (u, v) in self.permanently_destroyed_edges or not d["is_bridge"]:
                continue

            x1, y1 = self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]
            x2, y2 = self.graph.nodes[v]["x"], self.graph.nodes[v]["y"]

            # Bounding box rápido para descartar parcelas lejanas
            min_x, max_x = min(x1, x2) - buffer_m, max(x1, x2) + buffer_m
            min_y, max_y = min(y1, y2) - buffer_m, max(y1, y2) + buffer_m

            in_box = (
                (c_coords[:, 0] >= min_x)
                & (c_coords[:, 0] <= max_x)
                & (c_coords[:, 1] >= min_y)
                & (c_coords[:, 1] <= max_y)
            )
            candidate_points = c_coords[in_box]

            for px, py in candidate_points:
                dist = self._point_to_segment_distance(px, py, x1, y1, x2, y2)
                if dist <= buffer_m:
                    self.permanently_destroyed_edges.add((u, v))
                    self.permanently_destroyed_edges.add((v, u))
                    d["is_active"] = False
                    d["failure_reason"] = "COLAPSO_ESTRUCTURAL_INSAR"
                    self.graph[v][u]["is_active"] = False
                    self.graph[v][u]["failure_reason"] = "COLAPSO_ESTRUCTURAL_INSAR"
                    print(f"[InSAR] {d['name']} ({u} <-> {v}) destruido permanentemente por proximidad a colapso.")
                    break

    def evaluate_edge_hydraulics(
        self, u: str, v: str, h_water: float, vel_water: float
    ) -> Tuple[bool, float, str]:
        """
        Distingue rigurosamente entre calado del cauce (intradós) y agua sobre tablero:
        - Si es puente y h_water >= z_deck - resguardo: Falla por taponamiento / empuje hidrodinámico.
        - Si es puente y h_water < z_deck: El tablero está seco; velocidad nominal.
        - Si el agua rebasa el tablero o es calzada: Aplica curva de Pregnolato y criterio de arrastre.
        """
        if (u, v) in self.permanently_destroyed_edges:
            return False, float("inf"), "COLAPSO_ESTRUCTURAL_INSAR"

        edge = self.graph[u][v]
        z_deck = edge["z_deck"]
        is_bridge = edge["is_bridge"]
        base_time = edge["base_time_min"]
        v_nominal = edge["nominal_speed_kmh"]
        length_km = edge["length_m"] / 1000.0

        # 1. Hidráulica de Puente (Intradós y Capacidad de Desagüe)
        if is_bridge:
            intrados_critico = z_deck - self.DEFAULT_RESGUARDO_M
            if h_water >= intrados_critico:
                return False, float("inf"), "TAPONAMIENTO_INTRADOS_PUENTE"

            # Agua sobre la plataforma del puente
            h_deck = max(0.0, h_water - z_deck)
        else:
            # En calzada general, el calado inunda directamente la superficie
            h_deck = h_water

        # 2. Criterios de Transitabilidad en Superficie de Rodadura
        if h_deck >= 0.30:
            return False, float("inf"), "CALADO_CRITICO_FLOTABILIDAD"

        if (h_deck * vel_water) >= 0.50:
            return False, float("inf"), "ARRASTRE_HIDRODINAMICO_VH"

        # 3. Degradación Continua de Velocidad (Pregnolato et al., 2017)
        if h_deck <= 0.02:
            return True, base_time, "OPERATIVO_SECO"

        speed_factor = max(0.05, 1.0 - (h_deck / 0.30) ** 2)
        effective_speed = v_nominal * speed_factor
        travel_time_min = (length_km / effective_speed) * 60.0

        return True, travel_time_min, "OPERATIVO_CON_DEMORA"

    def update_network_hydraulics(
        self, flood_conditions: Dict[Tuple[str, str], Tuple[float, float]]
    ) -> None:
        """Actualiza el estado de las aristas asegurando coherencia bidireccional."""
        for (u, v), (h, vel) in flood_conditions.items():
            if not self.graph.has_edge(u, v):
                continue

            for start, end in [(u, v), (v, u)]:
                passable, t_travel, reason = self.evaluate_edge_hydraulics(start, end, h, vel)
                edge_data = self.graph[start][end]
                edge_data["is_active"] = passable
                edge_data["travel_time_min"] = t_travel
                edge_data["failure_reason"] = reason
                edge_data["h_water_m"] = round(h, 2)
                edge_data["vel_water_ms"] = round(vel, 2)

    def compute_optimal_hospital_routes(
        self, origins: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Resuelve el enrutamiento multi-hospital en una sola pasada de Dijkstra por origen.
        Identifica el mejor hospital accesible, tiempo estimado y ruta crítica.
        """
        active_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d["is_active"] and (u, v) not in self.permanently_destroyed_edges
        ]
        subgraph = self.graph.edge_subgraph(active_edges)

        routing_summary = {}

        for orig in origins:
            if orig not in subgraph:
                routing_summary[orig] = {
                    "status": "AISLADO_TOTAL",
                    "best_hospital": None,
                    "eta_min": float("inf"),
                    "path": [],
                    "all_destinations": {},
                }
                continue

            try:
                # Single-source Dijkstra hacia todos los nodos alcanzables
                distances, paths = nx.single_source_dijkstra(
                    subgraph, source=orig, weight="travel_time_min"
                )
            except nx.NetworkXError:
                distances, paths = {}, {}

            dest_reports = {}
            best_hosp = None
            min_eta = float("inf")

            for h_id, h_info in self.hospitals.items():
                if h_id in distances:
                    eta = round(distances[h_id], 2)
                    dest_reports[h_id] = {"name": h_info["name"], "eta_min": eta, "path": paths[h_id]}
                    if eta < min_eta:
                        min_eta = eta
                        best_hosp = h_info["name"]
                else:
                    dest_reports[h_id] = {"name": h_info["name"], "eta_min": float("inf"), "path": []}

            is_isolated = min_eta == float("inf")
            routing_summary[orig] = {
                "status": "AISLADO_TOTAL" if is_isolated else "CONECTADO",
                "best_hospital": best_hosp,
                "eta_min": min_eta if not is_isolated else float("inf"),
                "path": paths.get(best_hosp, []) if best_hosp else [],
                "all_destinations": dest_reports,
            }

        return routing_summary

    def simulate_temporal_rollout_and_tti(
        self,
        time_steps_min: List[int],
        hydraulic_timeseries: Dict[int, Dict[Tuple[str, str], Tuple[float, float]]],
        origins: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Nowcasting temporal de red: Evalúa la propagación de la inundación y determina
        el instante exacto de aislamiento total (Time-to-Isolation / TTI).
        """
        tti_report = {
            orig: {
                "t_cut_min": float("inf"),
                "initial_hospital": None,
                "initial_eta_min": float("inf"),
                "failure_phase": "CONECTIVIDAD_PRESERVADA",
            }
            for orig in origins
        }

        for idx, t in enumerate(time_steps_min):
            # Ingesta del estado hidráulico proyectado para el paso t
            if t in hydraulic_timeseries:
                self.update_network_hydraulics(hydraulic_timeseries[t])

            routes = self.compute_optimal_hospital_routes(origins)

            for orig in origins:
                res = routes[orig]

                # Registro de estado base (T=0)
                if idx == 0 and res["status"] == "CONECTADO":
                    tti_report[orig]["initial_hospital"] = res["best_hospital"]
                    tti_report[orig]["initial_eta_min"] = res["eta_min"]

                # Registro del primer corte irreversible
                if res["status"] == "AISLADO_TOTAL" and tti_report[orig]["t_cut_min"] == float("inf"):
                    tti_report[orig]["t_cut_min"] = t
                    tti_report[orig]["failure_phase"] = f"AISLADO_EN_T_{t}_MIN"

        return tti_report

    def export_gis_geojson(self, output_geojson_path: str) -> None:
        """Exporta el estado geográfico de la red a formato GeoJSON estándar para visores 112."""
        features = []

        # 1. Exportar Nodos
        for n, d in self.graph.nodes(data=True):
            feat = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [d["x"], d["y"]]},
                "properties": {
                    "id": n,
                    "municipality": d["mun"],
                    "is_hospital": n in self.hospitals,
                },
            }
            features.append(feat)

        # 2. Exportar Aristas (Líneas entre nodos)
        processed = set()
        for u, v, d in self.graph.edges(data=True):
            edge_id = tuple(sorted([u, v]))
            if edge_id in processed:
                continue
            processed.add(edge_id)

            p1 = [self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]]
            p2 = [self.graph.nodes[v]["x"], self.graph.nodes[v]["y"]]

            feat = {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [p1, p2]},
                "properties": {
                    "source": u,
                    "target": v,
                    "name": d["name"],
                    "is_bridge": d["is_bridge"],
                    "is_active": d["is_active"],
                    "travel_time_min": round(d["travel_time_min"], 2) if d["is_active"] else -1.0,
                    "failure_reason": d["failure_reason"],
                    "h_water_m": d["h_water_m"],
                    "vel_water_ms": d["vel_water_ms"],
                },
            }
            features.append(feat)

        geojson_data = {"type": "FeatureCollection", "features": features}
        with open(output_geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2, ensure_ascii=False)
        print(f"[GIS] Capa operativa GeoJSON serializada en: {output_geojson_path}")


if __name__ == "__main__":
    net = MissionCriticalRoadNetwork()
    monitored_municipalities = ["N_PAIPORTA", "N_SEDAVI", "N_TORRENT", "N_CHIVA"]

    # Serie temporal hidráulica (Simulación de la avenida entre T=0 y T=40 min)
    # A los 20 min el caudal en Paiporta alcanza 1.90 m (afecta intradós: z_deck=2.2m, resguardo=0.5m -> corte en 1.7m)
    time_series_mock = {
        0: {
            ("N_PICANYA", "N_PAIPORTA"): (0.40, 0.5),     # Tablero seco
            ("N_PAIPORTA", "N_V30_SUR"): (0.10, 0.2),
        },
        10: {
            ("N_PICANYA", "N_PAIPORTA"): (1.40, 1.2),    # Tablero seco (1.40 < 1.70)
            ("N_PAIPORTA", "N_V30_SUR"): (0.35, 0.8),    # Calzada anegada con calado moderado (demora)
        },
        20: {
            ("N_PICANYA", "N_PAIPORTA"): (1.85, 2.1),    # Colapso por intradós (1.85 >= 1.70 m)
            ("N_PAIPORTA", "N_V30_SUR"): (0.65, 1.5),    # Cortado por arrastre / flotabilidad
        },
        30: {
            ("N_PICANYA", "N_PAIPORTA"): (2.60, 2.8),
            ("N_PAIPORTA", "N_V30_SUR"): (1.90, 2.0),
            ("N_SEDAVI", "N_V30_SUR"): (1.20, 1.8),      # Aislamiento sur de Valencia
        },
    }

    report = net.simulate_temporal_rollout_and_tti(
        time_steps_min=[0, 10, 20, 30],
        hydraulic_timeseries=time_series_mock,
        origins=monitored_municipalities,
    )

    print("\n================ SINTESIS OPERATIVA TIME-TO-ISOLATION ================")
    for mun, data in report.items():
        t_cut = data["t_cut_min"]
        hosp = data["initial_hospital"]
        eta = data["initial_eta_min"]
        if t_cut < float("inf"):
            print(f"🔴 {mun:12s} | TTI: {t_cut:2d} min | Evacuación Inicial: {hosp} ({eta:.1f} min) | Estado: {data['failure_phase']}")
        else:
            print(f"🟢 {mun:12s} | TTI: RESILIENTE | Evacuación: {hosp} ({eta:.1f} min) | Estado: CONECTIVIDAD TOTAL")

    # Generación de capa vectorial para centros de control
    import os
    os.makedirs("data/processed", exist_ok=True)
    net.export_gis_geojson("data/processed/horta_sud_road_network_status.geojson")
