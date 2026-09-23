"""
POYO-NOWCAST: Integrador del Contrato Columnar Apache Parquet
Tecnología: Python (PyArrow, Apache Parquet, NumPy) + Analítica de Datos
Fusión multidominio: InSAR DPM (M1) + Hidráulica FNO 2D (M2) + Resiliencia Vial TTI (M3).

Autor: Kelvin Jesus Flores Yarihuaman (https://www.linkedin.com/in/kelvinflores-ingenieria)
Metasistema: YGGDRASIL-OS | Framework de Resiliencia Climática

LICENCIAMIENTO MODULAR:
- Documentación y Memoria Técnica: CC BY-NC-SA 4.0
- Código Fuente y Algoritmos: GNU GPLv3
- Datasets y Tablas Parquet: ODC-By
"""

from typing import Dict, Tuple, Optional, Any
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class UnifiedDamageMatrixIntegrator:
    """
    Consolidador del contrato de datos de emergencia para la Horta Sud.
    Cruza daño estructural por satélite, hidrodinámica neuronal y aislamiento vial.
    """

    # Esquema formal inmutable para analítica en tiempo real y peritaje CCS
    SCHEMA = pa.schema([
        ("parcel_id", pa.string()),
        ("municipality", pa.dictionary(pa.int8(), pa.string())),
        ("x_coord", pa.float64()),
        ("y_coord", pa.float64()),
        ("is_critical_infra", pa.bool_()),
        # Variables Hidráulicas (Módulo 2 - FNO)
        ("max_depth_m", pa.float32()),
        ("max_velocity_ms", pa.float32()),
        ("hazard_factor_vh", pa.float32()),
        ("time_to_inund_min", pa.float32()),
        # Variables InSAR DPM (Módulo 1)
        ("dpm_p90", pa.float32()),
        ("structural_collapse", pa.bool_()),
        ("damage_grade", pa.dictionary(pa.int8(), pa.string())),
        # Vulnerabilidad Económica (Funciones de daño JRC / CCS)
        ("asset_value_eur", pa.float64()),
        ("damage_ratio", pa.float32()),
        ("economic_loss_eur", pa.float64()),
        # Accesibilidad y Resiliencia Territorial (Módulo 3 - Grafos)
        ("time_to_isolation_min", pa.float32()),
        ("target_hospital", pa.string()),
        ("is_isolated", pa.bool_()),
        ("triage_priority", pa.dictionary(pa.int8(), pa.string())),
    ])

    @staticmethod
    def depth_damage_curve_ccs(h_depth: np.ndarray) -> np.ndarray:
        """
        Curva de vulnerabilidad física profundidad-daño para edificación urbana
        (Consorcio de Compensación de Seguros / JRC Europa):
        - h <= 0.05 m: Sin afección estructural (0%)
        - h = 0.50 m: Afección a plantas bajas (~25%)
        - h = 1.50 m: Daño severo instalaciones y bienes (~65%)
        - h >= 3.00 m: Daño catastrófico / ruina económica (~95-100%)
        """
        ratio = np.zeros_like(h_depth, dtype=np.float32)
        valid = h_depth > 0.05
        # Progresión logística normalizada
        ratio[valid] = 1.0 / (1.0 + np.exp(-1.8 * (h_depth[valid] - 1.2)))
        return np.clip(ratio, 0.0, 1.0)

    @classmethod
    def fuse_operational_modules(
        cls,
        insar_parquet_path: str,
        fno_huv_grid: np.ndarray,
        fno_grid_bounds: Tuple[float, float, float, float],
        tti_network_report: Dict[str, Dict[str, Any]],
        output_parquet_path: str,
    ) -> None:
        """
        Fusión real de artefactos de producción:
        Cruza las parcelas de M1 con la malla de M2 y los tiempos de aislamiento de M3.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)

        table_m1 = pq.read_table(insar_parquet_path)
        df = table_m1.to_pandas()
        n_records = len(df)

        x = df["x_coord"].to_numpy()
        y = df["y_coord"].to_numpy()

        # Muestreo espacial sobre la matriz de predicción del FNO
        min_x, min_y, max_x, max_y = fno_grid_bounds
        h_dim, w_dim = fno_huv_grid.shape[-2:]

        cols = np.clip(((x - min_x) / (max_x - min_x) * w_dim).astype(int), 0, w_dim - 1)
        rows = np.clip(((max_y - y) / (max_y - min_y) * h_dim).astype(int), 0, h_dim - 1)

        depths = fno_huv_grid[0, rows, cols].astype(np.float32)
        u_vel = fno_huv_grid[1, rows, cols].astype(np.float32)
        v_vel = fno_huv_grid[2, rows, cols].astype(np.float32)
        velocities = np.hypot(u_vel, v_vel).astype(np.float32)
        hazard_vh = (depths * velocities).astype(np.float32)

        # Pérdidas económicas
        damage_ratios = cls.depth_damage_curve_ccs(depths)
        assets = df.get("asset_value_eur", np.full(n_records, 185000.0)).to_numpy()
        losses = (assets * damage_ratios).astype(np.float64)

        # Enlace con Módulo 3 (TTI y aislamiento)
        mun_names = df.get("municipality", np.full(n_records, "Paiporta")).astype(str).to_numpy()
        tti_values = np.full(n_records, np.nan, dtype=np.float32)
        target_hosps = np.full(n_records, "HOSPITAL_LA_FE", dtype=object)
        is_isolated = np.zeros(n_records, dtype=bool)

        for mun, report in tti_network_report.items():
            mask = np.char.upper(mun_names) == mun.replace("N_", "").upper()
            t_cut = report.get("t_cut_min", np.nan)
            tti_values[mask] = t_cut if np.isfinite(t_cut) else np.nan
            target_hosps[mask] = report.get("best_hospital") or "INACCESIBLE"
            is_isolated[mask] = np.isfinite(t_cut)

        # Prioridad de Triaje (P1 a P4)
        triage = np.full(n_records, "P4_BAJA", dtype=object)
        p3_mask = (depths >= 0.30) | (damage_ratios >= 0.20)
        p2_mask = (depths >= 0.80) | (hazard_vh >= 0.50) | is_isolated
        p1_mask = (df["structural_collapse"].to_numpy()) | (hazard_vh >= 1.5) | (df["is_critical_infra"].to_numpy() & is_isolated)

        triage[p3_mask] = "P3_MODERADA"
        triage[p2_mask] = "P2_ALTA"
        triage[p1_mask] = "P1_CRITICA"

        arrays = [
            pa.array(df["parcel_id"].astype(str), type=pa.string()),
            pa.array(mun_names).dictionary_encode(),
            pa.array(x, type=pa.float64()),
            pa.array(y, type=pa.float64()),
            pa.array(df["is_critical_infra"].to_numpy(), type=pa.bool_()),
            pa.array(depths, type=pa.float32()),
            pa.array(velocities, type=pa.float32()),
            pa.array(hazard_vh, type=pa.float32()),
            pa.array(np.full(n_records, 45.0, dtype=np.float32), type=pa.float32()),
            pa.array(df["dpm_p90"].to_numpy().astype(np.float32), type=pa.float32()),
            pa.array(df["structural_collapse"].to_numpy(), type=pa.bool_()),
            pa.array(df["damage_grade"].astype(str)).dictionary_encode(),
            pa.array(assets, type=pa.float64()),
            pa.array(damage_ratios, type=pa.float32()),
            pa.array(losses, type=pa.float64()),
            pa.array(tti_values, type=pa.float32()),
            pa.array(target_hosps, type=pa.string()),
            pa.array(is_isolated, type=pa.bool_()),
            pa.array(triage).dictionary_encode(),
        ]

        table = pa.Table.from_arrays(arrays, schema=cls.SCHEMA)
        pq.write_table(table, output_parquet_path, compression="snappy")
        print(f"[CONTRATO PARQUET] Fusión completada: {n_records} inmuebles consolidados en {output_parquet_path}")

    @classmethod
    def generate_synthetic_benchmark(
        cls, output_path: str = "data/processed/flood_damage_matrix.parquet", n_records: int = 5000
    ) -> None:
        """Genera el dataset representativo asegurando compatibilidad 1:1 con el contrato."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.random.seed(46)

        mun_choices = ["Paiporta", "Catarroja", "Sedaví", "Massanassa", "Picanya", "Benetússer", "Alfafar"]
        mun_probs = [0.25, 0.20, 0.15, 0.12, 0.10, 0.10, 0.08]
        municipalities = np.random.choice(mun_choices, size=n_records, p=mun_probs)

        parcel_ids = [f"46{np.random.randint(100, 999)}A{i:05d}" for i in range(n_records)]
        x_coords = np.random.uniform(720000.0, 726500.0, n_records)
        y_coords = np.random.uniform(4365000.0, 4371000.0, n_records)

        # Distancia al eje de la Rambla del Poyo
        dist_to_rambla = np.abs((y_coords - 4368000.0) - 0.4 * (x_coords - 722000.0))
        max_depth_m = np.clip(3.2 * np.exp(-dist_to_rambla / 800.0) + np.random.normal(0, 0.05, n_records), 0.0, 4.2).astype(np.float32)
        max_velocity_ms = np.clip(2.6 * (max_depth_m / 3.0) + np.random.normal(0, 0.1, n_records), 0.0, 3.5).astype(np.float32)
        hazard_vh = (max_depth_m * max_velocity_ms).astype(np.float32)

        # InSAR DPM
        dpm_p90 = np.clip(0.15 + 0.22 * max_depth_m + np.random.normal(0, 0.06, n_records), 0.0, 1.0).astype(np.float32)
        structural_collapse = dpm_p90 >= 0.40

        damage_grade = np.full(n_records, "NONE", dtype=object)
        damage_grade[dpm_p90 >= 0.20] = "MODERATE"
        damage_grade[dpm_p90 >= 0.40] = "SEVERE"
        damage_grade[dpm_p90 >= 0.60] = "COLLAPSE"

        # Vulnerabilidad económica
        asset_values = np.random.lognormal(mean=12.1, sigma=0.45, size=n_records)
        damage_ratios = cls.depth_damage_curve_ccs(max_depth_m)
        economic_losses = asset_values * damage_ratios

        # Aislamiento vial (TTI)
        time_to_inund_min = np.where(
            max_depth_m > 0.30,
            np.clip(180.0 - (x_coords - 720000.0) / 50.0 + np.random.normal(0, 8, n_records), 25.0, 240.0),
            np.nan,
        ).astype(np.float32)

        is_isolated = np.isin(municipalities, ["Paiporta", "Picanya", "Sedaví"]) & (max_depth_m > 1.2)
        tti_min = np.where(is_isolated, np.random.uniform(20.0, 45.0, n_records), np.nan).astype(np.float32)

        is_critical = np.random.choice([True, False], size=n_records, p=[0.03, 0.97])

        # Triaje de Protección Civil
        triage = np.full(n_records, "P4_BAJA", dtype=object)
        triage[(max_depth_m >= 0.30) | (damage_ratios >= 0.20)] = "P3_MODERADA"
        triage[(max_depth_m >= 0.80) | (hazard_vh >= 0.50) | is_isolated] = "P2_ALTA"
        triage[structural_collapse | (hazard_vh >= 1.5) | (is_critical & is_isolated)] = "P1_CRITICA"

        arrays = [
            pa.array(parcel_ids, type=pa.string()),
            pa.array(municipalities).dictionary_encode(),
            pa.array(x_coords, type=pa.float64()),
            pa.array(y_coords, type=pa.float64()),
            pa.array(is_critical, type=pa.bool_()),
            pa.array(max_depth_m, type=pa.float32()),
            pa.array(max_velocity_ms, type=pa.float32()),
            pa.array(hazard_vh, type=pa.float32()),
            pa.array(time_to_inund_min, type=pa.float32()),
            pa.array(dpm_p90, type=pa.float32()),
            pa.array(structural_collapse, type=bool),
            pa.array(damage_grade).dictionary_encode(),
            pa.array(asset_values, type=pa.float64()),
            pa.array(damage_ratios, type=pa.float32()),
            pa.array(economic_losses, type=pa.float64()),
            pa.array(tti_min, type=pa.float32()),
            pa.array(np.full(n_records, "HOSPITAL_LA_FE", dtype=object), type=pa.string()),
            pa.array(is_isolated, type=pa.bool_()),
            pa.array(triage).dictionary_encode(),
        ]

        table = pa.Table.from_arrays(arrays, schema=cls.SCHEMA)
        pq.write_table(table, output_path, compression="snappy")
        print(f"[CONTRATO PARQUET] Benchmark sintético generado: {n_records} registros en '{output_path}'")


if __name__ == "__main__":
    UnifiedDamageMatrixIntegrator.generate_synthetic_benchmark()
```
