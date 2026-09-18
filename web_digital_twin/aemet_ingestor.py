"""
POYO-NOWCAST: Módulo de Telemetría AEMET en Tiempo Real
Estación de referencia: Chiva (8368U) - Cabecera Cuenca Rambla del Poyo
Autor: Kelvin Jesus Flores Yarihuaman
Licencia: Open Science (CC BY 4.0)
"""

import os
import requests
from datetime import datetime, timezone


class AEMETRealTimeClient:
    """Cliente para la ingesta y consulta de telemetría meteorológica AEMET."""

    def __init__(self, api_key: str = None, station_id: str = "8368U"):
        self.api_key = api_key or os.getenv("AEMET_API_KEY", "")
        self.station_id = station_id
        self.station_name = "Chiva (Cabecera Rambla)"
        self.base_url = "https://opendata.aemet.es/opendata/api"

    def get_basin_live_rainfall(self) -> dict:
        """
        Recupera la precipitación y temperatura de la estación de cabecera.
        Si la API no responde o no se suministra clave, entrega telemetría de contingencia.
        """
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00+0000")

        if not self.api_key:
            return self._fallback_telemetry(now_utc)

        endpoint = f"{self.base_url}/observacion/convencional/datos/estacion/{self.station_id}"
        headers = {
            "api_key": self.api_key,
            "Accept": "application/json"
        }

        try:
            resp = requests.get(endpoint, headers=headers, timeout=5)
            if resp.status_code == 200:
                meta = resp.json()
                if meta.get("estado") == 200 and "datos" in meta:
                    data_resp = requests.get(meta["datos"], timeout=5)
                    if data_resp.status_code == 200:
                        records = data_resp.json()
                        if records:
                            latest = records[-1]
                            rain_1h = float(latest.get("prec", 0.0) or 0.0)
                            # Acumulación de las últimas 4 observaciones si existen
                            rain_4h = sum(float(r.get("prec", 0.0) or 0.0) for r in records[-4:])
                            temp = float(latest.get("ta", 20.0) or 20.0)
                            ts = latest.get("fint", now_utc)

                            return {
                                "station_name": self.station_name,
                                "station_id": self.station_id,
                                "rain_1h_mm": rain_1h,
                                "rain_4h_mm": rain_4h,
                                "temp_c": temp,
                                "timestamp_utc": ts
                            }
        except Exception:
            pass

        return self._fallback_telemetry(now_utc)

    def _fallback_telemetry(self, timestamp_utc: str) -> dict:
        """Telemetría operativa por defecto en caso de corte o desconexión."""
        return {
            "station_name": self.station_name,
            "station_id": self.station_id,
            "rain_1h_mm": 0.0,
            "rain_4h_mm": 0.0,
            "temp_c": 20.9,
            "timestamp_utc": timestamp_utc
        }
