"""
POYO-NOWCAST: Módulo 2 - Hidráulica Neuronal 2D (Fourier Neural Operator)
Tecnología: Python (PyTorch, Tensores Complejos) + Física Informada (PINN)
Arquitectura: Domain-Padded SpectralConv2d -> Residual Fourier Blocks ->
              Wet-Dry Velocity Regularization -> Full 2D Saint-Venant Loss Engine.

Autor: Kelvin Jesus Flores Yarihuaman (https://www.linkedin.com/in/kelvinflores-ingenieria)
Metasistema: YGGDRASIL-OS | Framework de Resiliencia Climática

LICENCIAMIENTO MODULAR:
- Documentación y Memoria Técnica: CC BY-NC-SA 4.0
- Código Fuente y Algoritmos: GNU GPLv3
- Datasets y Tablas Parquet: ODC-By
"""

from typing import Tuple, Dict
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    Capa de convolución espectral 2D con inicialización de varianza preservada
    y contracción tensorial de modos dominantes.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Inicialización de varianza preservada en dominio complejo: 1 / sqrt(in_channels)
        scale = 1.0 / (in_channels ** 0.5)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _complex_mult2d(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # Contracción espectral: (batch, in, y, x) x (in, out, y, x) -> (batch, out, y, x)
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        h_dim, w_dim = x.size(-2), x.size(-1)

        # 1. Transformada de Fourier Real 2D hacia espacio espectral
        x_ft = torch.fft.rfft2(x)

        # 2. Asignación del espacio latente espectral truncado
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            h_dim,
            w_dim // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, h_dim // 2)
        m2 = min(self.modes2, (w_dim // 2 + 1))

        # Modos de frecuencia positiva (cuadrante superior)
        out_ft[:, :, :m1, :m2] = self._complex_mult2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        # Modos de frecuencia negativa (cuadrante inferior / reflejo conjugado)
        out_ft[:, :, -m1:, :m2] = self._complex_mult2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )

        # 3. Transformada Inversa conservando la resolución original
        return torch.fft.irfft2(out_ft, s=(h_dim, w_dim))


class FNO2d(nn.Module):
    """
    Fourier Neural Operator 2D con extensión de dominio (Anti-Gibbs padding)
    y regularización hidrodinámica estricta para frentes secos/mojados.
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        modes1: int = 16,
        modes2: int = 16,
        width: int = 48,
        num_blocks: int = 4,
        padding: int = 8,
        h_threshold_dry: float = 0.005,  # 5 mm límite de celda seca
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.padding = padding
        self.h_threshold = h_threshold_dry

        # Operador de elevación (Lifting P)
        self.p = nn.Conv2d(in_channels, width, kernel_size=1)

        # Bloques de Fourier residuales con derivaciones locales 1x1
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(num_blocks)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1) for _ in range(num_blocks)
        ])

        # Operador de proyección (Projection Q)
        self.q = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Extensión de dominio (Domain Padding) para suprimir artefactos periódicos de contorno
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding], mode="replicate")

        out = self.p(x)

        # 2. Propagación residual espacio-frecuencia
        for spec_conv, w_conv in zip(self.spectral_layers, self.ws):
            out = F.gelu(spec_conv(out) + w_conv(out))

        out = self.q(out)

        # 3. Recorte del acolchado perimetral
        if self.padding > 0:
            out = out[..., : -self.padding, : -self.padding]

        # 4. Restricciones físicas: calado no negativo (h >= 0)
        h = F.relu(out[:, 0:1, :, :])

        # Regularización cinemática: en suelo seco, u y v decaen analíticamente a cero
        raw_u = out[:, 1:2, :, :]
        raw_v = out[:, 2:3, :, :]
        
        # Factor de atenuación suave: h^2 / (h^2 + eps^2)
        wet_mask = (h ** 2) / (h ** 2 + (self.h_threshold) ** 2)
        u_reg = raw_u * wet_mask
        v_reg = raw_v * wet_mask

        return torch.cat([h, u_reg, v_reg], dim=1)


class SaintVenant2DPhysicsLoss:
    """
    Evaluador continuo del sistema completo de aguas someras 2D (Shallow Water Equations):
    - Conservación de masa (Continuidad)
    - Conservación de momento lineal en X
    - Conservación de momento lineal en Y
    Incluye empuje hidrostático, pendiente topográfica y disipación turbulenta de Manning.
    """

    def __init__(self, dx: float = 2.0, dy: float = 2.0, gravity: float = 9.81, eps_dry: float = 1e-4):
        self.dx = float(dx)
        self.dy = float(dy)
        self.g = float(gravity)
        self.eps = float(eps_dry)

    def compute_pde_residuals(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        z_b: torch.Tensor,
        n_manning: torch.Tensor,
        p_rain_net: torch.Tensor,
        dt_approx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula los residuos analíticos conservativos de Saint-Venant 2D.
        :param dt_approx: Tupla (dh_dt, duh_dt, dvh_dt) aproximados temporalmente.
        """
        dh_dt, duh_dt, dvh_dt = dt_approx

        # Caudales específicos
        uh = u * h
        vh = v * h

        # -------------------------------------------------------------
        # 1. Conservación de Masa: dh/dt + d(uh)/dx + d(vh)/dy - P_net = 0
        # -------------------------------------------------------------
        duh_dx = torch.gradient(uh, spacing=self.dx, dim=-1)[0]
        dvh_dy = torch.gradient(vh, spacing=self.dy, dim=-2)[0]
        res_mass = dh_dt + duh_dx + dvh_dy - p_rain_net

        # -------------------------------------------------------------
        # 2. Conservación de Momento X:
        # d(uh)/dt + d(u^2 h + 0.5 g h^2)/dx + d(uvh)/dy + g h dz_b/dx + g S_fx = 0
        # -------------------------------------------------------------
        flux_xx = (u ** 2) * h + 0.5 * self.g * (h ** 2)
        flux_xy = u * v * h

        dflux_xx_dx = torch.gradient(flux_xx, spacing=self.dx, dim=-1)[0]
        dflux_xy_dy = torch.gradient(flux_xy, spacing=self.dy, dim=-2)[0]
        dzb_dx = torch.gradient(z_b, spacing=self.dx, dim=-1)[0]

        vel_norm = torch.hypot(u, v)
        # Término de fondo Manning regularizado: g * n^2 * u * |u| / (h^(1/3))
        friction_x = (self.g * (n_manning ** 2) * u * vel_norm) / (h ** (1.0 / 3.0) + self.eps)

        res_mom_x = duh_dt + dflux_xx_dx + dflux_xy_dy + (self.g * h * dzb_dx) + friction_x

        # -------------------------------------------------------------
        # 3. Conservación de Momento Y:
        # d(vh)/dt + d(uvh)/dx + d(v^2 h + 0.5 g h^2)/dy + g h dz_b/dy + g S_fy = 0
        # -------------------------------------------------------------
        flux_yy = (v ** 2) * h + 0.5 * self.g * (h ** 2)

        dflux_xy_dx = torch.gradient(flux_xy, spacing=self.dx, dim=-1)[0]
        dflux_yy_dy = torch.gradient(flux_yy, spacing=self.dy, dim=-2)[0]
        dzb_dy = torch.gradient(z_b, spacing=self.dy, dim=-2)[0]

        friction_y = (self.g * (n_manning ** 2) * v * vel_norm) / (h ** (1.0 / 3.0) + self.eps)

        res_mom_y = dvh_dt + dflux_xy_dx + dflux_yy_dy + (self.g * h * dzb_dy) + friction_y

        # Máscara de validación (solo penaliza celdas con soporte húmedo real)
        wet_domain = (h > self.eps).float()

        loss_mass = torch.mean((res_mass * wet_domain) ** 2)
        loss_mom_x = torch.mean((res_mom_x * wet_domain) ** 2)
        loss_mom_y = torch.mean((res_mom_y * wet_domain) ** 2)

        return {
            "loss_mass": loss_mass,
            "loss_momentum_x": loss_mom_x,
            "loss_momentum_y": loss_mom_y,
            "total_physics_loss": loss_mass + 0.1 * (loss_mom_x + loss_mom_y),
        }


def generate_synthetic_poyo_domain(
    batch_size: int = 1, height: int = 256, width: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Genera un dominio coherente con la topografía de la Rambla del Poyo:
    Gradiente regional oeste-este (350m a 15m) con cauce inciso en V.
    """
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij"
    )

    # 1. Topografía: Pendiente regional + cauce sinuoso central excavado
    z_regional = 350.0 * (1.0 - grid_x) + 15.0
    channel_axis = 0.5 + 0.08 * torch.sin(grid_x * 6.28)
    d_channel = torch.abs(grid_y - channel_axis)
    z_channel = -12.0 * torch.exp(-(d_channel ** 2) / 0.002)
    z_b = z_regional + z_channel

    # 2. Rugosidad de Manning: cauce liso (0.030) vs llanura urbana (0.065)
    n_manning = torch.where(d_channel < 0.04, 0.030, 0.065)

    # 3. Forzamiento pluviométrico convectivo (DANA extrema en cabecera)
    p_rain = 0.00025 * torch.exp(-((grid_x - 0.15) ** 2 + (grid_y - 0.5) ** 2) / 0.02)

    features = torch.stack([z_b, n_manning, p_rain, grid_x, grid_y], dim=0)
    input_tensor = features.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    target_huv = torch.zeros(batch_size, 3, height, width)
    return input_tensor, target_huv


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[POYO-NOWCAST] Dispositivo de cómputo: {device}")

    # 1. Configuración del Operador Neuronal
    model = FNO2d(
        in_channels=5,
        out_channels=3,
        modes1=16,
        modes2=16,
        width=48,
        num_blocks=4,
        padding=8,
    ).to(device)

    # 2. Ingesta del dominio topográfico de la Rambla del Poyo
    x_in, _ = generate_synthetic_poyo_domain(batch_size=1, height=256, width=256)
    x_in = x_in.to(device)

    # 3. Calentamiento (Warm-up)
    with torch.no_grad():
        for _ in range(5):
            _ = model(x_in)

    # 4. Benchmark de Inferencia Operativa en Tiempo Real
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        sol = model(x_in)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inf_ms = (time.perf_counter() - t0) * 1000.0

    # 5. Validación de Pérdida Física Saint-Venant
    pde_engine = SaintVenant2DPhysicsLoss(dx=2.0, dy=2.0)
    h_pred = sol[:, 0:1]
    u_pred = sol[:, 1:2]
    v_pred = sol[:, 2:3]

    # Derivadas temporales dummy para prueba de paso estático
    dh_dt = torch.zeros_like(h_pred)
    duh_dt = torch.zeros_like(u_pred)
    dvh_dt = torch.zeros_like(v_pred)

    pde_metrics = pde_engine.compute_pde_residuals(
        h=h_pred,
        u=u_pred,
        v=v_pred,
        z_b=x_in[:, 0:1],
        n_manning=x_in[:, 1:2],
        p_rain_net=x_in[:, 2:3],
        dt_approx=(dh_dt, duh_dt, dvh_dt),
    )

    print(f"[LATENCIA] Inferencia FNO 2D (256x256): {t_inf_ms:.2f} ms")
    print(f"[ESTADO SLA] {'APROBADO (<100 ms)' if t_inf_ms < 100.0 else 'RECHAZADO'}")
    print(f"[FÍSICA] Residuo Masa: {pde_metrics['loss_mass'].item():.2e} | "
          f"Momento X: {pde_metrics['loss_momentum_x'].item():.2e} | "
          f"Momento Y: {pde_metrics['loss_momentum_y'].item():.2e}")
