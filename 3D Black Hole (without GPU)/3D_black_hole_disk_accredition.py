import numpy as np
import math
import numba
from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None
# ==============================================================================
# 1. SETTAGGI UTENTE
# ==============================================================================
CAM_DIST = 20.0
FOV_DEG  = 50.0
TEXTURE_ROTATION = 3.0
USE_MIRRORING = False

# Parametri Disco
DISK_R_MIN = 2.6
DISK_R_MAX = 20.0

# ==============================================================================
# 2. GESTIONE TEXTURE CIELO
# ==============================================================================
def get_sky_texture(path, w=2048, h=1024):
    if os.path.exists(path):
        print(f"Caricamento texture cielo: {path}")
        img = Image.open(path).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float64) / 255.0
    else:
        print("Texture cielo non trovata. Generazione cielo base.")
        arr = np.zeros((h, w, 3), dtype=np.float64)
        for y in range(h):
            for x in range(w):
                arr[y, x] = [x/w, y/h, 0.5]
    return np.ascontiguousarray(arr)

# ==============================================================================
# 3. CAMPIONAMENTO TEXTURE CIELO
# ==============================================================================
@numba.njit(fastmath=True)
def sample_bilinear(u, v, tex, use_mirror):
    h, w, _ = tex.shape

    if use_mirror:
        u_frac = u - math.floor(u)
        u_eff = 1.0 - abs(2.0 * u_frac - 1.0)
    else:
        u_eff = u % 1.0
        if u_eff < 0: u_eff += 1.0

    x = u_eff * w - 0.5
    y = v * h - 0.5

    x_floor = math.floor(x)
    y_floor = math.floor(y)
    x0, y0 = int(x_floor), int(y_floor)

    wx = x - x_floor
    wy = y - y_floor

    if use_mirror:
        x0_idx = max(0, min(w - 1, x0))
        x1_idx = max(0, min(w - 1, x0 + 1))
    else:
        x0_idx = (x0 % w + w) % w
        x1_idx = ((x0 + 1) % w + w) % w

    y0_idx = max(0, min(h - 1, y0))
    y1_idx = max(0, min(h - 1, y0 + 1))

    top = tex[y0_idx, x0_idx] * (1.0 - wx) + tex[y0_idx, x1_idx] * wx
    bot = tex[y1_idx, x0_idx] * (1.0 - wx) + tex[y1_idx, x1_idx] * wx

    return top * (1.0 - wy) + bot * wy

@numba.njit(fastmath=True)
def get_sky_color(phi, theta, tex, rotation_offset, mirror_mode):
    if not (math.isfinite(phi) and math.isfinite(theta)):
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    phi_rotated = phi + rotation_offset
    phi_norm = math.atan2(math.sin(phi_rotated), math.cos(phi_rotated))
    u = (phi_norm + math.pi) / (2.0 * math.pi)
    theta_safe = max(0.0, min(math.pi, theta % (2*math.pi)))
    v = theta_safe / math.pi
    return sample_bilinear(u, v, tex, mirror_mode)

# ==============================================================================
# 4. INTEGRATORE FISICO (RK4)
# ==============================================================================
@numba.njit(fastmath=True)
def derivatives(state):
    r, theta, phi, ur, uth, uph = state
    rs = 1.0
    r_rs = r - rs
    if r_rs < 1e-6: r_rs = 1e-6

    sin_th = math.sin(theta)
    if abs(sin_th) < 1e-6:
        sin_th = 1e-6 if sin_th >= 0 else -1e-6

    cos_th = math.cos(theta)

    acc_r = (rs * r_rs)/(2*r**3) - (rs/(2*r*r_rs))*ur**2 + r_rs*uth**2 + r_rs*(sin_th**2)*uph**2
    acc_th = (-2.0/r)*ur*uth + sin_th*cos_th*uph**2
    acc_ph = (-2.0/r)*ur*uph - 2.0*(cos_th/sin_th)*uth*uph

    return np.array([ur, uth, uph, acc_r, acc_th, acc_ph], dtype=np.float64)

@numba.njit(fastmath=True)
def rk4_step(state, h):
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5*h*k1)
    k3 = derivatives(state + 0.5*h*k2)
    k4 = derivatives(state + h*k3)
    return state + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ==============================================================================
# 5. RENDERING 3D (INTERPOLAZIONE ESATTA DEL PIANO)
# ==============================================================================
@numba.njit(parallel=True)
def render(W, H, sky_tex, cam_dist, fov, tex_rot, mirror_mode):
    img = np.zeros((H, W, 3), dtype=np.float64)
    rs = 1.0
    cam_theta = math.radians(82.0)

    scale = math.tan(math.radians(fov * 0.5))
    aspect = W / H
    cam_pos_sph = np.array([cam_dist, cam_theta, 0.0])
    lapse = math.sqrt(max(0.0, 1.0 - rs / cam_dist))

    for y in numba.prange(H):
        for x in range(W):

            u_ndc = (2.0 * (x + 0.5) / W - 1.0) * aspect * scale
            v_ndc = (1.0 - 2.0 * (y + 0.5) / H) * scale

            ray_len = math.sqrt(u_ndc**2 + v_ndc**2 + 1.0)
            d_right = u_ndc / ray_len
            d_up    = v_ndc / ray_len
            d_fwd   = 1.0 / ray_len

            ur = -d_fwd * lapse
            uth = -d_up / cam_dist
            uph = d_right / cam_dist

            state = np.array([cam_dist, cam_pos_sph[1], cam_pos_sph[2], ur, uth, uph], dtype=np.float64)
            pixel = np.zeros(3, dtype=np.float64)
            accumulated_glow = np.zeros(3, dtype=np.float64)

            transmittance = 1.0

            # Salviamo lo stato precedente PRIMA di iniziare i salti
            prev_r = state[0]
            prev_phi = state[2]
            prev_cos_theta = math.cos(state[1])

            for step in range(5000):
                r_curr = state[0]
                theta_curr = state[1]

                if r_curr < 1.01 * rs: break
                if r_curr > 100.0:
                    pixel = get_sky_color(state[2], state[1], sky_tex, tex_rot, mirror_mode)
                    break

                h = 0.1 * (r_curr - rs)
                sin_theta = abs(math.sin(theta_curr))
                h = max(1e-6, min(2.0, h * max(0.02, sin_theta)))

                # Salviamo la posizione di partenza di questo salto
                prev_r = state[0]
                prev_phi = state[2]

                state = rk4_step(state, h)
                curr_cos_theta = math.cos(state[1])

                # SE ATTRAVERSO IL DISCO DI ACCRESCIMENTO
                if prev_cos_theta * curr_cos_theta <= 0.0:
                    # --- FIX: INTERPOLAZIONE LINEARE PER TROVARE IL PUNTO ESATTO ---
                    # Calcoliamo a quale percentuale 't' del salto (tra 0.0 e 1.0) il raggio ha toccato l'equatore (cos_theta = 0)
                    t = prev_cos_theta / (prev_cos_theta - curr_cos_theta)

                    # Troviamo r e phi esatti in base alla percentuale t
                    r_cross = prev_r + t * (state[0] - prev_r)
                    phi_cross = prev_phi + t * (state[2] - prev_phi)

                    if DISK_R_MIN < r_cross < DISK_R_MAX:
                        r_frac = (r_cross - DISK_R_MIN) / (DISK_R_MAX - DISK_R_MIN)

                        swirl = phi_cross + r_frac * 12.0
                        noise = math.sin(swirl * 5.0) * 0.5 + 0.5
                        noise += math.sin(swirl * 20.0 + phi_cross) * 0.25
                        noise = max(0.0, min(1.0, noise))

                        temp = 1.0 - r_frac
                        r_col = min(1.0, temp * 1.5 + noise * 0.6)
                        g_col = min(1.0, temp * 0.8 + noise * 0.4)
                        b_col = min(1.0, temp * 0.3 + noise * 0.1)

                        disk_color = np.array([r_col, g_col, b_col], dtype=np.float64)

                        density = 3.0 / (r_cross - 1.0)**1.2
                        edge_fade = math.sqrt(max(0.0, 1.0 - r_frac))
                        opacity = min(1.0, density * edge_fade)

                        glow_intensity = opacity * 2.5
                        accumulated_glow += disk_color * glow_intensity * transmittance

                        transmittance *= (1.0 - opacity)

                        if transmittance < 0.01:
                            break

                prev_cos_theta = curr_cos_theta

            final_color = (pixel * transmittance) + accumulated_glow
            img[y, x] = np.array([min(1.0, final_color[0]), min(1.0, final_color[1]), min(1.0, final_color[2])])

    return img

# ==============================================================================
# 6. MAIN
# ==============================================================================
def main():
    W, H = 3840, 2160 # Alzalo a 3840x2160 per dettagli pazzeschi

    sky_tex = get_sky_texture("nasa3.jpg", w=2048, h=1024)

    print(f"Rendering (Shader Procedurale Infinito, Dist={CAM_DIST})...")
    img_64 = render(W, H, sky_tex, CAM_DIST, FOV_DEG, TEXTURE_ROTATION, USE_MIRRORING)

    img_final = np.clip(img_64 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_final).save("3d_smooth_blackhole.png")
    print("Fatto. Controlla il file 3d_smooth_blackhole.png")

if __name__ == "__main__":
    main()
