import numpy as np
import math
from numba import cuda
from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None
# ==============================================================================
# 1. SETTAGGI UTENTE
# ==============================================================================
CAM_DIST = 20.0       
FOV_DEG  = 50.0       
TEXTURE_ROTATION = 3.0 

# --- NUOVO PARAMETRO ---
USE_MIRRORING = False   # True = Specchia la texture (nasconde la riga)
                       # False = Wrapping normale (Immagine originale a 360°)

# ==============================================================================
# 2. GESTIONE TEXTURE
# ==============================================================================
def get_texture(path, w=2048, h=1024):
    if os.path.exists(path):
        print(f"Caricamento texture: {path}")
        img = Image.open(path).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float64) / 255.0
    else:
        print("Texture non trovata. Generazione procedurale.")
        arr = np.zeros((h, w, 3), dtype=np.float64)
        for y in range(h):
            for x in range(w):
                # Gradiente per testare il wrapping
                arr[y, x] = [x/w, y/h, 0.5]
    return np.ascontiguousarray(arr)

# ==============================================================================
# 3. CAMPIONAMENTO TEXTURE (MIRROR + BRIGHTNESS FIX)
# ==============================================================================
@cuda.jit(fastmath=True)
def sample_bilinear(u, v, tex, use_mirror):
    h, w, _ = tex.shape
    
    if use_mirror:
        
        u_frac = u - math.floor(u)
        u_eff = 1.0 - abs(2.0 * u_frac - 1.0)
    else:
        
        u_eff = u % 1.0
        if u_eff < 0: u_eff += 1.0

    # --- 2. COORDINATE PIXEL ---
    # Centriamo il campionamento (-0.5)
    x = u_eff * w - 0.5
    y = v * h - 0.5
    
    # Floor per trovare il pixel in alto a sinistra
    x_floor = math.floor(x)
    y_floor = math.floor(y)
    
    x0 = int(x_floor)
    y0 = int(y_floor)
    
    # --- 3. PESI  ---
    # Calcoliamo il peso sulla coordinata CONTINUA, non sull'indice wrappato.
    # Questo risolve il problema dell'immagine più scura da una parte che prima riscontravo
    wx = x - x_floor
    wy = y - y_floor
    
    if use_mirror:
        # Clamp (rimbalza sui bordi)
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

@cuda.jit(fastmath=True)
def get_sky_color(phi, theta, tex, rotation_offset, mirror_mode):
    # Safety Check
    if not (math.isfinite(phi) and math.isfinite(theta)):
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    phi_rotated = phi + rotation_offset
    
    phi_norm = math.atan2(math.sin(phi_rotated), math.cos(phi_rotated)) 
    
    
    u = (phi_norm + math.pi) / (2.0 * math.pi)
    
    
    theta_safe = max(0.0, min(math.pi, theta % (2*math.pi)))
    v = theta_safe / math.pi
    
    return sample_bilinear(u, v, tex, mirror_mode)

# ==============================================================================
# 4. Integratore (RK4)
# ==============================================================================
@cuda.jit(fastmath=True)
def derivatives(state):
    r, theta, phi, ur, uth, uph = state
    rs = 1.0
    r_rs = r - rs
    
    # Protezione divisione zero (Orizzonte)
    if r_rs < 1e-6: r_rs = 1e-6
    
    # Protezione divisione zero (Poli Sferici)
    sin_th = math.sin(theta)
    # Se siamo troppo vicini al polo, clampiamo il seno per evitare esplosioni
    if abs(sin_th) < 1e-6: 
        sin_th = 1e-6 if sin_th >= 0 else -1e-6
    
    cos_th = math.cos(theta)
    
    acc_r = (rs * r_rs)/(2*r**3) - (rs/(2*r*r_rs))*ur**2 + r_rs*uth**2 + r_rs*(sin_th**2)*uph**2
    acc_th = (-2.0/r)*ur*uth + sin_th*cos_th*uph**2
    acc_ph = (-2.0/r)*ur*uph - 2.0*(cos_th/sin_th)*uth*uph
    
    return np.array([ur, uth, uph, acc_r, acc_th, acc_ph], dtype=np.float64)

@cuda.jit(fastmath=True)
def rk4_step(state, h):
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5*h*k1)
    k3 = derivatives(state + 0.5*h*k2)
    k4 = derivatives(state + h*k3)
    return state + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ==============================================================================
# 5. RENDERING (STEP ADATTIVO POLARE)
# ==============================================================================
@cuda.jit(parallel=True)
def render(W, H, tex, cam_dist, fov, tex_rot, mirror_mode):
    img = np.zeros((H, W, 3), dtype=np.float64)
    rs = 1.0
    
    scale = math.tan(math.radians(fov * 0.5))
    aspect = W / H
    cam_pos_sph = np.array([cam_dist, math.pi/2.0, 0.0])
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
            
            # Aumentato max_steps per compensare i passi piccoli
            for step in range(5000):
                r_curr = state[0]
                theta_curr = state[1]
                
                if r_curr < 1.01 * rs: break
                if r_curr > 100.0:
                    pixel = get_sky_color(state[2], state[1], tex, tex_rot, mirror_mode)
                    break
                
                # --- CALCOLO STEP ADATTIVO ---
                # 1. Base sulla distanza (lontano = veloce)
                h = 0.1 * (r_curr - rs)
                
                # 2. Controllo vicinanza Poli (Singolarità coordinate)
                # Se theta è vicino a 0 o PI, sin(theta) è quasi 0.
                sin_theta = abs(math.sin(theta_curr))
                
                # Rallenta fino al 2% della velocità normale se siamo al polo
                pole_factor = max(0.02, sin_theta)
                
                # Applica fattore polare
                h = h * pole_factor
                
                # 3. Clamp finale (Mai sotto 1e-6, mai sopra 2.0)
                h = max(1e-6, min(2.0, h))
                
                state = rk4_step(state, h)
            
            img[y, x] = pixel
            
    return img

# ==============================================================================
# 6. MAIN
# ==============================================================================
def main():
    W, H = 3840, 2160
    
    tex = get_texture("nasa3.jpg", w=2048, h=1024)
    
    print(f"Rendering (Mirroring={USE_MIRRORING}, Dist={CAM_DIST})...")
    
    # Passiamo il booleano mirror_mode
    img_64 = render(W, H, tex, CAM_DIST, FOV_DEG, TEXTURE_ROTATION, USE_MIRRORING)
    
    img_final = np.clip(img_64 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_final).save("blackhole_final.png")
    print("Fatto.")

if __name__ == "__main__":
    main()