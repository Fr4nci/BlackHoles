import numpy as np
import cupy as cp
import math
import gc
import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ==============================================================================
# 1. SETTAGGI UTENTE
# ==============================================================================
CAM_DIST         = 20.0
FOV_DEG          = 50.0
TEXTURE_ROTATION = 3.0
USE_MIRRORING    = False
DISK_R_MIN       = 2.6
DISK_R_MAX       = 20.0

W, H             = 640, 360      # risoluzione output — alza a 1920x1080 per produzione
TEX_W, TEX_H     = 2048, 1024    # risoluzione texture cielo

# ==============================================================================
# 2. KERNEL CUDA C (compilato da CuPy con nvcc — funziona con qualsiasi driver)
#
# Vantaggi rispetto a Numba:
#   - Nessuna dipendenza da cudatoolkit separato: usa il driver già installato
#   - Il codice CUDA C viene compilato una volta sola e cachato su disco
#   - Accesso diretto a tutte le istruzioni PTX e agli intrinsic CUDA
#   - Nessun problema di compatibilità versione Numba/CUDA
# ==============================================================================

KERNEL_CODE = r"""
#include <math_constants.h>   // CUDART_PI_F, ecc.
// ============================================================
// DEVICE HELPERS
// ============================================================

__device__ __forceinline__
float3 sample_bilinear(
    const float* __restrict__ tex,
    int tex_h, int tex_w,
    float u, float v,
    bool mirror_mode)
{
    // wrap U in [0,1]
    u = u - floorf(u);

    // wrap / mirror V
    float vf = floorf(v);
    float vr = v - vf;
    if (mirror_mode && ((int)vf % 2 == 1))
        vr = 1.0f - vr;
    v = vr;

    float px = u * (tex_w - 1);
    float py = v * (tex_h - 1);
    int x0 = (int)px,  y0 = (int)py;
    int x1 = min(x0+1, tex_w-1);
    int y1 = min(y0+1, tex_h-1);
    float fx = px - x0, fy = py - y0;

    float w00=(1-fx)*(1-fy), w10=fx*(1-fy),
          w01=(1-fx)*fy,     w11=fx*fy;

    // layout: tex[y, x, c]  =>  index = (y*tex_w + x)*3 + c
    #define T(y,x,c) tex[((y)*tex_w+(x))*3+(c)]
    float3 col;
    col.x = T(y0,x0,0)*w00 + T(y0,x1,0)*w10 + T(y1,x0,0)*w01 + T(y1,x1,0)*w11;
    col.y = T(y0,x0,1)*w00 + T(y0,x1,1)*w10 + T(y1,x0,1)*w01 + T(y1,x1,1)*w11;
    col.z = T(y0,x0,2)*w00 + T(y0,x1,2)*w10 + T(y1,x0,2)*w01 + T(y1,x1,2)*w11;
    #undef T
    return col;
}

__device__ __forceinline__
float3 get_sky_color(
    float phi, float theta,
    const float* __restrict__ tex,
    int tex_h, int tex_w,
    float rotation_offset, bool mirror_mode)
{
    float phi_r   = phi + rotation_offset;
    float phi_n   = atan2f(sinf(phi_r), cosf(phi_r));
    float u       = (phi_n + CUDART_PI_F) / (2.0f*CUDART_PI_F);
    float th_safe = fmaxf(0.0f, fminf(CUDART_PI_F,
                          fmodf(theta, 2.0f*CUDART_PI_F)));
    float v       = th_safe / CUDART_PI_F;
    return sample_bilinear(tex, tex_h, tex_w, u, v, mirror_mode);
}

// ============================================================
// GEODESICA DI SCHWARZSCHILD  (rs = 1, unità geometrizzate)
// ============================================================
__device__ __forceinline__
void geodesic_derivs(
    float r, float theta, float phi,
    float ur, float uth, float uph,
    float& dr, float& dth, float& dphi,
    float& dur, float& duth, float& duph)
{
    const float rs  = 1.0f;
    float sin_th = sinf(theta);
    float cos_th = cosf(theta);
    float r2     = r*r, r3 = r2*r;
    float f      = 1.0f - rs/r;
    float f_inv  = 1.0f / fmaxf(f, 1e-10f);
    float sin2   = fmaxf(sin_th*sin_th, 1e-10f);

    dr   = f * ur;
    dth  = uth / r2;
    dphi = uph / (r2 * sin2);

    dur  = (-0.5f*rs/r2) * f_inv * ur*ur
           + (uph*uph/(r3*sin2) + uth*uth/r3) * f
           - 0.5f*rs*(uph*uph/(r2*sin2) + uth*uth/r2)/r2;
    duth = sin_th*cos_th*uph*uph / (r2*fmaxf(sin2*sin2, 1e-10f))
           - 2.0f*ur*uth/r;
    duph = -2.0f*(ur/r + cos_th*uth/fmaxf(fabsf(sin_th), 1e-10f))*uph;
}

__device__ __forceinline__
void rk4_step(
    float& r, float& theta, float& phi,
    float& ur, float& uth, float& uph, float h)
{
    float dr,dth,dp,dur,dut,dup;

    // k1
    geodesic_derivs(r,theta,phi,ur,uth,uph, dr,dth,dp,dur,dut,dup);
    float k1r=dr,k1t=dth,k1p=dp,k1ur=dur,k1ut=dut,k1up=dup;

    // k2
    geodesic_derivs(r+0.5f*h*k1r, theta+0.5f*h*k1t, phi+0.5f*h*k1p,
                    ur+0.5f*h*k1ur, uth+0.5f*h*k1ut, uph+0.5f*h*k1up,
                    dr,dth,dp,dur,dut,dup);
    float k2r=dr,k2t=dth,k2p=dp,k2ur=dur,k2ut=dut,k2up=dup;

    // k3
    geodesic_derivs(r+0.5f*h*k2r, theta+0.5f*h*k2t, phi+0.5f*h*k2p,
                    ur+0.5f*h*k2ur, uth+0.5f*h*k2ut, uph+0.5f*h*k2up,
                    dr,dth,dp,dur,dut,dup);
    float k3r=dr,k3t=dth,k3p=dp,k3ur=dur,k3ut=dut,k3up=dup;

    // k4
    geodesic_derivs(r+h*k3r, theta+h*k3t, phi+h*k3p,
                    ur+h*k3ur, uth+h*k3ut, uph+h*k3up,
                    dr,dth,dp,dur,dut,dup);

    float h6 = h/6.0f;
    r     += h6*(k1r  + 2*k2r  + 2*k3r  + dr);
    theta += h6*(k1t  + 2*k2t  + 2*k3t  + dth);
    phi   += h6*(k1p  + 2*k2p  + 2*k3p  + dp);
    ur    += h6*(k1ur + 2*k2ur + 2*k3ur + dur);
    uth   += h6*(k1ut + 2*k2ut + 2*k3ut + dut);
    uph   += h6*(k1up + 2*k2up + 2*k3up + dup);
}

__device__ __forceinline__
float aces_tonemap(float x)
{
    // ACES filmic tone mapping — preserva le alte luci del disco
    const float a=2.51f, b=0.03f, c=2.43f, d=0.59f, e=0.14f;
    return fmaxf(0.0f, fminf(1.0f, (x*(a*x+b)) / (x*(c*x+d)+e)));
}

// ============================================================
// KERNEL PRINCIPALE
// ============================================================
extern "C" __global__
void render_kernel(
    float* __restrict__ img_out,   // [H, W, 3]
    int W, int H,
    const float* __restrict__ sky_tex,
    int tex_h, int tex_w,
    float cam_dist, float fov_deg,
    float tex_rot, int mirror_mode,
    float time_t,
    float disk_rmin, float disk_rmax)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const float rs        = 1.0f;
    const float cam_theta = 82.0f * CUDART_PI_F / 180.0f;
    const float scale     = tanf(fov_deg * 0.5f * CUDART_PI_F / 180.0f);
    const float aspect    = (float)W / (float)H;
    const float lapse     = sqrtf(fmaxf(0.0f, 1.0f - rs/cam_dist));

    float u_ndc = (2.0f*(x+0.5f)/W - 1.0f) * aspect * scale;
    float v_ndc = (1.0f - 2.0f*(y+0.5f)/H) * scale;

    float ray_len = sqrtf(u_ndc*u_ndc + v_ndc*v_ndc + 1.0f);
    float d_right = u_ndc / ray_len;
    float d_up    = v_ndc / ray_len;
    float d_fwd   = 1.0f  / ray_len;

    float r     = cam_dist;
    float theta = cam_theta;
    float phi   = 0.0f;
    float ur    = -d_fwd  * lapse;
    float uth   = -d_up   / cam_dist;
    float uph   =  d_right / cam_dist;

    float pixel_r=0, pixel_g=0, pixel_b=0;
    float acc_r=0,   acc_g=0,   acc_b=0;
    float transmittance  = 1.0f;
    float prev_r         = r;
    float prev_phi       = phi;
    float prev_cos_theta = cosf(theta);

    for (int step = 0; step < 2500; ++step) {
        if (r < 1.01f * rs) break;
        if (r > 200.0f) {
            float3 sky = get_sky_color(phi, theta, sky_tex, tex_h, tex_w,
                                       tex_rot, (bool)mirror_mode);
            pixel_r = sky.x; pixel_g = sky.y; pixel_b = sky.z;
            break;
        }

        float sin_theta = fabsf(sinf(theta));
        float h_step    = fmaxf(1e-6f, fminf(2.0f,
                          0.1f*(r-rs)*fmaxf(0.02f, sin_theta)));

        prev_r   = r;
        prev_phi = phi;
        rk4_step(r, theta, phi, ur, uth, uph, h_step);

        float curr_cos_theta = cosf(theta);

        if (prev_cos_theta * curr_cos_theta <= 0.0f) {
            float denom    = prev_cos_theta - curr_cos_theta;
            float t_interp = (fabsf(denom) < 1e-7f) ? 0.5f
                             : prev_cos_theta / denom;
            float r_cross   = prev_r   + t_interp*(r   - prev_r);
            float phi_cross = prev_phi + t_interp*(phi - prev_phi);

            if (r_cross > disk_rmin && r_cross < disk_rmax) {
                float r_frac = (r_cross - disk_rmin) / (disk_rmax - disk_rmin);
                float anim_phi = phi_cross - time_t * (3.0f/r_cross);
                float swirl    = anim_phi + r_frac * 12.0f;
                float noise    = sinf(swirl*5.0f)*0.5f + 0.5f;
                noise         += sinf(swirl*20.0f + anim_phi*2.0f)*0.25f;
                noise          = fmaxf(0.0f, fminf(1.0f, noise));

                float temp  = 1.0f - r_frac;
                float col_r = fminf(1.0f, temp*1.5f + noise*0.6f);
                float col_g = fminf(1.0f, temp*0.8f + noise*0.4f);
                float col_b = fminf(1.0f, temp*0.3f + noise*0.1f);

                float density   = 3.0f / fmaxf(powf(r_cross-1.0f, 1.2f), 1e-6f);
                float edge_fade = sqrtf(fmaxf(0.0f, 1.0f - r_frac));
                float opacity   = fminf(1.0f, density*edge_fade);
                float glow      = opacity * 2.5f;

                acc_r += col_r * glow * transmittance;
                acc_g += col_g * glow * transmittance;
                acc_b += col_b * glow * transmittance;
                transmittance *= (1.0f - opacity);
                if (transmittance < 0.01f) break;
            }
        }
        prev_cos_theta = curr_cos_theta;
    }

    int idx = (y*W + x)*3;
    img_out[idx+0] = aces_tonemap(pixel_r*transmittance + acc_r);
    img_out[idx+1] = aces_tonemap(pixel_g*transmittance + acc_g);
    img_out[idx+2] = aces_tonemap(pixel_b*transmittance + acc_b);
}
"""

# ==============================================================================
# 3. DIAGNOSTICA GPU (con CuPy — nessun rischio segfault)
# ==============================================================================
def diagnose_gpu():
    print("=" * 60)
    print("DIAGNOSTICA GPU (CuPy)")
    print("=" * 60)
    dev = cp.cuda.Device(0)
    dev.use()
    props = cp.cuda.runtime.getDeviceProperties(0)
    name  = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    print(f"  GPU              : {name}")
    cc = (props["major"], props["minor"])
    print(f"  Compute cap.     : {cc[0]}.{cc[1]}")
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"  VRAM libera      : {free/1024**2:.0f} MB / {total/1024**2:.0f} MB")
    tex_mb = TEX_W * TEX_H * 3 * 4 / 1024**2
    out_mb = W * H * 3 * 4 / 1024**2
    print(f"  Memoria stimata  : texture={tex_mb:.1f} MB + output={out_mb:.1f} MB")
    if free/1024**2 < tex_mb + out_mb + 50:
        print("  AVVISO: VRAM scarsa! Riduci TEX_W/TEX_H o chiudi altri processi GPU.")
    else:
        print("  OK: VRAM sufficiente")
    print(f"  CuPy version     : {cp.__version__}")
    print("=" * 60 + "\n")

# ==============================================================================
# 4. GESTIONE TEXTURE CIELO
# ==============================================================================
def get_sky_texture(path, w=TEX_W, h=TEX_H):
    if os.path.exists(path):
        print(f"Caricamento texture: {path} -> {w}x{h}", flush=True)
        with Image.open(path) as img:
            img_rgb = img.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
            raw = np.array(img_rgb, dtype=np.float32)
        arr = raw / 255.0
    else:
        print(f"Texture non trovata. Cielo sintetico {w}x{h}.", flush=True)
        arr = np.zeros((h, w, 3), dtype=np.float32)
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        arr[..., 0] = xx
        arr[..., 1] = yy
        arr[..., 2] = 0.5

    result = np.ascontiguousarray(arr, dtype=np.float32)
    print(f"  shape={result.shape}, size={result.nbytes/1024**2:.1f} MB")
    gc.collect()
    return result

# ==============================================================================
# 5. MAIN
# ==============================================================================
def main():
    diagnose_gpu()

    # --- Carica e trasferisce texture ---
    sky_np    = get_sky_texture("nasa3.jpg")
    print("Trasferimento texture su GPU...", flush=True)
    d_sky_tex = cp.asarray(sky_np)   # cp.asarray: zero-copy se possibile, altrimenti DMA
    cp.cuda.stream.get_current_stream().synchronize()
    print(f"  OK — shape GPU: {d_sky_tex.shape}\n")
    del sky_np;  gc.collect()

    # Buffer output flat [H*W*3] — layout scelto per il kernel C
    d_img_out = cp.zeros(H * W * 3, dtype=cp.float32)

    # --- Compila con NVRTC (runtime compiler incluso nel driver, no nvcc) ---
    print("Compilazione kernel CUDA (NVRTC, no nvcc richiesto)...", flush=True)
    module = cp.RawModule(
        code=KERNEL_CODE,
        backend="nvrtc",                      # usa il compiler runtime del driver
        options=(
            "--std=c++14",
            "--gpu-architecture=sm_86",       # RTX 3050 Laptop = Ampere sm_86
        ),
    )
    kernel = module.get_function("render_kernel")
    print("  OK — kernel compilato\n")

    # Thread block 32×16 = 512 thread (ottimale per Ampere/RTX 30xx)
    block = (32, 16, 1)
    grid  = (math.ceil(W / 32), math.ceil(H / 16), 1)

    FPS = 30; SECONDS = 1; TOTAL_FRAMES = FPS * SECONDS
    os.makedirs("frames", exist_ok=True)

    # Puntatori GPU da passare al kernel
    tex_h = np.int32(d_sky_tex.shape[0])
    tex_w = np.int32(d_sky_tex.shape[1])

    print(f"Rendering: {TOTAL_FRAMES} frame @ {W}x{H}")
    print(f"Grid: {grid[:2]}  Block: {block[:2]}\n")

    for frame in range(TOTAL_FRAMES):
        time_t = np.float32(frame / FPS)
        print(f"  Frame {frame+1:3d}/{TOTAL_FRAMES}  t={float(time_t):.3f}s",
              end="\r", flush=True)

        kernel(
            grid, block,
            args=(
                d_img_out,
                np.int32(W), np.int32(H),
                d_sky_tex,
                tex_h, tex_w,
                np.float32(CAM_DIST),
                np.float32(FOV_DEG),
                np.float32(TEXTURE_ROTATION),
                np.int32(1 if USE_MIRRORING else 0),
                time_t,
                np.float32(DISK_R_MIN),
                np.float32(DISK_R_MAX),
            )
        )
        cp.cuda.stream.get_current_stream().synchronize()

        # Copia host e salvataggio PNG
        img_flat  = d_img_out.get()                          # numpy array flat
        img_32    = img_flat.reshape(H, W, 3)
        img_final = (np.clip(img_32, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(img_final).save(f"frames/frame_{frame:04d}.png")

    print("\n\nRendering completato!")

if __name__ == "__main__":
    main()
