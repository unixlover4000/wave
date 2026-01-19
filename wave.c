// aviz.c - Terminal audio visualiser (ALSA capture + FFT + ncurses)
// Arch deps: alsa-lib, ncurses
// Build: gcc -O2 -Wall -Wextra -std=c11 aviz.c -o aviz -lasound -lncurses -lm
// Run:   ./aviz
// Device override: ARECORD_DEVICE="hw:1,0" ./aviz

#define _POSIX_C_SOURCE 200809L
#include <alsa/asoundlib.h>
#include <math.h>
#include <ncurses.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int sig) { (void)sig; g_stop = 1; }

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void sleep_ms(int ms) {
    struct timespec req;
    req.tv_sec = ms / 1000;
    req.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&req, NULL);
}

// ------------------------ Complex + FFT ------------------------

typedef struct { float re, im; } cpx;

static inline cpx c_add(cpx a, cpx b){ return (cpx){a.re+b.re, a.im+b.im}; }
static inline cpx c_sub(cpx a, cpx b){ return (cpx){a.re-b.re, a.im-b.im}; }
static inline cpx c_mul(cpx a, cpx b){
    return (cpx){a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re};
}

static unsigned bitrev(unsigned x, int log2n){
    unsigned r = 0;
    for(int i=0;i<log2n;i++){
        r = (r<<1) | (x & 1u);
        x >>= 1u;
    }
    return r;
}

// In-place radix-2 FFT, n must be power of 2
static void fft(cpx *a, int n, int inverse){
    int log2n = 0;
    for(int t=n; t>1; t>>=1) log2n++;

    // bit-reversal permutation
    for(unsigned i=0;i<(unsigned)n;i++){
        unsigned j = bitrev(i, log2n);
        if(j > i){
            cpx tmp = a[i]; a[i] = a[j]; a[j] = tmp;
        }
    }

    for(int len=2; len<=n; len<<=1){
        float ang = (inverse ? 2.0f : -2.0f) * (float)M_PI / (float)len;
        cpx wlen = (cpx){cosf(ang), sinf(ang)};
        for(int i=0; i<n; i+=len){
            cpx w = (cpx){1.0f, 0.0f};
            for(int j=0; j<len/2; j++){
                cpx u = a[i + j];
                cpx v = c_mul(a[i + j + len/2], w);
                a[i + j] = c_add(u, v);
                a[i + j + len/2] = c_sub(u, v);
                w = c_mul(w, wlen);
            }
        }
    }

    if(inverse){
        float invn = 1.0f / (float)n;
        for(int i=0;i<n;i++){
            a[i].re *= invn;
            a[i].im *= invn;
        }
    }
}

// ------------------------ ALSA capture ------------------------

typedef struct {
    snd_pcm_t *handle;
    unsigned sample_rate;
    int channels;
} capture_t;

static void die_alsa(const char *msg, int err){
    endwin();
    fprintf(stderr, "%s: %s\n", msg, snd_strerror(err));
    exit(1);
}

static void setup_capture(capture_t *cap, const char *device, unsigned rate, int channels){
    memset(cap, 0, sizeof(*cap));
    cap->sample_rate = rate;
    cap->channels = channels;

    int err = snd_pcm_open(&cap->handle, device, SND_PCM_STREAM_CAPTURE, 0);
    if(err < 0) die_alsa("snd_pcm_open failed", err);

    snd_pcm_hw_params_t *hw = NULL;
    snd_pcm_hw_params_alloca(&hw);
    err = snd_pcm_hw_params_any(cap->handle, hw);
    if(err < 0) die_alsa("snd_pcm_hw_params_any failed", err);

    err = snd_pcm_hw_params_set_access(cap->handle, hw, SND_PCM_ACCESS_RW_INTERLEAVED);
    if(err < 0) die_alsa("set_access failed", err);

    err = snd_pcm_hw_params_set_format(cap->handle, hw, SND_PCM_FORMAT_S16_LE);
    if(err < 0) die_alsa("set_format failed (need S16_LE)", err);

    unsigned exact_rate = rate;
    err = snd_pcm_hw_params_set_rate_near(cap->handle, hw, &exact_rate, 0);
    if(err < 0) die_alsa("set_rate_near failed", err);
    cap->sample_rate = exact_rate;

    err = snd_pcm_hw_params_set_channels(cap->handle, hw, (unsigned)channels);
    if(err < 0) die_alsa("set_channels failed", err);

    // buffer/period tuning (safe defaults)
    snd_pcm_uframes_t period = 1024;
    snd_pcm_uframes_t buffer = 4096;
    snd_pcm_hw_params_set_period_size_near(cap->handle, hw, &period, 0);
    snd_pcm_hw_params_set_buffer_size_near(cap->handle, hw, &buffer);

    err = snd_pcm_hw_params(cap->handle, hw);
    if(err < 0) die_alsa("snd_pcm_hw_params failed", err);

    err = snd_pcm_prepare(cap->handle);
    if(err < 0) die_alsa("snd_pcm_prepare failed", err);
}

static int read_frames_s16(capture_t *cap, int16_t *dst, int frames){
    int got = 0;
    while(got < frames && !g_stop){
        int r = snd_pcm_readi(cap->handle, dst + got*cap->channels, frames - got);
        if(r == -EPIPE){
            // overrun
            snd_pcm_prepare(cap->handle);
            continue;
        } else if(r == -EAGAIN){
            continue;
        } else if(r < 0){
            snd_pcm_prepare(cap->handle);
            continue;
        }
        got += r;
    }
    return got;
}

// ------------------------ Visualiser ------------------------

static float clampf(float x, float a, float b){
    if(x < a) return a;
    if(x > b) return b;
    return x;
}

static void make_hann(float *w, int n){
    for(int i=0;i<n;i++){
        w[i] = 0.5f - 0.5f * cosf(2.0f*(float)M_PI*(float)i/(float)(n-1));
    }
}

// Log-spaced band mapping helper
static float lerpf(float a, float b, float t){ return a + (b-a)*t; }

int main(void){
    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);

    // Choose ALSA device
    const char *dev = getenv("ARECORD_DEVICE");
    if(!dev) dev = "default";

    // Audio settings
    const unsigned SR = 44100;
    const int CH = 1;

    // FFT settings (power of 2)
    const int N = 2048;     // FFT size
    const int HOP = 1024;   // how many new frames per update
    const float EPS = 1e-9f;

    // Setup ALSA capture
    capture_t cap;
    setup_capture(&cap, dev, SR, CH);

    // Init ncurses
    initscr();
    cbreak();
    noecho();
    nodelay(stdscr, TRUE);
    keypad(stdscr, TRUE);
    curs_set(0);

    // Allocate buffers
    int16_t *pcm = (int16_t*)calloc((size_t)HOP * (size_t)CH, sizeof(int16_t));
    float *ring = (float*)calloc((size_t)N, sizeof(float));
    float *win = (float*)malloc((size_t)N * sizeof(float));
    cpx *spec = (cpx*)malloc((size_t)N * sizeof(cpx));
    if(!pcm || !ring || !win || !spec){
        endwin();
        fprintf(stderr, "Out of memory\n");
        return 1;
    }
    make_hann(win, N);

    // Smoothed bar values
    int last_cols = 0;
    float *bars = NULL;        // size = cols
    float *bars_smooth = NULL; // size = cols

    // Visual tuning
    const float smooth_attack = 0.55f;  // higher = faster rise
    const float smooth_release = 0.18f; // higher = faster fall
    const float floor_db = -70.0f;      // noise floor
    const float ceil_db  = -5.0f;       // near max

    double last_frame = now_sec();

    while(!g_stop){
        int ch = getch();
        if(ch == 'q' || ch == 'Q') break;

        // Resize handling
        int rows, cols;
        getmaxyx(stdscr, rows, cols);
        if(cols < 10 || rows < 5){
            erase();
            mvprintw(0,0,"Terminal too small. Resize or press q.");
            refresh();
            sleep_ms(50);
            continue;
        }
        if(cols != last_cols){
            free(bars);
            free(bars_smooth);
            bars = (float*)calloc((size_t)cols, sizeof(float));
            bars_smooth = (float*)calloc((size_t)cols, sizeof(float));
            last_cols = cols;
        }

        // Read HOP frames
        int got = read_frames_s16(&cap, pcm, HOP);
        if(got <= 0) break;

        // Shift ring buffer left by HOP, append new samples
        memmove(ring, ring + HOP, (size_t)(N - HOP) * sizeof(float));
        for(int i=0;i<HOP;i++){
            // mono S16 -> float [-1,1]
            float x = (float)pcm[i*CH] / 32768.0f;
            ring[N - HOP + i] = x;
        }

        // Window + copy to complex buffer
        for(int i=0;i<N;i++){
            float xw = ring[i] * win[i];
            spec[i].re = xw;
            spec[i].im = 0.0f;
        }

        // FFT
        fft(spec, N, 0);

        // Compute magnitudes for bins 1..N/2-1
        // We will map these to screen columns using log-frequency bands.
        int usable_bins = N/2;
        float nyquist = (float)cap.sample_rate * 0.5f;

        // Frequency range to show (skip sub-bass rumble)
        float fmin = 40.0f;
        float fmax = nyquist * 0.95f;

        // Clear bars
        for(int x=0;x<cols;x++) bars[x] = 0.0f;

        for(int x=0; x<cols; x++){
            // Log-spaced bands
            float t0 = (float)x / (float)cols;
            float t1 = (float)(x+1) / (float)cols;

            // Log mapping: f = fmin * (fmax/fmin)^t
            float f0 = fmin * powf(fmax / fmin, t0);
            float f1 = fmin * powf(fmax / fmin, t1);

            int b0 = (int)floorf(f0 * (float)N / (float)cap.sample_rate);
            int b1 = (int)ceilf (f1 * (float)N / (float)cap.sample_rate);

            if(b0 < 1) b0 = 1;
            if(b1 > usable_bins-1) b1 = usable_bins-1;
            if(b1 <= b0) b1 = b0 + 1;
            if(b1 > usable_bins-1) b1 = usable_bins-1;

            // Average power in band
            float acc = 0.0f;
            int count = 0;
            for(int k=b0; k<=b1; k++){
                float re = spec[k].re;
                float im = spec[k].im;
                float mag2 = re*re + im*im;
                acc += mag2;
                count++;
            }
            float p = (count > 0) ? (acc / (float)count) : 0.0f;

            // Convert to dB-ish scale (relative)
            float db = 10.0f * log10f(p + EPS);

            // Normalize to 0..1 in [floor_db, ceil_db]
            float norm = (db - floor_db) / (ceil_db - floor_db);
            norm = clampf(norm, 0.0f, 1.0f);

            bars[x] = norm;
        }

        // Smooth bars and draw
        erase();

        // Header
        mvprintw(0, 0, "aviz  | device: %s  | SR: %u  | N:%d HOP:%d  | q=quit",
                 dev, cap.sample_rate, N, HOP);

        int vis_top = 1;
        int vis_h = rows - vis_top;

        for(int x=0; x<cols; x++){
            float v = bars[x];

            // simple smoothing (different attack/release)
            float prev = bars_smooth[x];
            float a = (v > prev) ? smooth_attack : smooth_release;
            float sm = prev + a * (v - prev);
            bars_smooth[x] = sm;

            int h = (int)lroundf(sm * (float)(vis_h - 1));
            if(h < 0) h = 0;
            if(h > vis_h - 1) h = vis_h - 1;

            // draw from bottom up
            for(int yy=0; yy<h; yy++){
                int y = rows - 1 - yy;
                mvaddch(y, x, '|');
            }
        }

        refresh();

        // Aim for ~30 FPS (but capture pacing also matters)
        double t = now_sec();
        double dt = t - last_frame;
        last_frame = t;
        (void)dt;
        sleep_ms(10);
    }

    // Cleanup
    endwin();
    if(cap.handle) snd_pcm_close(cap.handle);
    free(pcm);
    free(ring);
    free(win);
    free(spec);
    free(bars);
    free(bars_smooth);
    return 0;
}
