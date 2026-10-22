/*
 * mack_probe.c -- CPU-side ruler and load generator for Mack's calibration.
 *
 * The triad kernel and the persistent-process design come from
 * odi2027/scripts/cpu_probe.c: buffers are allocated and first-touched once, so
 * no probe pays (or causes) page-fault traffic, and the 192 MB working set is
 * ~24x the SLC so every pass is real DRAM traffic.
 *
 * Two modes.
 *
 *   mack_probe
 *       Probe. One line in, one line out:
 *         "t <burst_ms>"  STREAM triad burst      -> "RATE <GB/s> MS <ms>"
 *         "c <burst_ms>"  cache-resident FMA loop -> "RATE <Giter/s> MS <ms>"
 *         "q"             quit
 *       The triad reports memory-fabric pressure as a CPU agent sees it; the
 *       compute kernel reports core availability and clock state. Callers must
 *       take two back-to-back bursts and use the second: a burst after an idle
 *       gap measures the DVFS ramp, not contention (odi2027 N10).
 *
 *   mack_probe load <threads> <duty>
 *       Load generator: triad on <threads> threads, busy for <duty> of each
 *       50 ms period. Prints "LOAD <GB/s>" once a second. Runs until killed.
 *
 * Both modes run at user-interactive QoS. The probe stands in for the user's
 * foreground work, which the scheduler keeps on the performance cores; at default
 * QoS it sometimes landed on an efficiency core and read ~50% slow for reasons
 * that had nothing to do with Mack.
 *
 * Build: clang -O2 -Wall mack_probe.c -o mack_probe -lpthread
 */
#include <pthread.h>
#include <pthread/qos.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define ARRAY_MB 64
#define WORDS ((size_t)ARRAY_MB * 1024 * 1024 / sizeof(double))
#define PERIOD_S 0.050

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct {
    double *a, *b, *c;
} bufs_t;

static int alloc_bufs(bufs_t *x) {
    x->a = malloc(WORDS * 8);
    x->b = malloc(WORDS * 8);
    x->c = malloc(WORDS * 8);
    if (!x->a || !x->b || !x->c) return 0;
    for (size_t i = 0; i < WORDS; i++) {
        x->a[i] = 1.0;
        x->b[i] = 2.0;
        x->c[i] = 0.0;
    }
    return 1;
}

/* Reading the result back keeps the compiler from deleting the triad as dead
 * stores (it did: the first build reported ten million GB/s). */
static volatile double g_sink;

/* One triad pass; returns words moved (STREAM convention: 2 reads + 1 write). */
static size_t triad_pass(bufs_t *x) {
    const double scalar = 3.0;
    double *a = x->a, *b = x->b, *c = x->c;
    for (size_t i = 0; i < WORDS; i++) c[i] = a[i] + scalar * b[i];
    g_sink = c[WORDS / 2] + c[WORDS - 1];
    return WORDS * 3;
}

static double triad_burst(bufs_t *x, double burst_s, double *elapsed) {
    size_t words = 0;
    double t0 = now_s(), t1;
    do {
        words += triad_pass(x);
        t1 = now_s();
    } while (t1 - t0 < burst_s);
    *elapsed = t1 - t0;
    return (double)words * 8.0 / (t1 - t0) / 1e9;
}

/* Cache-resident FMA chains: 8 KB of state, so this never touches the fabric. */
static double compute_burst(double burst_s, double *elapsed) {
    enum { N = 1024, CHUNK = 20000 };
    static double v[N];
    for (int i = 0; i < N; i++) v[i] = 1.0 + i * 1e-6;
    size_t iters = 0;
    volatile double sink = 0.0;
    double t0 = now_s(), t1;
    do {
        for (int r = 0; r < CHUNK / N + 1; r++) {
            double s = 0.0;
            for (int i = 0; i < N; i++) {
                v[i] = v[i] * 1.0000001 + 1e-9;
                s += v[i];
            }
            sink += s;
            iters += N;
        }
        t1 = now_s();
    } while (t1 - t0 < burst_s);
    *elapsed = t1 - t0;
    return (double)iters / (t1 - t0) / 1e9;
}

typedef struct {
    double duty;
    volatile double gbps;
} load_arg_t;

static void *load_thread(void *p) {
    load_arg_t *arg = p;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    bufs_t x;
    if (!alloc_bufs(&x)) return NULL;
    for (;;) {
        double t0 = now_s();
        size_t words = 0;
        while (now_s() - t0 < PERIOD_S * arg->duty) words += triad_pass(&x);
        double busy_end = now_s();
        double rest = PERIOD_S - (busy_end - t0);
        if (rest > 0) usleep((useconds_t)(rest * 1e6));
        arg->gbps = (double)words * 8.0 / (now_s() - t0) / 1e9;
    }
    return NULL;
}

static int run_load(int threads, double duty) {
    if (threads < 1) threads = 1;
    if (duty < 0.01) duty = 0.01;
    if (duty > 1.0) duty = 1.0;
    load_arg_t *args = calloc(threads, sizeof *args);
    pthread_t *tids = calloc(threads, sizeof *tids);
    for (int i = 0; i < threads; i++) {
        args[i].duty = duty;
        pthread_create(&tids[i], NULL, load_thread, &args[i]);
    }
    for (;;) {
        sleep(1);
        double total = 0;
        for (int i = 0; i < threads; i++) total += args[i].gbps;
        printf("LOAD %.2f\n", total);
        fflush(stdout);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 4 && strcmp(argv[1], "load") == 0) return run_load(atoi(argv[2]), atof(argv[3]));

    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    bufs_t x;
    if (!alloc_bufs(&x)) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    printf("READY working_set_mb=%d\n", ARRAY_MB * 3);
    fflush(stdout);

    char line[64];
    while (fgets(line, sizeof line, stdin)) {
        if (line[0] == 'q') break;
        double burst_ms = atof(line + 1);
        if (burst_ms <= 0) burst_ms = 300.0;
        double elapsed = 0, rate;
        if (line[0] == 'c')
            rate = compute_burst(burst_ms / 1000.0, &elapsed);
        else
            rate = triad_burst(&x, burst_ms / 1000.0, &elapsed);
        printf("RATE %.4f MS %.2f\n", rate, elapsed * 1000.0);
        fflush(stdout);
    }
    return 0;
}
