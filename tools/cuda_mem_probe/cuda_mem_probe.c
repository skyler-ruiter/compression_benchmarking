// cuda_mem_probe — LD_PRELOAD peak device-memory tracker. See README.md for the
// design and the OUTPUT CONTRACT (JSON keys are load-bearing; do not rename).
//
// Dependency-free (dlfcn + pthread + stdio only); does NOT link against libcudart —
// wrappers take/return void*/size_t so we never need CUDA headers here.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <stdint.h>

// ---- state (guard all access with g_lock) ----------------------------------
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static size_t g_live = 0, g_peak = 0;
static uint64_t g_n_alloc = 0, g_n_free = 0, g_untracked_frees = 0, g_vmm = 0;

// ---- ptr -> size map --------------------------------------------------------
// Fixed-growth open-addressing hash table (linear probing). These tools make
// thousands, not millions, of allocations, so a simple table with occasional
// doubling is plenty. All access MUST happen under g_lock (callers hold it).
typedef struct {
    void   *ptr;    // NULL = empty slot; (void*)-1 sentinel is not used since we
                     // never store the value that a real cudaMalloc ptr would be
                     // for these two sentinels in practice on any target we run on.
    size_t  size;
    int     used;    // 0 = empty, 1 = occupied
} map_slot_t;

static map_slot_t *g_map = NULL;
static size_t g_map_cap = 0;   // capacity, power of two
static size_t g_map_count = 0; // occupied slots

static size_t hash_ptr(void *p) {
    uintptr_t x = (uintptr_t)p;
    // simple mix (splitmix64-ish), works for 32/64-bit uintptr_t
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (size_t)x;
}

// Forward decl.
static void map_grow(void);

// Insert or overwrite. Must be called with g_lock held.
static void map_put(void *ptr, size_t size) {
    if (!ptr) return;
    if (!g_map || (g_map_count + 1) * 4 >= g_map_cap * 3) { // load factor > 0.75
        map_grow();
    }
    size_t mask = g_map_cap - 1;
    size_t i = hash_ptr(ptr) & mask;
    for (;;) {
        if (!g_map[i].used) {
            g_map[i].used = 1;
            g_map[i].ptr = ptr;
            g_map[i].size = size;
            g_map_count++;
            return;
        }
        if (g_map[i].ptr == ptr) {
            // Overwrite (e.g. a stale entry from an untracked pairing); keep count.
            g_map[i].size = size;
            return;
        }
        i = (i + 1) & mask;
    }
}

// Remove and return size, or 0 if absent. Must be called with g_lock held.
static size_t map_take(void *ptr) {
    if (!ptr || !g_map) return 0;
    size_t mask = g_map_cap - 1;
    size_t i = hash_ptr(ptr) & mask;
    size_t start = i;
    for (;;) {
        if (!g_map[i].used) return 0; // not found
        if (g_map[i].ptr == ptr) {
            size_t sz = g_map[i].size;
            // Tombstone-free deletion via backward-shift to keep probing correct.
            g_map[i].used = 0;
            g_map[i].ptr = NULL;
            g_map[i].size = 0;
            g_map_count--;
            size_t j = i;
            for (;;) {
                size_t k = (j + 1) & mask;
                if (!g_map[k].used) break;
                size_t home = hash_ptr(g_map[k].ptr) & mask;
                // If k's home slot does not lie strictly between j+1 and k (cyclically),
                // moving it to j keeps it findable.
                int should_move;
                if (j <= k) {
                    should_move = !(home > j && home <= k);
                } else {
                    should_move = (home <= k) || (home > j);
                }
                if (should_move) {
                    g_map[j] = g_map[k];
                    g_map[k].used = 0;
                    g_map[k].ptr = NULL;
                    g_map[k].size = 0;
                    j = k;
                } else {
                    break;
                }
            }
            return sz;
        }
        i = (i + 1) & mask;
        if (i == start) return 0; // table full & not found (shouldn't happen)
    }
}

static void map_grow(void) {
    size_t new_cap = g_map_cap ? g_map_cap * 2 : 1024;
    map_slot_t *new_map = (map_slot_t *)calloc(new_cap, sizeof(map_slot_t));
    if (!new_map) return; // out of memory: degrade gracefully, keep old table
    map_slot_t *old_map = g_map;
    size_t old_cap = g_map_cap;
    g_map = new_map;
    g_map_cap = new_cap;
    g_map_count = 0;
    if (old_map) {
        size_t mask = new_cap - 1;
        for (size_t k = 0; k < old_cap; k++) {
            if (!old_map[k].used) continue;
            size_t i = hash_ptr(old_map[k].ptr) & mask;
            while (g_map[i].used) i = (i + 1) & mask;
            g_map[i] = old_map[k];
            g_map_count++;
        }
        free(old_map);
    }
}

// Note: dlsym re-entrancy is not a concern here because our wrappers call the
// real symbol via a cached dlsym(RTLD_NEXT, ...) result and never invoke a hooked
// CUDA entry point from inside map_put/map_take/dump_stats (those only touch
// calloc/free/stdio). No __thread guard is needed.

static void record_alloc(void *ptr, size_t size) {
    if (!ptr || size == 0) return;
    pthread_mutex_lock(&g_lock);
    map_put(ptr, size);
    g_live += size;
    if (g_live > g_peak) g_peak = g_live;
    g_n_alloc++;
    pthread_mutex_unlock(&g_lock);
}
static void record_free(void *ptr) {
    if (!ptr) return;
    pthread_mutex_lock(&g_lock);
    size_t s = map_take(ptr);
    if (s) { g_live -= s; g_n_free++; } else { g_untracked_frees++; }
    pthread_mutex_unlock(&g_lock);
}

static void dump_stats(void) {
    const char *path = getenv("CUDA_MEM_PROBE_OUT");
    char buf[512];
    size_t peak, live;
    uint64_t n_alloc, n_free, untracked, vmm;
    pthread_mutex_lock(&g_lock);
    peak = g_peak; live = g_live;
    n_alloc = g_n_alloc; n_free = g_n_free; untracked = g_untracked_frees; vmm = g_vmm;
    pthread_mutex_unlock(&g_lock);
    int n = snprintf(buf, sizeof buf,
        "{\"peak_device_bytes\": %zu, \"live_at_exit_bytes\": %zu, "
        "\"n_alloc\": %llu, \"n_free\": %llu, \"untracked_frees\": %llu, "
        "\"device_count_seen\": %d, \"vmm_calls_seen\": %llu}\n",
        peak, live, (unsigned long long)n_alloc, (unsigned long long)n_free,
        (unsigned long long)untracked, 1, (unsigned long long)vmm);
    if (n < 0) return;
    if (path) { FILE *f = fopen(path, "w"); if (f) { fwrite(buf, 1, (size_t)n, f); fclose(f); } }
    else { fprintf(stderr, "[cuda_mem_probe] %s", buf); }
}

// ---- signal handlers ---------------------------------------------------------
static void on_sigusr1(int signo) {
    (void)signo;
    pthread_mutex_lock(&g_lock);
    g_peak = g_live;
    pthread_mutex_unlock(&g_lock);
}
static void on_sigusr2(int signo) {
    (void)signo;
    dump_stats();
}

__attribute__((constructor)) static void probe_init(void) {
    struct sigaction sa1, sa2;
    memset(&sa1, 0, sizeof sa1);
    sa1.sa_handler = on_sigusr1;
    sigemptyset(&sa1.sa_mask);
    sa1.sa_flags = SA_RESTART;
    sigaction(SIGUSR1, &sa1, NULL);

    memset(&sa2, 0, sizeof sa2);
    sa2.sa_handler = on_sigusr2;
    sigemptyset(&sa2.sa_mask);
    sa2.sa_flags = SA_RESTART;
    sigaction(SIGUSR2, &sa2, NULL);
}
__attribute__((destructor))  static void probe_fini(void) { dump_stats(); }

// ---- wrappers: runtime API ----------------------------------------------------
// The runtime API returns cudaError_t (an int/enum); 0 == cudaSuccess.
typedef int (*cudaMalloc_t)(void **, size_t);
typedef int (*cudaFree_t)(void *);
typedef int (*cudaMallocAsync_t)(void **, size_t, void *);
typedef int (*cudaFreeAsync_t)(void *, void *);
typedef int (*cudaMallocManaged_t)(void **, size_t, unsigned);
typedef int (*cudaMallocPitch_t)(void **, size_t *, size_t, size_t);
// Allocation from an explicit cudaMemPool_t (cudaMemPoolCreate). Distinct symbol
// from cudaMallocAsync — this is what a pool-based allocator (e.g. FZGM's
// MemoryPool) actually calls, so it must be wrapped or per-buffer allocs are missed.
typedef int (*cudaMallocFromPoolAsync_t)(void **, size_t, void *, void *);

int cudaMalloc(void **devPtr, size_t size) {
    static cudaMalloc_t real = NULL;
    if (!real) real = (cudaMalloc_t)dlsym(RTLD_NEXT, "cudaMalloc");
    int rc = real(devPtr, size);
    if (rc == 0 && devPtr) record_alloc(*devPtr, size);
    return rc;
}
int cudaFree(void *devPtr) {
    static cudaFree_t real = NULL;
    if (!real) real = (cudaFree_t)dlsym(RTLD_NEXT, "cudaFree");
    record_free(devPtr);
    return real(devPtr);
}
int cudaMallocAsync(void **devPtr, size_t size, void *stream) {
    static cudaMallocAsync_t real = NULL;
    if (!real) real = (cudaMallocAsync_t)dlsym(RTLD_NEXT, "cudaMallocAsync");
    int rc = real(devPtr, size, stream);
    if (rc == 0 && devPtr) record_alloc(*devPtr, size);
    return rc;
}
int cudaFreeAsync(void *devPtr, void *stream) {
    static cudaFreeAsync_t real = NULL;
    if (!real) real = (cudaFreeAsync_t)dlsym(RTLD_NEXT, "cudaFreeAsync");
    record_free(devPtr);
    return real(devPtr, stream);
}
int cudaMallocManaged(void **devPtr, size_t size, unsigned flags) {
    static cudaMallocManaged_t real = NULL;
    if (!real) real = (cudaMallocManaged_t)dlsym(RTLD_NEXT, "cudaMallocManaged");
    int rc = real(devPtr, size, flags);
    if (rc == 0 && devPtr) record_alloc(*devPtr, size);
    return rc;
}
int cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) {
    static cudaMallocPitch_t real = NULL;
    if (!real) real = (cudaMallocPitch_t)dlsym(RTLD_NEXT, "cudaMallocPitch");
    int rc = real(devPtr, pitch, width, height);
    if (rc == 0 && devPtr && pitch) record_alloc(*devPtr, (*pitch) * height);
    return rc;
}
int cudaMallocFromPoolAsync(void **devPtr, size_t size, void *pool, void *stream) {
    static cudaMallocFromPoolAsync_t real = NULL;
    if (!real) real = (cudaMallocFromPoolAsync_t)dlsym(RTLD_NEXT, "cudaMallocFromPoolAsync");
    int rc = real(devPtr, size, pool, stream);
    if (rc == 0 && devPtr) record_alloc(*devPtr, size);
    return rc;
}

// ---- wrappers: driver API ------------------------------------------------------
// The driver API returns CUresult (an int); 0 == CUDA_SUCCESS. cuMemAlloc's first
// argument is a CUdeviceptr* (an integer handle, not a real pointer) but we key the
// map on its bit pattern the same way, treating it as void* for bookkeeping only.
typedef int (*cuMemAlloc_t)(void **, size_t);
typedef int (*cuMemAllocPitch_t)(void **, size_t *, size_t, size_t, unsigned);
typedef int (*cuMemAllocManaged_t)(void **, size_t, unsigned);
typedef int (*cuMemFree_t)(void *);
typedef int (*cuMemCreate_t)(void *, size_t, const void *, unsigned long long);

int cuMemAlloc(void **dptr, size_t bytesize) {
    static cuMemAlloc_t real = NULL;
    if (!real) real = (cuMemAlloc_t)dlsym(RTLD_NEXT, "cuMemAlloc");
    int rc = real(dptr, bytesize);
    if (rc == 0 && dptr) record_alloc(*dptr, bytesize);
    return rc;
}
int cuMemAllocPitch(void **dptr, size_t *pPitch, size_t WidthInBytes,
                     size_t Height, unsigned ElementSizeBytes) {
    static cuMemAllocPitch_t real = NULL;
    if (!real) real = (cuMemAllocPitch_t)dlsym(RTLD_NEXT, "cuMemAllocPitch");
    int rc = real(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
    if (rc == 0 && dptr && pPitch) record_alloc(*dptr, (*pPitch) * Height);
    return rc;
}
int cuMemAllocManaged(void **dptr, size_t bytesize, unsigned flags) {
    static cuMemAllocManaged_t real = NULL;
    if (!real) real = (cuMemAllocManaged_t)dlsym(RTLD_NEXT, "cuMemAllocManaged");
    int rc = real(dptr, bytesize, flags);
    if (rc == 0 && dptr) record_alloc(*dptr, bytesize);
    return rc;
}
int cuMemFree(void *dptr) {
    static cuMemFree_t real = NULL;
    if (!real) real = (cuMemFree_t)dlsym(RTLD_NEXT, "cuMemFree");
    record_free(dptr);
    return real(dptr);
}

// VMM path (cuMemCreate + cuMemMap) is out of scope for v1 sizing: just count that
// it happened so the report can warn the peak may undercount.
int cuMemCreate(void *handle, size_t size, const void *prop, unsigned long long flags) {
    static cuMemCreate_t real = NULL;
    if (!real) real = (cuMemCreate_t)dlsym(RTLD_NEXT, "cuMemCreate");
    int rc = real(handle, size, prop, flags);
    if (rc == 0) {
        pthread_mutex_lock(&g_lock);
        g_vmm++;
        pthread_mutex_unlock(&g_lock);
    }
    return rc;
}
