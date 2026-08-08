/**
 * nvcomp_cli — a benchkit-shaped CLI over the nvCOMP high-level (manager) API.
 *
 * nvCOMP ships as a library, not a tool: there is no vendor binary that reads a
 * file, compresses it, and reports device time in a form the harness can consume.
 * This is that binary. It exists so nvCOMP can be measured under exactly the same
 * contract as every other adapter in benchkit (see docs/adapters/nvcomp.md):
 *
 *   - compress/decompress modes produce *artifacts*; the harness computes CR and
 *     quality from them and ignores their timing.
 *   - benchmark mode produces *timing* from an in-process repeat loop, with the
 *     input already resident on the device, CUDA events bracketing only the
 *     manager's compress()/decompress() call. No H2D/D2H, no file I/O, no process
 *     startup inside the timed region — the same "device kernel time" the FZGM and
 *     cuSZp adapters report.
 *
 * Every algorithm here is LOSSLESS. There is no error bound: nvCOMP compresses a
 * byte stream. That is the whole point of the comparison it was built for (FZGM's
 * GPU-Zstd back end vs. nvCOMP's Zstd on identical bytes), but it means rows from
 * this tool carry error_mode=lossless and max_abs_err=0 — see the validity-gate
 * carve-out in benchkit/validity.py.
 *
 * Usage:
 *   nvcomp_cli --compress   -i <in>     -o <out.nvc> -a zstd [--chunk-size N]
 *   nvcomp_cli --decompress -i <in.nvc> -o <out.bin> -a zstd
 *   nvcomp_cli --benchmark  -i <in>                  -a zstd --reps N
 *   common: --report-json <file>  --level N  --quiet
 *
 * Algorithms: zstd, lz4, deflate, gdeflate, ans, snappy, gzip, bitcomp, cascaded.
 *
 * TYPE-AWARE vs BYTE-ORIENTED CODERS
 * ----------------------------------
 * The first seven treat the input as an undifferentiated byte stream, so there is
 * nothing to tell them about it. **Bitcomp and Cascaded do not** — both take an
 * element type and model the input as an array of that type (Cascaded's delta and
 * bitpack passes operate on elements; Bitcomp's whole scheme is type-directed).
 * Handing them the default `NVCOMP_TYPE_UCHAR` on a uint16 or f32 field is the
 * D34 mistake in a new place: a documented default that is badly wrong for the
 * input, producing a number that understates the reference tool. Hence --dtype,
 * and hence it is *required* (not defaulted) for those two.
 *
 * nvCOMP 5.3 has NO f32 element type — the enum runs
 * char/uchar/short/ushort/int/uint/longlong/ulonglong/float16, and value 8
 * (NVCOMP_TYPE_FLOAT in older releases) has been removed. Raw f32 fields can
 * therefore only be described to these two coders by 4-byte *integer* width
 * (`--dtype uint`), which is a real modelling limitation of the comparison and
 * not a choice on our side. Record it whenever a raw-f32 Bitcomp/Cascaded row is
 * quoted. See docs/adapters/nvcomp.md.
 */

#include <nvcomp/ans.hpp>
#include <nvcomp/bitcomp.hpp>
#include <nvcomp/cascaded.hpp>
#include <nvcomp/deflate.hpp>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/gzip.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/snappy.hpp>
#include <nvcomp/zstd.hpp>
#include <nvcomp/nvcompManager.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

// ── error handling ───────────────────────────────────────────────────────────
// Every failure path ends in a JSON report with status != "ok" when
// --report-json was requested, because base.load_report_json() treats a missing
// file as "stale binary" and a non-ok status as a tool failure. Dying silently
// would be reported to the user as the wrong problem.

std::string g_report_path;

[[noreturn]] void fail(const std::string& msg)
{
    if (!g_report_path.empty()) {
        std::ofstream fh(g_report_path);
        fh << "{\n  \"status\": \"error\",\n  \"error_message\": \"" << msg << "\"\n}\n";
    }
    std::fprintf(stderr, "nvcomp_cli: %s\n", msg.c_str());
    std::exit(1);
}

void cudaCheck(cudaError_t rc, const char* what)
{
    if (rc != cudaSuccess)
        fail(std::string(what) + ": " + cudaGetErrorString(rc));
}

// ── element types (bitcomp / cascaded only) ──────────────────────────────────
// No "float"/"f32" spelling is accepted, because nvCOMP 5.3 has no such type and
// silently aliasing it to `uint` would bury the limitation in this file instead
// of surfacing it in the config that chose it. Callers with f32 data must write
// `--dtype uint` and mean it.

struct DtypeEntry { const char* name; nvcompType_t type; size_t width; };

const DtypeEntry kDtypes[] = {
    {"char",      NVCOMP_TYPE_CHAR,      1},
    {"uchar",     NVCOMP_TYPE_UCHAR,     1},
    {"short",     NVCOMP_TYPE_SHORT,     2},
    {"ushort",    NVCOMP_TYPE_USHORT,    2},
    {"int",       NVCOMP_TYPE_INT,       4},
    {"uint",      NVCOMP_TYPE_UINT,      4},
    {"longlong",  NVCOMP_TYPE_LONGLONG,  8},
    {"ulonglong", NVCOMP_TYPE_ULONGLONG, 8},
};

const DtypeEntry& lookupDtype(const std::string& name)
{
    for (const auto& e : kDtypes)
        if (name == e.name) return e;
    std::string have;
    for (const auto& e : kDtypes) { have += have.empty() ? "" : ", "; have += e.name; }
    fail("unknown --dtype '" + name + "' (have: " + have + "). nvCOMP 5.3 has no "
         "32-bit float type; use 'uint' for f32 data and record that in the result.");
}

// Type-aware coders index the input as an array. A trailing partial element is
// undefined behaviour in both (bitcomp.hpp warns that it "may crash or result in
// invalid output"), so it is rejected here rather than left to corrupt a row.
bool algoIsTyped(const std::string& algo)
{
    return algo == "bitcomp" || algo == "cascaded";
}

// ── algorithm dispatch ───────────────────────────────────────────────────────
// nvCOMP's managers are distinct types sharing nvcompManagerBase, and each takes
// its own opts struct. create_manager() exists but only reconstructs a manager
// from an already-compressed buffer, so compression still needs this switch.
//
// `level` maps to Deflate/Gdeflate's `algorithm` field (0 = entropy-only,
// 1 = default, 5 = highest ratio) and to Bitcomp's (0 = default, 1 = sparse).
// Zstd, LZ4, ANS, Snappy and Gzip expose no level in 5.3 — passing one for them
// is a hard error rather than a silent no-op, so a config that thinks it is
// sweeping levels cannot quietly produce identical rows.
//
// `casc` carries Cascaded's scheme (RLE passes, delta passes, bitpack). It is a
// configurable *pipeline*, not a single algorithm, which is why it is the one
// nvCOMP entry that FZGM can mirror stage for stage rather than pair off against
// a single coder — see configs/pipelines/cascaded_mirror_*.toml.

struct CascadedOpts { int num_rles = 2, num_deltas = 1, use_bp = 1; };

std::shared_ptr<nvcomp::nvcompManagerBase>
makeManager(const std::string& algo, size_t chunk_size, int level,
            nvcompType_t dtype, const CascadedOpts& casc, cudaStream_t stream)
{
    using namespace nvcomp;
    if (algo == "zstd") {
        return std::make_shared<ZstdManager>(
            chunk_size, nvcompBatchedZstdCompressDefaultOpts,
            nvcompBatchedZstdDecompressDefaultOpts, stream);
    }
    if (algo == "lz4") {
        return std::make_shared<LZ4Manager>(
            chunk_size, nvcompBatchedLZ4CompressDefaultOpts,
            nvcompBatchedLZ4DecompressDefaultOpts, stream);
    }
    if (algo == "ans") {
        return std::make_shared<ANSManager>(
            chunk_size, nvcompBatchedANSCompressDefaultOpts,
            nvcompBatchedANSDecompressDefaultOpts, stream);
    }
    if (algo == "deflate") {
        auto opts = nvcompBatchedDeflateCompressDefaultOpts;
        if (level >= 0) opts.algorithm = level;
        return std::make_shared<DeflateManager>(
            chunk_size, opts, nvcompBatchedDeflateDecompressDefaultOpts, stream);
    }
    if (algo == "gdeflate") {
        auto opts = nvcompBatchedGdeflateCompressDefaultOpts;
        if (level >= 0) opts.algorithm = level;
        return std::make_shared<GdeflateManager>(
            chunk_size, opts, nvcompBatchedGdeflateDecompressDefaultOpts, stream);
    }
    if (algo == "snappy") {
        return std::make_shared<SnappyManager>(
            chunk_size, nvcompBatchedSnappyCompressDefaultOpts,
            nvcompBatchedSnappyDecompressDefaultOpts, stream);
    }
    if (algo == "gzip") {
        // Gzip is decompress-oriented in nvCOMP (it exists to ingest host-produced
        // .gz); the manager still round-trips, and it is the only nvCOMP entry that
        // is byte-identical in format to what CPU zlib produces.
        return std::make_shared<GzipManager>(
            chunk_size, nvcompBatchedGzipCompressDefaultOpts,
            nvcompBatchedGzipDecompressDefaultOpts, stream);
    }
    if (algo == "bitcomp") {
        auto opts = nvcompBatchedBitcompCompressDefaultOpts;
        opts.data_type = dtype;
        if (level >= 0) opts.algorithm = level;
        return std::make_shared<BitcompManager>(
            chunk_size, opts, nvcompBatchedBitcompDecompressDefaultOpts, stream);
    }
    if (algo == "cascaded") {
        auto opts = nvcompBatchedCascadedCompressDefaultOpts;
        opts.type       = dtype;
        opts.num_RLEs   = casc.num_rles;
        opts.num_deltas = casc.num_deltas;
        opts.use_bp     = casc.use_bp;
        return std::make_shared<CascadedManager>(
            chunk_size, opts, nvcompBatchedCascadedDecompressDefaultOpts, stream);
    }
    fail("unknown algorithm '" + algo + "' (have: zstd, lz4, deflate, gdeflate, ans, "
         "snappy, gzip, bitcomp, cascaded)");
}

bool algoHasLevel(const std::string& algo)
{
    return algo == "deflate" || algo == "gdeflate" || algo == "bitcomp";
}

// ── file I/O ─────────────────────────────────────────────────────────────────

std::vector<uint8_t> readFile(const std::string& path)
{
    std::ifstream fh(path, std::ios::binary | std::ios::ate);
    if (!fh) fail("cannot open input file: " + path);
    const std::streamsize n = fh.tellg();
    if (n <= 0) fail("input file is empty: " + path);
    fh.seekg(0);
    std::vector<uint8_t> buf(static_cast<size_t>(n));
    if (!fh.read(reinterpret_cast<char*>(buf.data()), n))
        fail("short read on input file: " + path);
    return buf;
}

void writeFile(const std::string& path, const uint8_t* data, size_t n)
{
    std::ofstream fh(path, std::ios::binary);
    if (!fh) fail("cannot open output file: " + path);
    fh.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    if (!fh) fail("short write on output file: " + path);
}

// ── timing ───────────────────────────────────────────────────────────────────

struct EventPair {
    cudaEvent_t start{}, stop{};
    EventPair()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~EventPair() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin(cudaStream_t s) { cudaCheck(cudaEventRecord(start, s), "event record"); }
    float end(cudaStream_t s) {
        cudaCheck(cudaEventRecord(stop, s), "event record");
        cudaCheck(cudaEventSynchronize(stop), "event sync");
        float ms = 0.0f;
        cudaCheck(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
        return ms;
    }
};

std::string jsonArray(const std::vector<double>& v)
{
    std::string out = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s%.6f", i ? ", " : "", v[i]);
        out += buf;
    }
    return out + "]";
}

void usage()
{
    std::printf(
        "nvcomp_cli — benchkit CLI over the nvCOMP high-level API (all algorithms lossless)\n\n"
        "  nvcomp_cli --compress   -i <in>     -o <out.nvc> -a <algo> [--chunk-size N]\n"
        "  nvcomp_cli --decompress -i <in.nvc> -o <out.bin> -a <algo>\n"
        "  nvcomp_cli --benchmark  -i <in>                  -a <algo> --reps N\n\n"
        "Options:\n"
        "  -a, --algo <name>      zstd | lz4 | deflate | gdeflate | ans | snappy |\n"
        "                         gzip | bitcomp | cascaded        (default zstd)\n"
        "      --chunk-size N     uncompressed chunk size in bytes (default 65536)\n"
        "      --level N          deflate/gdeflate: 0=entropy-only .. 5=max ratio;\n"
        "                         bitcomp: 0=default, 1=sparse\n"
        "      --dtype <name>     REQUIRED for bitcomp/cascaded (they model the input\n"
        "                         as an array, not bytes). One of char, uchar, short,\n"
        "                         ushort, int, uint, longlong, ulonglong. nvCOMP 5.3\n"
        "                         has no f32 type — use 'uint' for f32 and say so.\n"
        "      --rles N           cascaded only: run-length passes   (default 2)\n"
        "      --deltas N         cascaded only: delta passes        (default 1)\n"
        "      --bp 0|1           cascaded only: final bitpack layer (default 1)\n"
        "      --reps N           benchmark: timed repetitions (default 5)\n"
        "      --warmup N         benchmark: untimed iterations first (default 0;\n"
        "                         benchkit drops its own warmup reps instead)\n"
        "      --report-json <f>  write a machine-readable report\n"
        "      --quiet            suppress the human-readable summary\n");
}

}  // namespace

int main(int argc, char** argv)
{
    std::string mode, input, output, algo = "zstd";
    size_t chunk_size = 65536;   // nvCOMP's documented sweet spot for Zstd
    int    level      = -1;      // -1 = use the algorithm's default
    int    reps = 5, warmup = 0;
    bool   quiet = false;
    std::string  dtype_name;     // empty = not supplied
    CascadedOpts casc;
    bool casc_flag_seen = false;

    auto need = [&](int i, const char* flag) -> const char* {
        if (i + 1 >= argc) fail(std::string("missing argument to ") + flag);
        return argv[i + 1];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "-h" || a == "--help")   { usage(); return 0; }
        else if (a == "-z" || a == "--compress")   mode = "compress";
        else if (a == "-x" || a == "--decompress") mode = "decompress";
        else if (a == "-b" || a == "--benchmark")  mode = "benchmark";
        else if (a == "-i" || a == "--input")  input  = need(i++, "-i");
        else if (a == "-o" || a == "--output") output = need(i++, "-o");
        else if (a == "-a" || a == "--algo")   algo   = need(i++, "-a");
        else if (a == "--chunk-size") chunk_size = std::strtoull(need(i++, "--chunk-size"), nullptr, 10);
        else if (a == "--level")      level  = std::atoi(need(i++, "--level"));
        else if (a == "--dtype")      dtype_name = need(i++, "--dtype");
        else if (a == "--rles")   { casc.num_rles   = std::atoi(need(i++, "--rles"));   casc_flag_seen = true; }
        else if (a == "--deltas") { casc.num_deltas = std::atoi(need(i++, "--deltas")); casc_flag_seen = true; }
        else if (a == "--bp")     { casc.use_bp     = std::atoi(need(i++, "--bp"));     casc_flag_seen = true; }
        else if (a == "--reps")       reps   = std::atoi(need(i++, "--reps"));
        else if (a == "--warmup")     warmup = std::atoi(need(i++, "--warmup"));
        else if (a == "--report-json") g_report_path = need(i++, "--report-json");
        else if (a == "--quiet") quiet = true;
        else fail("unknown option '" + a + "' (try --help)");
    }

    if (mode.empty())  fail("pick one of --compress / --decompress / --benchmark");
    if (input.empty()) fail("no input file (-i)");
    if (mode != "benchmark" && output.empty()) fail("no output file (-o)");
    if (level >= 0 && !algoHasLevel(algo))
        fail("--level is only meaningful for deflate/gdeflate/bitcomp; nvCOMP 5.3 "
             "exposes no compression level for '" + algo + "'");
    if (chunk_size == 0) fail("--chunk-size must be > 0");
    if (mode == "benchmark" && reps <= 0) fail("--reps must be > 0");

    // Inapplicable knobs are errors, not no-ops: a sweep that silently ignored
    // --rles would emit N identical rows under N distinct variant names.
    if (casc_flag_seen && algo != "cascaded")
        fail("--rles/--deltas/--bp are cascaded-only; '" + algo + "' has no such scheme");
    if (!dtype_name.empty() && !algoIsTyped(algo))
        fail("--dtype is only meaningful for bitcomp/cascaded; '" + algo +
             "' compresses an undifferentiated byte stream");
    if (dtype_name.empty() && algoIsTyped(algo))
        fail("'" + algo + "' models its input as a typed array and has no sensible "
             "default (nvCOMP's is uchar, which is wrong for every field in this "
             "corpus). Pass --dtype explicitly.");

    nvcompType_t dtype = NVCOMP_TYPE_UCHAR;
    size_t       dtype_width = 1;
    if (algoIsTyped(algo)) {
        const DtypeEntry& e = lookupDtype(dtype_name);
        dtype = e.type;
        dtype_width = e.width;
    }

    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream), "stream create");

    const std::vector<uint8_t> host_in = readFile(input);
    const size_t in_bytes = host_in.size();

    // Only on the compress side: in --decompress the input is the bitstream, whose
    // length has nothing to do with the element type.
    if (algoIsTyped(algo) && mode != "decompress" && in_bytes % dtype_width != 0)
        fail("input is " + std::to_string(in_bytes) + " bytes, not a multiple of the "
             + std::to_string(dtype_width) + "-byte --dtype " + dtype_name +
             ". nvCOMP's typed coders do not define a trailing partial element "
             "(bitcomp.hpp: \"may crash or result in invalid output\").");

    uint8_t* d_in = nullptr;
    cudaCheck(cudaMalloc(&d_in, in_bytes), "cudaMalloc input");
    cudaCheck(cudaMemcpy(d_in, host_in.data(), in_bytes, cudaMemcpyHostToDevice),
              "H2D input");

    auto mgr = makeManager(algo, chunk_size, level, dtype, casc, stream);

    size_t comp_bytes = 0, decomp_bytes = 0;
    std::vector<double> comp_ms, decomp_ms;
    std::vector<double> comp_host_ms, decomp_host_ms;

    if (mode == "decompress") {
        // d_in holds the compressed bitstream. NVCOMP_NATIVE embeds the metadata,
        // so the manager recovers chunking and sizes from the buffer itself; the
        // -a passed here only has to name the same family.
        auto dcfg = mgr->configure_decompression(d_in);
        decomp_bytes = dcfg.decomp_data_size;
        uint8_t* d_out = nullptr;
        cudaCheck(cudaMalloc(&d_out, decomp_bytes), "cudaMalloc decompressed");
        mgr->decompress(d_out, d_in, dcfg);
        cudaCheck(cudaStreamSynchronize(stream), "decompress sync");

        std::vector<uint8_t> host_out(decomp_bytes);
        cudaCheck(cudaMemcpy(host_out.data(), d_out, decomp_bytes, cudaMemcpyDeviceToHost),
                  "D2H decompressed");
        writeFile(output, host_out.data(), decomp_bytes);
        comp_bytes = in_bytes;
        cudaFree(d_out);
    } else {
        auto ccfg = mgr->configure_compression(in_bytes);
        uint8_t* d_comp = nullptr;
        cudaCheck(cudaMalloc(&d_comp, ccfg.max_compressed_buffer_size),
                  "cudaMalloc compressed");

        if (mode == "compress") {
            mgr->compress(d_in, d_comp, ccfg);
            cudaCheck(cudaStreamSynchronize(stream), "compress sync");
            comp_bytes = mgr->get_compressed_output_size(d_comp);

            std::vector<uint8_t> host_comp(comp_bytes);
            cudaCheck(cudaMemcpy(host_comp.data(), d_comp, comp_bytes, cudaMemcpyDeviceToHost),
                      "D2H compressed");
            writeFile(output, host_comp.data(), comp_bytes);
            decomp_bytes = in_bytes;
        } else {
            // ── benchmark ────────────────────────────────────────────────────
            // One manager, reused across every repetition: nvCOMP allocates its
            // scratch buffer lazily on the first compress, and re-creating the
            // manager per rep would time that allocation instead of the codec.
            // The buffers below are allocated once for the same reason.
            //
            // Decompression is timed against a bitstream produced by an untimed
            // compress, and configure_decompression() (which parses the header and
            // synchronizes the stream) is hoisted out of the loop — it is setup,
            // not decode work.
            EventPair ev;

            mgr->compress(d_in, d_comp, ccfg);
            cudaCheck(cudaStreamSynchronize(stream), "warmup compress sync");
            comp_bytes = mgr->get_compressed_output_size(d_comp);

            auto dcfg = mgr->configure_decompression(d_comp);
            decomp_bytes = dcfg.decomp_data_size;
            if (decomp_bytes != in_bytes)
                fail("round-trip size mismatch: compressed " + std::to_string(in_bytes) +
                     " bytes but the bitstream reports " + std::to_string(decomp_bytes));
            uint8_t* d_dec = nullptr;
            cudaCheck(cudaMalloc(&d_dec, decomp_bytes), "cudaMalloc decompressed");

            // Warm BOTH phases. nvCOMP allocates decompression scratch lazily on
            // the first decompress just as it does for compress, and measured
            // rep 0 at 20.4 ms against a 13.8 ms steady state (+48%) on
            // CESM-2D/CLDHGH. benchkit's warmup-rep drop hid that from the median,
            // but a caller running --reps 1 would have silently reported the cold
            // number, and the compress path was already being warmed here — so the
            // asymmetry was ours, not nvCOMP's.
            mgr->decompress(d_dec, d_comp, dcfg);
            cudaCheck(cudaStreamSynchronize(stream), "warmup decompress sync");

            for (int i = 0; i < warmup; ++i) {
                mgr->compress(d_in, d_comp, ccfg);
                mgr->decompress(d_dec, d_comp, dcfg);
            }
            cudaCheck(cudaStreamSynchronize(stream), "warmup sync");

            // Host wall time is recorded alongside device time for every rep. It is
            // not a second throughput number — it is the audit that says how much
            // work a device-only figure is excluding. FZGM's split-mode compress
            // reports 1.26 ms device against 3.60 ms host (the four coded ports are
            // assembled into one archive on the host, outside its event bracket);
            // nvCOMP's manager writes one contiguous buffer on device and shows
            // ~0.01 ms of gap. Comparing the two device numbers without that ratio
            // in view overstates the difference by ~3x. See docs/adapters/nvcomp.md.
            for (int i = 0; i < reps; ++i) {
                auto h0 = std::chrono::high_resolution_clock::now();
                ev.begin(stream);
                mgr->compress(d_in, d_comp, ccfg);
                comp_ms.push_back(ev.end(stream));
                comp_host_ms.push_back(
                    std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - h0).count());

                auto h1 = std::chrono::high_resolution_clock::now();
                ev.begin(stream);
                mgr->decompress(d_dec, d_comp, dcfg);
                decomp_ms.push_back(ev.end(stream));
                decomp_host_ms.push_back(
                    std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - h1).count());
            }
            cudaFree(d_dec);
        }
        cudaFree(d_comp);
    }

    cudaFree(d_in);

    const double cr = comp_bytes ? static_cast<double>(
        mode == "decompress" ? decomp_bytes : in_bytes) / comp_bytes : 0.0;

    if (!quiet) {
        std::printf("nvcomp_cli %s  algo=%s chunk=%zu", mode.c_str(), algo.c_str(), chunk_size);
        if (level >= 0) std::printf(" level=%d", level);
        if (algoIsTyped(algo)) std::printf(" dtype=%s", dtype_name.c_str());
        if (algo == "cascaded")
            std::printf(" rles=%d deltas=%d bp=%d", casc.num_rles, casc.num_deltas, casc.use_bp);
        std::printf("\n  uncompressed: %zu bytes\n  compressed:   %zu bytes  (%.4fx)\n",
                    mode == "decompress" ? decomp_bytes : in_bytes, comp_bytes, cr);
        for (size_t i = 0; i < comp_ms.size(); ++i)
            std::printf("  rep %zu: compress %.4f ms   decompress %.4f ms\n",
                        i, comp_ms[i], decomp_ms[i]);
    }

    if (!g_report_path.empty()) {
        std::ofstream fh(g_report_path);
        fh << "{\n"
           << "  \"status\": \"ok\",\n"
           << "  \"tool\": \"nvcomp_cli\",\n"
           << "  \"mode\": \"" << mode << "\",\n"
           << "  \"algorithm\": \"" << algo << "\",\n"
           << "  \"lossless\": true,\n"
           << "  \"chunk_size\": " << chunk_size << ",\n"
           << "  \"level\": " << level << ",\n"
           << "  \"dtype\": " << (algoIsTyped(algo) ? "\"" + dtype_name + "\"" : "null") << ",\n";
        if (algo == "cascaded")
            fh << "  \"cascaded_num_rles\": "   << casc.num_rles   << ",\n"
               << "  \"cascaded_num_deltas\": " << casc.num_deltas << ",\n"
               << "  \"cascaded_use_bp\": "     << casc.use_bp     << ",\n";
        fh
           << "  \"uncompressed_bytes\": "
           << (mode == "decompress" ? decomp_bytes : in_bytes) << ",\n"
           << "  \"compressed_bytes\": " << comp_bytes << ",\n"
           << "  \"compression_ratio\": " << cr << ",\n"
           << "  \"compress_device_ms\": " << jsonArray(comp_ms) << ",\n"
           << "  \"decompress_device_ms\": " << jsonArray(decomp_ms) << ",\n"
           << "  \"compress_host_ms\": " << jsonArray(comp_host_ms) << ",\n"
           << "  \"decompress_host_ms\": " << jsonArray(decomp_host_ms) << ",\n"
           << "  \"timing_method\": \"cuda_events_device_only_in_process\",\n"
           << "  \"host_ms_note\": \"wall time around the same call; the gap vs "
              "device_ms is how much a device-only figure excludes\"\n"
           << "}\n";
    }
    return 0;
}
