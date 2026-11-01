#include <Arduino.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include "MPCNN_weights.h"

#if MPCNN_IMG_SIZE != 180
#error "Header incompatible: IMG_SIZE"
#endif
#if MPCNN_GRID != 4
#error "Header incompatible: GRID"
#endif
#if MPCNN_TILE_SIZE != 45
#error "Header incompatible: TILE_SIZE"
#endif

static constexpr int IMG = 180;
static constexpr int GRID = 4;
static constexpr int TILE = 45;
static constexpr int C1 = 16;
static constexpr int C2 = 16;
static constexpr int C3 = 32;
static constexpr int L1 = 45;
static constexpr int L2 = 22;
static constexpr int L3 = 11;
static constexpr int FEATURES = 512;
static constexpr int HIDDEN = 32;
static constexpr int BENCH_RUNS = 5;

static_assert(sizeof(MPCNN_W_CONV1) == 3 * 135 * 16, "CONV1 incompatible");
static_assert(sizeof(MPCNN_W_CONV2) == 5 * 16 * 16, "CONV2 incompatible");
static_assert(sizeof(MPCNN_W_CONV3) == 6 * 16 * 32, "CONV3 incompatible");
static_assert(sizeof(MPCNN_W_TILE_DENSE) == 32 * 32, "TILE_DENSE incompatible");
static_assert(sizeof(MPCNN_W_MLP_DENSE) == 512 * 32, "MLP_DENSE incompatible");
static_assert(sizeof(MPCNN_W_OUTPUT) == 32, "OUTPUT incompatible");

static uint8_t patch_rows[TILE * TILE * 3];

static uint8_t conv1_out[L1 * C1];
static uint8_t pool1_out[L2 * C1];
static uint8_t conv2_out[L2 * C2];
static uint8_t pool2_out[L3 * C2];
static uint8_t conv3_out[L3 * C3];

static uint8_t gap_out[C3];
static uint8_t tile_features[C3];
static uint8_t all_features[FEATURES];
static uint8_t hidden_out[HIDDEN];

static size_t scratch_bytes() {
    return sizeof(patch_rows)
         + sizeof(conv1_out)
         + sizeof(pool1_out)
         + sizeof(conv2_out)
         + sizeof(pool2_out)
         + sizeof(conv3_out)
         + sizeof(gap_out)
         + sizeof(tile_features)
         + sizeof(all_features)
         + sizeof(hidden_out);
}

static size_t model_bytes() {
    return sizeof(MPCNN_W_CONV1) + sizeof(MPCNN_B_CONV1) + sizeof(MPCNN_MULT_CONV1)
         + sizeof(MPCNN_W_CONV2) + sizeof(MPCNN_B_CONV2) + sizeof(MPCNN_MULT_CONV2)
         + sizeof(MPCNN_W_CONV3) + sizeof(MPCNN_B_CONV3) + sizeof(MPCNN_MULT_CONV3)
         + sizeof(MPCNN_W_TILE_DENSE) + sizeof(MPCNN_B_TILE_DENSE) + sizeof(MPCNN_MULT_TILE_DENSE)
         + sizeof(MPCNN_W_MLP_DENSE) + sizeof(MPCNN_B_MLP_DENSE) + sizeof(MPCNN_MULT_MLP_DENSE)
         + sizeof(MPCNN_W_OUTPUT) + sizeof(MPCNN_B_OUTPUT)
         + sizeof(MPCNN_GAP_MULTIPLIER) + sizeof(MPCNN_LOGIT_SCALE)
         + sizeof(MPCNN_SCALE_CONV1) + sizeof(MPCNN_SCALE_CONV2)
         + sizeof(MPCNN_SCALE_CONV3) + sizeof(MPCNN_TILE_DENSE_SCALE)
         + sizeof(MPCNN_MLP_DENSE_SCALE);
}

static void print_memory(const char *label) {
    const uint32_t free_heap =
        heap_caps_get_free_size(MALLOC_CAP_DEFAULT);
    const uint32_t min_heap =
        heap_caps_get_minimum_free_size(MALLOC_CAP_DEFAULT);
    const uint32_t largest_heap =
        heap_caps_get_largest_free_block(MALLOC_CAP_DEFAULT);

    const uint32_t free_internal =
        heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const uint32_t min_internal =
        heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const uint32_t largest_internal =
        heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    Serial.printf("\n--- RAM: %s ---\n", label);

    Serial.printf("Free heap                : %lu B (%.2f KB)\n",
        (unsigned long)free_heap, free_heap / 1024.0f);

    Serial.printf("Historical minimum heap  : %lu B (%.2f KB)\n",
        (unsigned long)min_heap, min_heap / 1024.0f);

    Serial.printf("Largest heap block       : %lu B (%.2f KB)\n",
        (unsigned long)largest_heap, largest_heap / 1024.0f);

    Serial.printf("Free internal RAM        : %lu B (%.2f KB)\n",
        (unsigned long)free_internal, free_internal / 1024.0f);

    Serial.printf("Minimum internal RAM     : %lu B (%.2f KB)\n",
        (unsigned long)min_internal, min_internal / 1024.0f);

    Serial.printf("Largest internal block   : %lu B (%.2f KB)\n",
        (unsigned long)largest_internal, largest_internal / 1024.0f);

    if (psramFound()) {
        const uint32_t total_psram =
            heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
        const uint32_t free_psram =
            heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        const uint32_t min_psram =
            heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM);

        Serial.printf("Total PSRAM              : %lu B (%.2f KB)\n",
            (unsigned long)total_psram, total_psram / 1024.0f);

        Serial.printf("Free PSRAM               : %lu B (%.2f KB)\n",
            (unsigned long)free_psram, free_psram / 1024.0f);

        Serial.printf("Historical minimum PSRAM : %lu B (%.2f KB)\n",
            (unsigned long)min_psram, min_psram / 1024.0f);
    } else {
        Serial.println("PSRAM                    : not detected");
    }

    Serial.printf("FreeRTOS stack HWM       : %u units\n",
        (unsigned)uxTaskGetStackHighWaterMark(NULL));

    Serial.printf("Static buffers           : %u B (%.2f KB)\n",
        (unsigned)scratch_bytes(), scratch_bytes() / 1024.0f);

    Serial.printf("Model parameters         : %u B (%.2f KB)\n",
        (unsigned)model_bytes(), model_bytes() / 1024.0f);

    Serial.println("------------------------------");
}

static void print_flash() {
    Serial.println("\n--- FLASH ---");

    Serial.printf("Physical flash           : %lu B (%.2f MB)\n",
        (unsigned long)ESP.getFlashChipSize(),
        ESP.getFlashChipSize() / (1024.0f * 1024.0f));

    Serial.printf("Firmware/sketch          : %lu B (%.2f KB)\n",
        (unsigned long)ESP.getSketchSize(),
        ESP.getSketchSize() / 1024.0f);

    Serial.printf("Free OTA space            : %lu B (%.2f KB)\n",
        (unsigned long)ESP.getFreeSketchSpace(),
        ESP.getFreeSketchSpace() / 1024.0f);

    Serial.printf("CPU frequency            : %u MHz\n",
        ESP.getCpuFreqMHz());

    Serial.println("Firmware size includes code and libraries.");
    Serial.println("------------------------------");
}

static inline uint8_t requantize_relu(int32_t acc, float multiplier) {
    float value = (float)acc * multiplier;

    if (!(value > 0.0f)) return 0;

    value = floorf(value + 0.5f);

    if (value >= 255.0f) return 255;

    return (uint8_t)value;
}

static void conv1d_u8(
    const uint8_t *input,
    int length,
    int cin,
    int cout,
    int kernel,
    const int8_t *weights,
    const int32_t *bias,
    const float *multipliers,
    uint8_t *output
) {
    const int pad_left = (kernel - 1) / 2;

    for (int t = 0; t < length; ++t) {
        for (int oc = 0; oc < cout; ++oc) {
            int32_t acc = bias[oc];

            for (int k = 0; k < kernel; ++k) {
                const int pos = t + k - pad_left;

                if (pos < 0 || pos >= length) continue;

                const uint8_t *x =
                    input + (size_t)pos * cin;

                const size_t base =
                    ((size_t)k * cin) * cout + oc;

                for (int ic = 0; ic < cin; ++ic) {
                    acc += (int32_t)x[ic] *
                        (int32_t)weights[base + (size_t)ic * cout];
                }
            }

            output[(size_t)t * cout + oc] =
                requantize_relu(acc, multipliers[oc]);
        }
    }
}

static void max_pool1d_u8(
    const uint8_t *input,
    int length,
    int channels,
    uint8_t *output
) {
    const int out_length = length / 2;

    for (int t = 0; t < out_length; ++t) {
        const uint8_t *a =
            input + (size_t)(2 * t) * channels;
        const uint8_t *b =
            input + (size_t)(2 * t + 1) * channels;
        uint8_t *out =
            output + (size_t)t * channels;

        for (int c = 0; c < channels; ++c) {
            out[c] = a[c] > b[c] ? a[c] : b[c];
        }
    }
}

static void gap_u8(
    const uint8_t *input,
    int length,
    int channels,
    float multiplier,
    uint8_t *output
) {
    for (int c = 0; c < channels; ++c) {
        int32_t sum = 0;

        for (int t = 0; t < length; ++t) {
            sum += input[(size_t)t * channels + c];
        }

        float level = ((float)sum / (float)length) * multiplier;
        level = floorf(level + 0.5f);

        if (!(level > 0.0f)) output[c] = 0;
        else if (level >= 255.0f) output[c] = 255;
        else output[c] = (uint8_t)level;
    }
}

static void dense_relu_u8(
    const uint8_t *input,
    int cin,
    int cout,
    const int8_t *weights,
    const int32_t *bias,
    const float *multipliers,
    uint8_t *output
) {
    for (int o = 0; o < cout; ++o) {
        int32_t acc = bias[o];

        for (int i = 0; i < cin; ++i) {
            acc += (int32_t)input[i] *
                (int32_t)weights[(size_t)i * cout + o];
        }

        output[o] = requantize_relu(acc, multipliers[o]);
    }
}

static void process_tile(const uint8_t *patch, uint8_t *features_out) {
    conv1d_u8(
        patch, L1, 135, C1, 3,
        MPCNN_W_CONV1, MPCNN_B_CONV1, MPCNN_MULT_CONV1,
        conv1_out
    );

    max_pool1d_u8(conv1_out, L1, C1, pool1_out);

    conv1d_u8(
        pool1_out, L2, C1, C2, 5,
        MPCNN_W_CONV2, MPCNN_B_CONV2, MPCNN_MULT_CONV2,
        conv2_out
    );

    max_pool1d_u8(conv2_out, L2, C2, pool2_out);

    conv1d_u8(
        pool2_out, L3, C2, C3, 6,
        MPCNN_W_CONV3, MPCNN_B_CONV3, MPCNN_MULT_CONV3,
        conv3_out
    );

    gap_u8(
        conv3_out, L3, C3,
        MPCNN_GAP_MULTIPLIER,
        gap_out
    );

    dense_relu_u8(
        gap_out, C3, C3,
        MPCNN_W_TILE_DENSE,
        MPCNN_B_TILE_DENSE,
        MPCNN_MULT_TILE_DENSE,
        tile_features
    );

    memcpy(features_out, tile_features, C3);
}

static bool MPCNN_predict_reference(float *probability) {
    if (probability == nullptr) return false;

    for (int tile_index = 0; tile_index < GRID * GRID; ++tile_index) {
        process_tile(
            patch_rows,
            all_features + (size_t)tile_index * C3
        );
    }

    dense_relu_u8(
        all_features, FEATURES, HIDDEN,
        MPCNN_W_MLP_DENSE,
        MPCNN_B_MLP_DENSE,
        MPCNN_MULT_MLP_DENSE,
        hidden_out
    );

    int32_t acc = MPCNN_B_OUTPUT[0];

    for (int i = 0; i < HIDDEN; ++i) {
        acc += (int32_t)hidden_out[i] *
               (int32_t)MPCNN_W_OUTPUT[i];
    }

    const float logit = (float)acc * MPCNN_LOGIT_SCALE;
    float p;

    if (logit >= 0.0f) {
        p = 1.0f / (1.0f + expf(-logit));
    } else {
        const float e = expf(logit);
        p = e / (1.0f + e);
    }

    *probability = p;
    return true;
}

static void run_benchmark() {
    uint32_t times[BENCH_RUNS];
    uint64_t total_us = 0;
    uint32_t min_us = UINT32_MAX;
    uint32_t max_us = 0;
    float probability = 0.0f;

    Serial.println("\n====================================");
    Serial.println("MPCNN INT8 BENCHMARK START");
    Serial.println("Synthetic RGB=128 input");
    Serial.println("16 tiles of 45x45x3");
    Serial.println("====================================");

    print_memory("BEFORE INFERENCE");

    Serial.println("\n[WARMUP] INFERENCE START");
    Serial.flush();

    uint32_t t0 = micros();
    bool ok = MPCNN_predict_reference(&probability);
    uint32_t warmup_us = micros() - t0;

    Serial.println("[WARMUP] INFERENCE END");

    if (!ok) {
        Serial.println("ERROR: inference failed.");
        return;
    }

    Serial.printf(
        "Warm-up: %lu us (%.3f ms)\n",
        (unsigned long)warmup_us,
        warmup_us / 1000.0f
    );

    Serial.printf("FIRE probability: %.6f\n", probability);

    for (int i = 0; i < BENCH_RUNS; ++i) {
        Serial.printf(
            "\n[%d/%d] INFERENCE START\n",
            i + 1, BENCH_RUNS
        );
        Serial.flush();

        t0 = micros();
        ok = MPCNN_predict_reference(&probability);
        const uint32_t elapsed = micros() - t0;

        Serial.printf(
            "[%d/%d] INFERENCE END\n",
            i + 1, BENCH_RUNS
        );

        if (!ok) {
            Serial.println("ERROR: inference failed.");
            return;
        }

        times[i] = elapsed;
        total_us += elapsed;

        if (elapsed < min_us) min_us = elapsed;
        if (elapsed > max_us) max_us = elapsed;

        Serial.printf(
            "Time: %lu us (%.3f ms)\n",
            (unsigned long)elapsed,
            elapsed / 1000.0f
        );

        Serial.printf("Probability: %.6f\n", probability);
        Serial.printf("Class (threshold 0.50): %s\n",
            probability >= 0.5f ? "FIRE" : "NOFIRE");
    }

    Serial.println("\n========== RESULTS ==========");

    for (int i = 0; i < BENCH_RUNS; ++i) {
        Serial.printf(
            "Run %d: %lu us (%.3f ms)\n",
            i + 1,
            (unsigned long)times[i],
            times[i] / 1000.0f
        );
    }

    Serial.printf(
        "Average: %.3f ms\n",
        ((double)total_us / BENCH_RUNS) / 1000.0
    );

    Serial.printf("Minimum: %.3f ms\n", min_us / 1000.0f);
    Serial.printf("Maximum: %.3f ms\n", max_us / 1000.0f);

    print_memory("AFTER INFERENCE");
    print_flash();

    Serial.println("\nBENCHMARK COMPLETED");
    Serial.println("Synthetic input cannot be used to measure accuracy.");
}

void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.println();
    Serial.println("HELLO");
    Serial.println("MPCNN INT8 - ESP32");
    Serial.println("Initialization successful.");

    memset(patch_rows, 128, sizeof(patch_rows));

    Serial.printf(
        "Reused input buffer: %u bytes\n",
        (unsigned)sizeof(patch_rows)
    );

    Serial.printf(
        "Static working memory: %u bytes\n",
        (unsigned)scratch_bytes()
    );

    print_flash();
    print_memory("STARTUP");

    run_benchmark();
}

void loop() {
    delay(1000);
}
