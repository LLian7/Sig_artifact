#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>
#include <string.h>

#define DIRECT_SUBSET_MASK_THRESHOLD 262144u
#define RANK_U8_REJECT 0xFFu
#define RANK_U16_REJECT 0xFFFFu
#define PROFILE_INC_OR_REJECT(COUNTS, WINDOW_HIGH, VALUE) \
    do { \
        unsigned int _profile_value = (unsigned int)(VALUE); \
        uint32_t _profile_count = (COUNTS)[_profile_value] + 1u; \
        (COUNTS)[_profile_value] = _profile_count; \
        if ((int)_profile_count > (WINDOW_HIGH)) { \
            return 1; \
        } \
    } while (0)

static PyObject *g_shake128_ctor = NULL;
static PyObject *g_shake256_ctor = NULL;
static PyObject *g_accept_check_fast_hash_len_obj = NULL;
static PyObject *g_accept_check_fast_max_g_bit_obj = NULL;
static PyObject *g_accept_check_fast_window_low_obj = NULL;
static PyObject *g_accept_check_fast_window_high_obj = NULL;
static Py_ssize_t g_accept_check_fast_hash_len = 0;
static int g_accept_check_fast_max_g_bit = 0;
static int g_accept_check_fast_max_g_value = 0;
static int g_accept_check_fast_window_low = 0;
static int g_accept_check_fast_window_high = 0;
static Py_ssize_t g_accept_check_fast_expected_len = 0;
static PyObject *g_w4_accept_check_fast_hash_len_obj = NULL;
static PyObject *g_w4_accept_check_fast_window_low_obj = NULL;
static PyObject *g_w4_accept_check_fast_window_high_obj = NULL;
static Py_ssize_t g_w4_accept_check_fast_hash_len = 0;
static int g_w4_accept_check_fast_window_low = 0;
static int g_w4_accept_check_fast_window_high = 0;
static Py_ssize_t g_w4_accept_check_fast_expected_len = 0;
static const char g_hy_prefix[] = "ValStrictISP/HY/";
static const char g_sample_seed_prefix[] = "ValStrictISP/SamplePosition/XOF/";
static uint64_t g_w2_byte_count_lane01[256];
static uint64_t g_w2_byte_count_lane23[256];
static uint64_t g_w2_prefix_byte_count_lane01[4][256];
static uint64_t g_w2_prefix_byte_count_lane23[4][256];
static uint64_t g_comb_table[65][65];
static unsigned char g_rank_byte_lengths[65][65];
static __uint128_t g_rank_thresholds[65][65];
static uint64_t *g_subset_mask_rows[65][65];
static unsigned char g_rank_u8_rows[9][9][256];
static uint16_t *g_rank_u16_rows[65][65];

typedef struct {
    uint64_t block_size;
    uint64_t right_combinations;
    uint64_t *left_row;
    uint64_t *right_row;
} SplitSubsetStep;

typedef struct {
    unsigned int left_size;
    unsigned int step_count;
    SplitSubsetStep steps[65];
} SplitSubsetPlan;

static SplitSubsetPlan *g_split_subset_plans[65][65];

static inline int
counts_within_window(
    const uint32_t *counts,
    int max_g_value,
    int window_low,
    int window_high
);

static void
init_count_tables(void)
{
    int byte;
    for (byte = 0; byte < 256; ++byte) {
        int value;
        int counts[4] = {0, 0, 0, 0};
        for (value = 0; value < 4; ++value) {
            int count = 0;
            int shift_index;
            static const int shifts[4] = {6, 4, 2, 0};
            for (shift_index = 0; shift_index < 4; ++shift_index) {
                if (((byte >> shifts[shift_index]) & 0x03) == value) {
                    count += 1;
                }
            }
            counts[value] = count;
        }
        g_w2_byte_count_lane01[byte] = ((uint64_t)(uint32_t)counts[0]) | (((uint64_t)(uint32_t)counts[1]) << 32);
        g_w2_byte_count_lane23[byte] = ((uint64_t)(uint32_t)counts[2]) | (((uint64_t)(uint32_t)counts[3]) << 32);
    }

    for (byte = 0; byte < 256; ++byte) {
        int pair_count;
        for (pair_count = 1; pair_count <= 3; ++pair_count) {
            int value;
            int counts[4] = {0, 0, 0, 0};
            for (value = 0; value < 4; ++value) {
                int count = 0;
                int shift_index;
                static const int shifts[4] = {6, 4, 2, 0};
                for (shift_index = 0; shift_index < pair_count; ++shift_index) {
                    if (((byte >> shifts[shift_index]) & 0x03) == value) {
                        count += 1;
                    }
                }
                counts[value] = count;
            }
            g_w2_prefix_byte_count_lane01[pair_count][byte] =
                ((uint64_t)(uint32_t)counts[0]) | (((uint64_t)(uint32_t)counts[1]) << 32);
            g_w2_prefix_byte_count_lane23[pair_count][byte] =
                ((uint64_t)(uint32_t)counts[2]) | (((uint64_t)(uint32_t)counts[3]) << 32);
        }
    }
}

static void
init_comb_table(void)
{
    unsigned int n;
    memset(g_comb_table, 0, sizeof(g_comb_table));
    memset(g_rank_byte_lengths, 0, sizeof(g_rank_byte_lengths));
    memset(g_rank_thresholds, 0, sizeof(g_rank_thresholds));
    memset(g_rank_u8_rows, RANK_U8_REJECT, sizeof(g_rank_u8_rows));
    g_comb_table[0][0] = 1;
    for (n = 1; n <= 64; ++n) {
        unsigned int k;
        g_comb_table[n][0] = 1;
        g_comb_table[n][n] = 1;
        for (k = 1; k < n; ++k) {
            g_comb_table[n][k] = g_comb_table[n - 1][k - 1] + g_comb_table[n - 1][k];
        }
    }
    for (n = 0; n <= 64; ++n) {
        unsigned int k;
        for (k = 0; k <= n; ++k) {
            uint64_t bound = g_comb_table[n][k];
            unsigned char byte_len = 1;
            __uint128_t upper;
            while (byte_len < 8 && bound > ((((uint64_t)1) << (8 * byte_len)) - 1u)) {
                byte_len += 1;
            }
            upper = ((__uint128_t)1) << (8 * byte_len);
            g_rank_byte_lengths[n][k] = byte_len;
            g_rank_thresholds[n][k] = upper - (upper % (__uint128_t)bound);
        }
    }
    for (n = 0; n <= 8; ++n) {
        unsigned int k;
        for (k = 0; k <= n; ++k) {
            uint64_t bound = g_comb_table[n][k];
            unsigned int threshold = 0;
            unsigned int candidate;
            if (bound == 0 || bound > 255) {
                continue;
            }
            threshold = 256u - (256u % (unsigned int)bound);
            for (candidate = 0; candidate < 256; ++candidate) {
                if (candidate < threshold) {
                    g_rank_u8_rows[n][k][candidate] = (unsigned char)(candidate % (unsigned int)bound);
                }
            }
        }
    }
}

static int
init_subset_mask_tables(void)
{
    memset(g_subset_mask_rows, 0, sizeof(g_subset_mask_rows));
    memset(g_rank_u16_rows, 0, sizeof(g_rank_u16_rows));
    memset(g_split_subset_plans, 0, sizeof(g_split_subset_plans));
    return 1;
}

static void
fill_subset_mask_row(
    uint64_t *row,
    uint64_t *rank,
    unsigned int n,
    unsigned int start,
    unsigned int remaining,
    uint64_t mask
)
{
    unsigned int position;

    if (remaining == 0) {
        row[*rank] = mask;
        *rank += 1;
        return;
    }

    for (position = start; position <= n - remaining; ++position) {
        fill_subset_mask_row(
            row,
            rank,
            n,
            position + 1,
            remaining - 1,
            mask | (((uint64_t)1) << position)
        );
    }
}

static uint64_t *
ensure_subset_mask_row(unsigned int n, unsigned int count)
{
    uint64_t subset_count;
    uint64_t *row;
    uint64_t rank = 0;

    if (n > 64 || count > n) {
        PyErr_SetString(PyExc_ValueError, "invalid subset-mask table dimensions");
        return NULL;
    }

    row = g_subset_mask_rows[n][count];
    if (row != NULL) {
        return row;
    }

    subset_count = g_comb_table[n][count];
    if (subset_count == 0 || subset_count > DIRECT_SUBSET_MASK_THRESHOLD) {
        return NULL;
    }

    row = (uint64_t *)PyMem_Calloc((size_t)subset_count, sizeof(uint64_t));
    if (row == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    fill_subset_mask_row(row, &rank, n, 0, count, 0);
    g_subset_mask_rows[n][count] = row;
    return row;
}

static uint16_t *
ensure_rank_u16_row(unsigned int n, unsigned int count)
{
    uint64_t subset_count;
    uint16_t *row;
    unsigned int candidate;
    unsigned int threshold;

    if (n > 64 || count > n) {
        PyErr_SetString(PyExc_ValueError, "invalid rank table dimensions");
        return NULL;
    }

    row = g_rank_u16_rows[n][count];
    if (row != NULL) {
        return row;
    }

    subset_count = g_comb_table[n][count];
    if (subset_count == 0 || subset_count > RANK_U16_REJECT || g_rank_byte_lengths[n][count] != 2) {
        return NULL;
    }

    row = (uint16_t *)PyMem_Malloc(65536u * sizeof(uint16_t));
    if (row == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    threshold = 65536u - (65536u % (unsigned int)subset_count);
    for (candidate = 0; candidate < 65536u; ++candidate) {
        row[candidate] = (
            candidate < threshold
            ? (uint16_t)(candidate % (unsigned int)subset_count)
            : (uint16_t)RANK_U16_REJECT
        );
    }

    g_rank_u16_rows[n][count] = row;
    return row;
}

static inline int
tz_u64(uint64_t value)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(value);
#else
    int offset = 0;
    while ((value & 1u) == 0) {
        value >>= 1;
        offset += 1;
    }
    return offset;
#endif
}

static void
profile_counts_w4_lanes(
    const unsigned char *partition_value,
    Py_ssize_t partition_len,
    Py_ssize_t hash_len,
    uint64_t *count_lane01,
    uint64_t *count_lane23
)
{
    Py_ssize_t full_bytes = hash_len / 8;
    int prefix_pairs = (int)((hash_len % 8) / 2);
    Py_ssize_t index;
    uint64_t lane01 = 0;
    uint64_t lane23 = 0;

    (void)partition_len;

    for (index = 0; index < full_bytes; ++index) {
        unsigned char byte = partition_value[index];
        lane01 += g_w2_byte_count_lane01[byte];
        lane23 += g_w2_byte_count_lane23[byte];
    }

    if (prefix_pairs > 0) {
        unsigned char byte = partition_value[full_bytes];
        lane01 += g_w2_prefix_byte_count_lane01[prefix_pairs][byte];
        lane23 += g_w2_prefix_byte_count_lane23[prefix_pairs][byte];
    }

    *count_lane01 = lane01;
    *count_lane23 = lane23;
}

static void
profile_counts_w4(
    const unsigned char *partition_value,
    Py_ssize_t partition_len,
    Py_ssize_t hash_len,
    uint32_t counts[4]
)
{
    uint64_t count_lane01 = 0;
    uint64_t count_lane23 = 0;
    profile_counts_w4_lanes(partition_value, partition_len, hash_len, &count_lane01, &count_lane23);
    counts[0] = (uint32_t)count_lane01;
    counts[1] = (uint32_t)(count_lane01 >> 32);
    counts[2] = (uint32_t)count_lane23;
    counts[3] = (uint32_t)(count_lane23 >> 32);
}

static int
profile_counts_w4_checked(
    const unsigned char *partition_value,
    Py_ssize_t partition_len,
    Py_ssize_t hash_len,
    uint32_t counts[4],
    int window_low,
    int window_high,
    int *accepted
)
{
    Py_ssize_t full_bytes = hash_len / 8;
    int prefix_pairs = (int)((hash_len % 8) / 2);
    Py_ssize_t index;
    uint64_t lane01 = 0;
    uint64_t lane23 = 0;

    (void)partition_len;

    *accepted = 0;
    if (window_high < 0) {
        return 1;
    }

    for (index = 0; index < full_bytes; ++index) {
        unsigned char byte = partition_value[index];
        lane01 += g_w2_byte_count_lane01[byte];
        lane23 += g_w2_byte_count_lane23[byte];
        if (
            (int)(uint32_t)lane01 > window_high
            || (int)(uint32_t)(lane01 >> 32) > window_high
            || (int)(uint32_t)lane23 > window_high
            || (int)(uint32_t)(lane23 >> 32) > window_high
        ) {
            return 1;
        }
    }

    if (prefix_pairs > 0) {
        unsigned char byte = partition_value[full_bytes];
        lane01 += g_w2_prefix_byte_count_lane01[prefix_pairs][byte];
        lane23 += g_w2_prefix_byte_count_lane23[prefix_pairs][byte];
        if (
            (int)(uint32_t)lane01 > window_high
            || (int)(uint32_t)(lane01 >> 32) > window_high
            || (int)(uint32_t)lane23 > window_high
            || (int)(uint32_t)(lane23 >> 32) > window_high
        ) {
            return 1;
        }
    }

    counts[0] = (uint32_t)lane01;
    counts[1] = (uint32_t)(lane01 >> 32);
    counts[2] = (uint32_t)lane23;
    counts[3] = (uint32_t)(lane23 >> 32);
    *accepted = (
        (int)counts[0] >= window_low && (int)counts[0] <= window_high
        && (int)counts[1] >= window_low && (int)counts[1] <= window_high
        && (int)counts[2] >= window_low && (int)counts[2] <= window_high
        && (int)counts[3] >= window_low && (int)counts[3] <= window_high
    );
    return 1;
}

static inline uint64_t
read_be_u64(const unsigned char *data, Py_ssize_t byte_len)
{
    switch (byte_len) {
        case 1:
            return (uint64_t)data[0];
        case 2:
            return ((uint64_t)data[0] << 8) | (uint64_t)data[1];
        case 3:
            return ((uint64_t)data[0] << 16) | ((uint64_t)data[1] << 8) | (uint64_t)data[2];
        case 4:
            return ((uint64_t)data[0] << 24)
                | ((uint64_t)data[1] << 16)
                | ((uint64_t)data[2] << 8)
                | (uint64_t)data[3];
        case 5:
            return ((uint64_t)data[0] << 32)
                | ((uint64_t)data[1] << 24)
                | ((uint64_t)data[2] << 16)
                | ((uint64_t)data[3] << 8)
                | (uint64_t)data[4];
        case 6:
            return ((uint64_t)data[0] << 40)
                | ((uint64_t)data[1] << 32)
                | ((uint64_t)data[2] << 24)
                | ((uint64_t)data[3] << 16)
                | ((uint64_t)data[4] << 8)
                | (uint64_t)data[5];
        case 7:
            return ((uint64_t)data[0] << 48)
                | ((uint64_t)data[1] << 40)
                | ((uint64_t)data[2] << 32)
                | ((uint64_t)data[3] << 24)
                | ((uint64_t)data[4] << 16)
                | ((uint64_t)data[5] << 8)
                | (uint64_t)data[6];
        default:
            return ((uint64_t)data[0] << 56)
                | ((uint64_t)data[1] << 48)
                | ((uint64_t)data[2] << 40)
                | ((uint64_t)data[3] << 32)
                | ((uint64_t)data[4] << 24)
                | ((uint64_t)data[5] << 16)
                | ((uint64_t)data[6] << 8)
                | (uint64_t)data[7];
    }
}

static inline int
read_random_subset_rank(
    const unsigned char *random_bytes,
    Py_ssize_t random_len,
    Py_ssize_t *offset,
    int partition_num,
    int count,
    uint64_t subset_count,
    Py_ssize_t byte_len,
    __uint128_t threshold,
    uint64_t *rank
)
{
    if (byte_len == 1 && partition_num <= 8 && subset_count <= RANK_U8_REJECT) {
        while (1) {
            unsigned char mapped;
            if (*offset >= random_len) {
                PyErr_SetString(PyExc_ValueError, "insufficient random bytes");
                return 0;
            }
            mapped = g_rank_u8_rows[partition_num][count][random_bytes[*offset]];
            *offset += 1;
            if (mapped != RANK_U8_REJECT) {
                *rank = (uint64_t)mapped;
                return 1;
            }
        }
    }

    if (byte_len == 2 && subset_count <= RANK_U16_REJECT) {
        uint16_t *rank_row = ensure_rank_u16_row((unsigned int)partition_num, (unsigned int)count);
        if (rank_row != NULL) {
            while (1) {
                unsigned int candidate;
                uint16_t mapped;
                if (*offset + 2 > random_len) {
                    PyErr_SetString(PyExc_ValueError, "insufficient random bytes");
                    return 0;
                }
                candidate = ((unsigned int)random_bytes[*offset] << 8)
                    | (unsigned int)random_bytes[*offset + 1];
                *offset += 2;
                mapped = rank_row[candidate];
                if (mapped != RANK_U16_REJECT) {
                    *rank = (uint64_t)mapped;
                    return 1;
                }
            }
        }
        if (PyErr_Occurred()) {
            return 0;
        }
    }

    if (byte_len == 3 && subset_count <= 0xFFFFFFu) {
        uint32_t threshold32 = (uint32_t)threshold;
        uint32_t subset_count32 = (uint32_t)subset_count;
        while (1) {
            uint32_t candidate;
            if (*offset + 3 > random_len) {
                PyErr_SetString(PyExc_ValueError, "insufficient random bytes");
                return 0;
            }
            candidate = ((uint32_t)random_bytes[*offset] << 16)
                | ((uint32_t)random_bytes[*offset + 1] << 8)
                | (uint32_t)random_bytes[*offset + 2];
            *offset += 3;
            if (candidate < threshold32) {
                *rank = (uint64_t)(candidate % subset_count32);
                return 1;
            }
        }
    }

    if (byte_len == 4 && subset_count <= 0xFFFFFFFFu) {
        uint64_t threshold32 = (uint64_t)threshold;
        uint32_t subset_count32 = (uint32_t)subset_count;
        while (1) {
            uint32_t candidate;
            if (*offset + 4 > random_len) {
                PyErr_SetString(PyExc_ValueError, "insufficient random bytes");
                return 0;
            }
            candidate = ((uint32_t)random_bytes[*offset] << 24)
                | ((uint32_t)random_bytes[*offset + 1] << 16)
                | ((uint32_t)random_bytes[*offset + 2] << 8)
                | (uint32_t)random_bytes[*offset + 3];
            *offset += 4;
            if ((uint64_t)candidate < threshold32) {
                *rank = (uint64_t)(candidate % subset_count32);
                return 1;
            }
        }
    }

    while (1) {
        Py_ssize_t end = *offset + byte_len;
        uint64_t candidate;
        if (end > random_len) {
            PyErr_SetString(PyExc_ValueError, "insufficient random bytes");
            return 0;
        }
        candidate = read_be_u64(random_bytes + *offset, byte_len);
        *offset = end;
        if ((__uint128_t)candidate < threshold) {
            *rank = candidate % subset_count;
            return 1;
        }
    }
}

static PyObject *
shake_digest(PyObject *ctor, PyObject *seed_material, Py_ssize_t output_len)
{
    PyObject *hash_obj = PyObject_CallFunctionObjArgs(ctor, seed_material, NULL);
    PyObject *digest_method = NULL;
    PyObject *digest = NULL;

    if (hash_obj == NULL) {
        return NULL;
    }

    digest_method = PyObject_GetAttrString(hash_obj, "digest");
    if (digest_method == NULL) {
        Py_DECREF(hash_obj);
        return NULL;
    }

    digest = PyObject_CallFunction(digest_method, "n", output_len);
    Py_DECREF(digest_method);
    Py_DECREF(hash_obj);
    if (digest == NULL) {
        return NULL;
    }
    if (!PyBytes_Check(digest)) {
        Py_DECREF(digest);
        PyErr_SetString(PyExc_TypeError, "digest() did not return bytes");
        return NULL;
    }
    return digest;
}

static void
write_be_u64(unsigned char *destination, uint64_t value)
{
    int shift;
    for (shift = 56; shift >= 0; shift -= 8) {
        *destination++ = (unsigned char)((value >> shift) & 0xFFu);
    }
}

static void
write_low_aligned_payload(
    unsigned char *destination,
    const unsigned char *partition_value,
    Py_ssize_t partition_len,
    int extra_bits
)
{
    Py_ssize_t index;

    if (extra_bits == 0) {
        memcpy(destination, partition_value, (size_t)partition_len);
        return;
    }

    {
        unsigned int carry = 0;
        unsigned int carry_mask = ((unsigned int)1 << extra_bits) - 1u;
        int left_shift = 8 - extra_bits;
        for (index = 0; index < partition_len; ++index) {
            unsigned int byte = partition_value[index];
            destination[index] = (unsigned char)((carry << left_shift) | (byte >> extra_bits));
            carry = byte & carry_mask;
        }
    }
}

static PyObject *
build_default_seed_material_for_bytes(
    const unsigned char *partition_value,
    Py_ssize_t partition_len,
    Py_ssize_t hash_len,
    int use_shake128
)
{
    const char *hash_name = use_shake128 ? "shake_128" : "shake_256";
    Py_ssize_t hash_name_len = use_shake128 ? 9 : 9;
    Py_ssize_t hy_prefix_len = (Py_ssize_t)(sizeof(g_hy_prefix) - 1);
    Py_ssize_t sample_prefix_len = (Py_ssize_t)(sizeof(g_sample_seed_prefix) - 1);
    Py_ssize_t serialized_len = 8 + partition_len;
    Py_ssize_t hy_input_len = hy_prefix_len + hash_name_len + 1 + serialized_len;
    Py_ssize_t seed_material_len = sample_prefix_len + hash_name_len + 1 + 64;
    int extra_bits = (int)(8 * partition_len - hash_len);
    PyObject *hy_input = NULL;
    PyObject *seed_digest = NULL;
    PyObject *seed_material = NULL;
    unsigned char *hy_buffer;
    unsigned char *seed_buffer;

    if (extra_bits < 0 || extra_bits > 7) {
        PyErr_SetString(PyExc_ValueError, "invalid hash_len for partition byte length");
        return NULL;
    }

    hy_input = PyBytes_FromStringAndSize(NULL, hy_input_len);
    if (hy_input == NULL) {
        return NULL;
    }
    hy_buffer = (unsigned char *)PyBytes_AS_STRING(hy_input);
    memcpy(hy_buffer, g_hy_prefix, (size_t)hy_prefix_len);
    hy_buffer += hy_prefix_len;
    memcpy(hy_buffer, hash_name, (size_t)hash_name_len);
    hy_buffer += hash_name_len;
    *hy_buffer++ = '/';
    write_be_u64(hy_buffer, (uint64_t)hash_len);
    hy_buffer += 8;
    write_low_aligned_payload(hy_buffer, partition_value, partition_len, extra_bits);

    seed_digest = shake_digest(use_shake128 ? g_shake128_ctor : g_shake256_ctor, hy_input, 64);
    Py_DECREF(hy_input);
    if (seed_digest == NULL) {
        return NULL;
    }

    seed_material = PyBytes_FromStringAndSize(NULL, seed_material_len);
    if (seed_material == NULL) {
        Py_DECREF(seed_digest);
        return NULL;
    }
    seed_buffer = (unsigned char *)PyBytes_AS_STRING(seed_material);
    memcpy(seed_buffer, g_sample_seed_prefix, (size_t)sample_prefix_len);
    seed_buffer += sample_prefix_len;
    memcpy(seed_buffer, hash_name, (size_t)hash_name_len);
    seed_buffer += hash_name_len;
    *seed_buffer++ = '/';
    memcpy(seed_buffer, PyBytes_AS_STRING(seed_digest), 64);

    Py_DECREF(seed_digest);
    return seed_material;
}

static inline int
mark_subset_from_rank(
    uint8_t *group_masks,
    int partition_num,
    int count,
    uint64_t rank,
    uint8_t value_bit
)
{
    int position;
    int remaining = count;
    uint64_t current_rank = rank;

    for (position = 0; position < partition_num && remaining > 0; ++position) {
        uint64_t include_count =
            g_comb_table[partition_num - position - 1][remaining - 1];
        if (current_rank < include_count) {
            group_masks[position] |= value_bit;
            remaining -= 1;
        } else {
            current_rank -= include_count;
        }
    }

    return remaining == 0;
}

static inline int
mark_subset_from_rank_u32(
    uint8_t *group_masks,
    int partition_num,
    int count,
    uint32_t rank,
    uint8_t value_bit
)
{
    int position;
    int remaining = count;
    uint32_t current_rank = rank;

    for (position = 0; position < partition_num && remaining > 0; ++position) {
        uint32_t include_count =
            (uint32_t)g_comb_table[partition_num - position - 1][remaining - 1];
        if (current_rank < include_count) {
            group_masks[position] |= value_bit;
            remaining -= 1;
        } else {
            current_rank -= include_count;
        }
    }

    return remaining == 0;
}

static inline void
mark_position_mask_u8(
    uint8_t *group_masks,
    uint64_t position_mask,
    int base_position,
    uint8_t value_bit
)
{
    while (position_mask != 0) {
        group_masks[base_position + tz_u64(position_mask)] |= value_bit;
        position_mask &= position_mask - 1u;
    }
}

static SplitSubsetPlan *
ensure_split_subset_plan(unsigned int universe_size, unsigned int count)
{
    SplitSubsetPlan *plan;
    unsigned int left_size;
    unsigned int right_size;
    unsigned int min_left;
    unsigned int max_left;
    unsigned int left_count;
    unsigned int step_index = 0;

    if (universe_size > 64 || count > universe_size) {
        PyErr_SetString(PyExc_ValueError, "invalid split subset dimensions");
        return NULL;
    }
    if (count == 0 || universe_size <= 8 || universe_size > 40) {
        return NULL;
    }

    plan = g_split_subset_plans[universe_size][count];
    if (plan != NULL) {
        return plan;
    }

    left_size = universe_size / 2;
    right_size = universe_size - left_size;
    min_left = count > right_size ? count - right_size : 0;
    max_left = count < left_size ? count : left_size;

    plan = (SplitSubsetPlan *)PyMem_Calloc(1, sizeof(SplitSubsetPlan));
    if (plan == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    plan->left_size = left_size;

    for (left_count = min_left; left_count <= max_left; ++left_count) {
        unsigned int right_count = count - left_count;
        uint64_t left_combinations = g_comb_table[left_size][left_count];
        uint64_t right_combinations = g_comb_table[right_size][right_count];
        uint64_t *left_row = ensure_subset_mask_row(left_size, left_count);
        uint64_t *right_row = ensure_subset_mask_row(right_size, right_count);

        if (left_row == NULL || right_row == NULL) {
            PyMem_Free(plan);
            if (PyErr_Occurred()) {
                return NULL;
            }
            PyErr_SetString(PyExc_RuntimeError, "split subset row is unavailable");
            return NULL;
        }

        plan->steps[step_index].block_size = left_combinations * right_combinations;
        plan->steps[step_index].right_combinations = right_combinations;
        plan->steps[step_index].left_row = left_row;
        plan->steps[step_index].right_row = right_row;
        step_index += 1;
    }

    plan->step_count = step_index;
    g_split_subset_plans[universe_size][count] = plan;
    return plan;
}

static inline int
subset_split_mask_from_rank(
    int partition_num,
    int count,
    uint64_t rank,
    uint64_t *position_mask
)
{
    SplitSubsetPlan *plan = ensure_split_subset_plan((unsigned int)partition_num, (unsigned int)count);
    int left_size = partition_num / 2;
    int right_size = partition_num - left_size;
    int min_left = count > right_size ? count - right_size : 0;
    int max_left = count < left_size ? count : left_size;
    int left_count;

    if (plan != NULL) {
        unsigned int index;
        left_size = (int)plan->left_size;
        for (index = 0; index < plan->step_count; ++index) {
            SplitSubsetStep *step = &plan->steps[index];
            if (rank < step->block_size) {
                uint64_t left_rank = rank / step->right_combinations;
                uint64_t right_rank = rank - left_rank * step->right_combinations;
                *position_mask = step->left_row[left_rank] | (step->right_row[right_rank] << left_size);
                return 1;
            }
            rank -= step->block_size;
        }
        return 0;
    }
    if (PyErr_Occurred()) {
        return 0;
    }

    for (left_count = min_left; left_count <= max_left; ++left_count) {
        int right_count = count - left_count;
        uint64_t left_combinations = g_comb_table[left_size][left_count];
        uint64_t right_combinations = g_comb_table[right_size][right_count];
        uint64_t block_size = left_combinations * right_combinations;

        if (rank < block_size) {
            uint64_t left_rank = rank / right_combinations;
            uint64_t right_rank = rank - left_rank * right_combinations;
            uint64_t *left_row = ensure_subset_mask_row((unsigned int)left_size, (unsigned int)left_count);
            uint64_t *right_row = ensure_subset_mask_row((unsigned int)right_size, (unsigned int)right_count);

            if (left_row == NULL || right_row == NULL) {
                if (PyErr_Occurred()) {
                    return 0;
                }
                PyErr_SetString(PyExc_RuntimeError, "split subset row is unavailable");
                return 0;
            }
            *position_mask = left_row[left_rank] | (right_row[right_rank] << left_size);
            return 1;
        }
        rank -= block_size;
    }

    return 0;
}

static inline int
mark_subset_split_u8_from_rank(
    uint8_t *group_masks,
    int partition_num,
    int count,
    uint64_t rank,
    uint8_t value_bit
)
{
    uint64_t position_mask = 0;
    if (!subset_split_mask_from_rank(partition_num, count, rank, &position_mask)) {
        return 0;
    }
    mark_position_mask_u8(group_masks, position_mask, 0, value_bit);
    return 1;
}

static PyObject *
group_masks_to_groups(const uint8_t *group_masks, int partition_num)
{
    PyObject *groups = PyList_New(partition_num);
    int position;

    if (groups == NULL) {
        return NULL;
    }

    for (position = 0; position < partition_num; ++position) {
        uint8_t mask = group_masks[position];
        Py_ssize_t subgroup_len = (mask & 1u) + ((mask >> 1) & 1u) + ((mask >> 2) & 1u) + ((mask >> 3) & 1u);
        PyObject *subgroup = PyList_New(subgroup_len);
        Py_ssize_t offset = 0;
        int value;

        if (subgroup == NULL) {
            Py_DECREF(groups);
            return NULL;
        }

        for (value = 0; value < 4; ++value) {
            if (mask & (1u << value)) {
                PyObject *py_value = PyLong_FromLong(value);
                if (py_value == NULL) {
                    Py_DECREF(subgroup);
                    Py_DECREF(groups);
                    return NULL;
                }
                PyList_SET_ITEM(subgroup, offset, py_value);
                offset += 1;
            }
        }

        PyList_SET_ITEM(groups, position, subgroup);
    }

    return groups;
}

static int
profile_counts_native(
    const unsigned char *partition_value,
    Py_ssize_t partition_len,
    Py_ssize_t hash_len,
    int max_g_bit,
    int max_g_value,
    uint32_t counts[64]
)
{
    memset(counts, 0, (size_t)max_g_value * sizeof(uint32_t));

    if (max_g_bit == 2 && max_g_value == 4) {
        profile_counts_w4(partition_value, partition_len, hash_len, counts);
        return 1;
    }

    if (max_g_bit == 3 && max_g_value == 8) {
        Py_ssize_t block_count = hash_len / 3;
        Py_ssize_t full_groups = block_count / 8;
        Py_ssize_t group;
        Py_ssize_t index = 0;
        for (group = 0; group < full_groups; ++group, index += 3) {
            unsigned char b0 = partition_value[index];
            unsigned char b1 = partition_value[index + 1];
            unsigned char b2 = partition_value[index + 2];
            counts[b0 >> 5] += 1;
            counts[(b0 >> 2) & 0x07u] += 1;
            counts[((b0 & 0x03u) << 1) | (b1 >> 7)] += 1;
            counts[(b1 >> 4) & 0x07u] += 1;
            counts[(b1 >> 1) & 0x07u] += 1;
            counts[((b1 & 0x01u) << 2) | (b2 >> 6)] += 1;
            counts[(b2 >> 3) & 0x07u] += 1;
            counts[b2 & 0x07u] += 1;
        }
        if ((block_count & 7) != 0) {
            Py_ssize_t block_index;
            Py_ssize_t tail_blocks = block_count & 7;
            uint32_t accumulator = 0;
            int accumulator_bits = 0;
            for (block_index = 0; block_index < tail_blocks; ++block_index) {
                unsigned int value;
                int shift;
                while (accumulator_bits < 3) {
                    if (index >= partition_len) {
                        PyErr_SetString(PyExc_ValueError, "partition_value ended before hash_len bits");
                        return 0;
                    }
                    accumulator = (accumulator << 8) | (uint32_t)partition_value[index++];
                    accumulator_bits += 8;
                }
                shift = accumulator_bits - 3;
                value = (unsigned int)((accumulator >> shift) & 0x07u);
                counts[value] += 1;
                accumulator_bits = shift;
                accumulator &= (accumulator_bits == 0) ? 0u : (((uint32_t)1u << accumulator_bits) - 1u);
            }
        }
        return 1;
    }

    if (max_g_bit == 4 && max_g_value == 16 && (hash_len % 8) == 0) {
        Py_ssize_t index;
        (void)partition_len;
        for (index = 0; index < hash_len / 8; ++index) {
            unsigned char byte = partition_value[index];
            counts[byte >> 4] += 1;
            counts[byte & 0x0F] += 1;
        }
        return 1;
    }

    if (max_g_bit == 5 && max_g_value == 32) {
        Py_ssize_t block_count = hash_len / 5;
        Py_ssize_t full_groups = block_count / 8;
        Py_ssize_t group;
        Py_ssize_t index = 0;
        for (group = 0; group < full_groups; ++group, index += 5) {
            unsigned char b0 = partition_value[index];
            unsigned char b1 = partition_value[index + 1];
            unsigned char b2 = partition_value[index + 2];
            unsigned char b3 = partition_value[index + 3];
            unsigned char b4 = partition_value[index + 4];
            counts[b0 >> 3] += 1;
            counts[((b0 & 0x07u) << 2) | (b1 >> 6)] += 1;
            counts[(b1 >> 1) & 0x1Fu] += 1;
            counts[((b1 & 0x01u) << 4) | (b2 >> 4)] += 1;
            counts[((b2 & 0x0Fu) << 1) | (b3 >> 7)] += 1;
            counts[(b3 >> 2) & 0x1Fu] += 1;
            counts[((b3 & 0x03u) << 3) | (b4 >> 5)] += 1;
            counts[b4 & 0x1Fu] += 1;
        }
        if ((block_count & 7) != 0) {
            Py_ssize_t block_index;
            Py_ssize_t tail_blocks = block_count & 7;
            uint64_t accumulator = 0;
            int accumulator_bits = 0;
            for (block_index = 0; block_index < tail_blocks; ++block_index) {
                unsigned int value;
                int shift;
                while (accumulator_bits < 5) {
                    if (index >= partition_len) {
                        PyErr_SetString(PyExc_ValueError, "partition_value ended before hash_len bits");
                        return 0;
                    }
                    accumulator = (accumulator << 8) | (uint64_t)partition_value[index++];
                    accumulator_bits += 8;
                }
                shift = accumulator_bits - 5;
                value = (unsigned int)((accumulator >> shift) & 0x1Fu);
                counts[value] += 1;
                accumulator_bits = shift;
                accumulator &= (accumulator_bits == 0) ? 0u : (((uint64_t)1u << accumulator_bits) - 1u);
            }
        }
        return 1;
    }

    if (max_g_bit == 6 && max_g_value == 64) {
        Py_ssize_t block_count = hash_len / 6;
        Py_ssize_t full_groups = block_count / 4;
        Py_ssize_t group;
        Py_ssize_t index = 0;
        for (group = 0; group < full_groups; ++group, index += 3) {
            unsigned char b0 = partition_value[index];
            unsigned char b1 = partition_value[index + 1];
            unsigned char b2 = partition_value[index + 2];
            counts[b0 >> 2] += 1;
            counts[((b0 & 0x03u) << 4) | (b1 >> 4)] += 1;
            counts[((b1 & 0x0Fu) << 2) | (b2 >> 6)] += 1;
            counts[b2 & 0x3Fu] += 1;
        }
        if ((block_count & 3) != 0) {
            Py_ssize_t block_index;
            Py_ssize_t tail_blocks = block_count & 3;
            uint32_t accumulator = 0;
            int accumulator_bits = 0;
            for (block_index = 0; block_index < tail_blocks; ++block_index) {
                unsigned int value;
                int shift;
                while (accumulator_bits < 6) {
                    if (index >= partition_len) {
                        PyErr_SetString(PyExc_ValueError, "partition_value ended before hash_len bits");
                        return 0;
                    }
                    accumulator = (accumulator << 8) | (uint32_t)partition_value[index++];
                    accumulator_bits += 8;
                }
                shift = accumulator_bits - 6;
                value = (unsigned int)((accumulator >> shift) & 0x3Fu);
                counts[value] += 1;
                accumulator_bits = shift;
                accumulator &= (accumulator_bits == 0) ? 0u : (((uint32_t)1u << accumulator_bits) - 1u);
            }
        }
        return 1;
    }

    {
        Py_ssize_t block_count = hash_len / max_g_bit;
        Py_ssize_t block_index;
        Py_ssize_t byte_index = 0;
        uint64_t accumulator = 0;
        int accumulator_bits = 0;
        uint64_t value_mask = (((uint64_t)1) << max_g_bit) - 1u;

        for (block_index = 0; block_index < block_count; ++block_index) {
            unsigned int value;
            int shift;
            while (accumulator_bits < max_g_bit) {
                if (byte_index >= partition_len) {
                    PyErr_SetString(PyExc_ValueError, "partition_value ended before hash_len bits");
                    return 0;
                }
                accumulator = (accumulator << 8) | (uint64_t)partition_value[byte_index++];
                accumulator_bits += 8;
            }
            shift = accumulator_bits - max_g_bit;
            value = (unsigned int)((accumulator >> shift) & value_mask);
            counts[value] += 1;
            accumulator_bits = shift;
            if (accumulator_bits == 0) {
                accumulator = 0;
            } else {
                accumulator &= (((uint64_t)1) << accumulator_bits) - 1u;
            }
        }
    }

    return 1;
}

static int
counts_within_window(
    const uint32_t *counts,
    int max_g_value,
    int window_low,
    int window_high
)
{
    int value;
    for (value = 0; value < max_g_value; ++value) {
        if ((int)counts[value] < window_low || (int)counts[value] > window_high) {
            return 0;
        }
    }
    return 1;
}

static int
profile_counts_native_checked(
    const unsigned char *partition_value,
    Py_ssize_t partition_len,
    Py_ssize_t hash_len,
    int max_g_bit,
    int max_g_value,
    uint32_t counts[64],
    int window_low,
    int window_high,
    int *accepted
)
{
    if (max_g_bit == 2 && max_g_value == 4) {
        return profile_counts_w4_checked(
            partition_value,
            partition_len,
            hash_len,
            counts,
            window_low,
            window_high,
            accepted
        );
    }

    memset(counts, 0, (size_t)max_g_value * sizeof(uint32_t));
    *accepted = 0;

    if (max_g_bit == 3 && max_g_value == 8) {
        Py_ssize_t block_count = hash_len / 3;
        Py_ssize_t full_groups = block_count / 8;
        Py_ssize_t group;
        Py_ssize_t index = 0;
        for (group = 0; group < full_groups; ++group, index += 3) {
            unsigned char b0 = partition_value[index];
            unsigned char b1 = partition_value[index + 1];
            unsigned char b2 = partition_value[index + 2];
            PROFILE_INC_OR_REJECT(counts, window_high, b0 >> 5);
            PROFILE_INC_OR_REJECT(counts, window_high, (b0 >> 2) & 0x07u);
            PROFILE_INC_OR_REJECT(counts, window_high, ((b0 & 0x03u) << 1) | (b1 >> 7));
            PROFILE_INC_OR_REJECT(counts, window_high, (b1 >> 4) & 0x07u);
            PROFILE_INC_OR_REJECT(counts, window_high, (b1 >> 1) & 0x07u);
            PROFILE_INC_OR_REJECT(counts, window_high, ((b1 & 0x01u) << 2) | (b2 >> 6));
            PROFILE_INC_OR_REJECT(counts, window_high, (b2 >> 3) & 0x07u);
            PROFILE_INC_OR_REJECT(counts, window_high, b2 & 0x07u);
        }
        if ((block_count & 7) != 0) {
            Py_ssize_t block_index;
            Py_ssize_t tail_blocks = block_count & 7;
            uint32_t accumulator = 0;
            int accumulator_bits = 0;
            for (block_index = 0; block_index < tail_blocks; ++block_index) {
                unsigned int value;
                int shift;
                while (accumulator_bits < 3) {
                    if (index >= partition_len) {
                        PyErr_SetString(PyExc_ValueError, "partition_value ended before hash_len bits");
                        return 0;
                    }
                    accumulator = (accumulator << 8) | (uint32_t)partition_value[index++];
                    accumulator_bits += 8;
                }
                shift = accumulator_bits - 3;
                value = (unsigned int)((accumulator >> shift) & 0x07u);
                PROFILE_INC_OR_REJECT(counts, window_high, value);
                accumulator_bits = shift;
                accumulator &= (accumulator_bits == 0) ? 0u : (((uint32_t)1u << accumulator_bits) - 1u);
            }
        }
        *accepted = counts_within_window(counts, max_g_value, window_low, window_high);
        return 1;
    }

    if (max_g_bit == 4 && max_g_value == 16 && (hash_len % 8) == 0) {
        Py_ssize_t index;
        (void)partition_len;
        for (index = 0; index < hash_len / 8; ++index) {
            unsigned char byte = partition_value[index];
            unsigned int high = byte >> 4;
            unsigned int low = byte & 0x0F;
            PROFILE_INC_OR_REJECT(counts, window_high, high);
            PROFILE_INC_OR_REJECT(counts, window_high, low);
        }
        *accepted = counts_within_window(counts, max_g_value, window_low, window_high);
        return 1;
    }

    if (max_g_bit == 5 && max_g_value == 32) {
        Py_ssize_t block_count = hash_len / 5;
        Py_ssize_t full_groups = block_count / 8;
        Py_ssize_t group;
        Py_ssize_t index = 0;
        for (group = 0; group < full_groups; ++group, index += 5) {
            unsigned char b0 = partition_value[index];
            unsigned char b1 = partition_value[index + 1];
            unsigned char b2 = partition_value[index + 2];
            unsigned char b3 = partition_value[index + 3];
            unsigned char b4 = partition_value[index + 4];
            PROFILE_INC_OR_REJECT(counts, window_high, b0 >> 3);
            PROFILE_INC_OR_REJECT(counts, window_high, ((b0 & 0x07u) << 2) | (b1 >> 6));
            PROFILE_INC_OR_REJECT(counts, window_high, (b1 >> 1) & 0x1Fu);
            PROFILE_INC_OR_REJECT(counts, window_high, ((b1 & 0x01u) << 4) | (b2 >> 4));
            PROFILE_INC_OR_REJECT(counts, window_high, ((b2 & 0x0Fu) << 1) | (b3 >> 7));
            PROFILE_INC_OR_REJECT(counts, window_high, (b3 >> 2) & 0x1Fu);
            PROFILE_INC_OR_REJECT(counts, window_high, ((b3 & 0x03u) << 3) | (b4 >> 5));
            PROFILE_INC_OR_REJECT(counts, window_high, b4 & 0x1Fu);
        }
        if ((block_count & 7) != 0) {
            Py_ssize_t block_index;
            Py_ssize_t tail_blocks = block_count & 7;
            uint64_t accumulator = 0;
            int accumulator_bits = 0;
            for (block_index = 0; block_index < tail_blocks; ++block_index) {
                unsigned int value;
                int shift;
                while (accumulator_bits < 5) {
                    if (index >= partition_len) {
                        PyErr_SetString(PyExc_ValueError, "partition_value ended before hash_len bits");
                        return 0;
                    }
                    accumulator = (accumulator << 8) | (uint64_t)partition_value[index++];
                    accumulator_bits += 8;
                }
                shift = accumulator_bits - 5;
                value = (unsigned int)((accumulator >> shift) & 0x1Fu);
                PROFILE_INC_OR_REJECT(counts, window_high, value);
                accumulator_bits = shift;
                accumulator &= (accumulator_bits == 0) ? 0u : (((uint64_t)1u << accumulator_bits) - 1u);
            }
        }
        *accepted = counts_within_window(counts, max_g_value, window_low, window_high);
        return 1;
    }

    if (max_g_bit == 6 && max_g_value == 64) {
        Py_ssize_t block_count = hash_len / 6;
        Py_ssize_t full_groups = block_count / 4;
        Py_ssize_t group;
        Py_ssize_t index = 0;
        for (group = 0; group < full_groups; ++group, index += 3) {
            unsigned char b0 = partition_value[index];
            unsigned char b1 = partition_value[index + 1];
            unsigned char b2 = partition_value[index + 2];
            PROFILE_INC_OR_REJECT(counts, window_high, b0 >> 2);
            PROFILE_INC_OR_REJECT(counts, window_high, ((b0 & 0x03u) << 4) | (b1 >> 4));
            PROFILE_INC_OR_REJECT(counts, window_high, ((b1 & 0x0Fu) << 2) | (b2 >> 6));
            PROFILE_INC_OR_REJECT(counts, window_high, b2 & 0x3Fu);
        }
        if ((block_count & 3) != 0) {
            Py_ssize_t block_index;
            Py_ssize_t tail_blocks = block_count & 3;
            uint32_t accumulator = 0;
            int accumulator_bits = 0;
            for (block_index = 0; block_index < tail_blocks; ++block_index) {
                unsigned int value;
                int shift;
                while (accumulator_bits < 6) {
                    if (index >= partition_len) {
                        PyErr_SetString(PyExc_ValueError, "partition_value ended before hash_len bits");
                        return 0;
                    }
                    accumulator = (accumulator << 8) | (uint32_t)partition_value[index++];
                    accumulator_bits += 8;
                }
                shift = accumulator_bits - 6;
                value = (unsigned int)((accumulator >> shift) & 0x3Fu);
                PROFILE_INC_OR_REJECT(counts, window_high, value);
                accumulator_bits = shift;
                accumulator &= (accumulator_bits == 0) ? 0u : (((uint32_t)1u << accumulator_bits) - 1u);
            }
        }
        *accepted = counts_within_window(counts, max_g_value, window_low, window_high);
        return 1;
    }

    {
        Py_ssize_t block_count = hash_len / max_g_bit;
        Py_ssize_t block_index;
        Py_ssize_t byte_index = 0;
        uint64_t accumulator = 0;
        int accumulator_bits = 0;
        uint64_t value_mask = (((uint64_t)1) << max_g_bit) - 1u;

        for (block_index = 0; block_index < block_count; ++block_index) {
            unsigned int value;
            int shift;
            while (accumulator_bits < max_g_bit) {
                if (byte_index >= partition_len) {
                    PyErr_SetString(PyExc_ValueError, "partition_value ended before hash_len bits");
                    return 0;
                }
                accumulator = (accumulator << 8) | (uint64_t)partition_value[byte_index++];
                accumulator_bits += 8;
            }
            shift = accumulator_bits - max_g_bit;
            value = (unsigned int)((accumulator >> shift) & value_mask);
            PROFILE_INC_OR_REJECT(counts, window_high, value);
            accumulator_bits = shift;
            if (accumulator_bits == 0) {
                accumulator = 0;
            } else {
                accumulator &= (((uint64_t)1) << accumulator_bits) - 1u;
            }
        }
    }

    *accepted = counts_within_window(counts, max_g_value, window_low, window_high);
    return 1;
}

static int
mark_position_mask(
    uint64_t *group_masks,
    uint64_t position_mask,
    int base_position,
    uint64_t value_bit
)
{
    while (position_mask != 0) {
        uint64_t low_bit = position_mask & (~position_mask + 1u);
        int position = tz_u64(low_bit);
        group_masks[base_position + position] |= value_bit;
        position_mask ^= low_bit;
    }
    return 1;
}

static int
mark_subset_split64_from_rank(
    uint64_t *group_masks,
    int partition_num,
    int count,
    uint64_t rank,
    uint64_t value_bit
)
{
    int left_size = partition_num / 2;
    int right_size = partition_num - left_size;
    int min_left = count > right_size ? count - right_size : 0;
    int max_left = count < left_size ? count : left_size;
    int left_count;
    SplitSubsetPlan *plan = ensure_split_subset_plan((unsigned int)partition_num, (unsigned int)count);

    if (plan != NULL) {
        unsigned int index;
        left_size = (int)plan->left_size;
        for (index = 0; index < plan->step_count; ++index) {
            SplitSubsetStep *step = &plan->steps[index];
            if (rank < step->block_size) {
                uint64_t left_rank = rank / step->right_combinations;
                uint64_t right_rank = rank - left_rank * step->right_combinations;
                mark_position_mask(group_masks, step->left_row[left_rank], 0, value_bit);
                mark_position_mask(group_masks, step->right_row[right_rank], left_size, value_bit);
                return 1;
            }
            rank -= step->block_size;
        }
        return 0;
    }
    if (PyErr_Occurred()) {
        return 0;
    }

    for (left_count = min_left; left_count <= max_left; ++left_count) {
        int right_count = count - left_count;
        uint64_t left_combinations = g_comb_table[left_size][left_count];
        uint64_t right_combinations = g_comb_table[right_size][right_count];
        uint64_t block_size = left_combinations * right_combinations;

        if (rank < block_size) {
            uint64_t left_rank = rank / right_combinations;
            uint64_t right_rank = rank - left_rank * right_combinations;
            uint64_t *left_row = ensure_subset_mask_row((unsigned int)left_size, (unsigned int)left_count);
            uint64_t *right_row = ensure_subset_mask_row((unsigned int)right_size, (unsigned int)right_count);

            if (left_row == NULL || right_row == NULL) {
                if (PyErr_Occurred()) {
                    return 0;
                }
                PyErr_SetString(PyExc_RuntimeError, "split subset row is unavailable");
                return 0;
            }
            mark_position_mask(group_masks, left_row[left_rank], 0, value_bit);
            mark_position_mask(group_masks, right_row[right_rank], left_size, value_bit);
            return 1;
        }
        rank -= block_size;
    }

    return 0;
}

static int
mark_subset64_from_rank_offset(
    uint64_t *group_masks,
    int universe_size,
    int count,
    uint64_t rank,
    uint64_t value_bit,
    int base_position
)
{
    uint64_t *row = ensure_subset_mask_row((unsigned int)universe_size, (unsigned int)count);
    if (row != NULL) {
        return mark_position_mask(group_masks, row[rank], base_position, value_bit);
    }
    if (PyErr_Occurred()) {
        return 0;
    }

    {
        int position;
        int remaining = count;
        uint64_t current_rank = rank;

        for (position = 0; position < universe_size && remaining > 0; ++position) {
            uint64_t include_count =
                g_comb_table[universe_size - position - 1][remaining - 1];
            if (current_rank < include_count) {
                group_masks[base_position + position] |= value_bit;
                remaining -= 1;
            } else {
                current_rank -= include_count;
            }
        }

        return remaining == 0;
    }
}

static int
mark_subset64_from_rank(
    uint64_t *group_masks,
    int partition_num,
    int count,
    uint64_t rank,
    uint64_t value_bit
)
{
    return mark_subset64_from_rank_offset(group_masks, partition_num, count, rank, value_bit, 0);
}

static int
prepare_unrank_plan(unsigned int universe_size, unsigned int count)
{
    uint64_t *row;
    uint16_t *rank_row;

    if (count == 0 || count > universe_size || universe_size > 64) {
        return 1;
    }

    if (universe_size > 8 && universe_size <= 40) {
        SplitSubsetPlan *plan = ensure_split_subset_plan(universe_size, count);
        if (plan == NULL && PyErr_Occurred()) {
            return 0;
        }
    }

    row = ensure_subset_mask_row(universe_size, count);
    if (row == NULL && PyErr_Occurred()) {
        return 0;
    }

    rank_row = ensure_rank_u16_row(universe_size, count);
    if (rank_row != NULL) {
        return 1;
    }
    if (PyErr_Occurred()) {
        return 0;
    }

    return 1;
}

static PyObject *
group_masks64_to_python_masks(const uint64_t *group_masks, int partition_num, int max_g_value)
{
    int position;

    if (max_g_value <= 8) {
        PyObject *result = PyBytes_FromStringAndSize(NULL, partition_num);
        char *buffer;

        if (result == NULL) {
            return NULL;
        }
        buffer = PyBytes_AS_STRING(result);
        for (position = 0; position < partition_num; ++position) {
            buffer[position] = (char)(unsigned char)group_masks[position];
        }
        return result;
    }

    {
        PyObject *result = PyTuple_New(partition_num);
        if (result == NULL) {
            return NULL;
        }
        for (position = 0; position < partition_num; ++position) {
            PyObject *mask = PyLong_FromUnsignedLongLong(group_masks[position]);
            if (mask == NULL) {
                Py_DECREF(result);
                return NULL;
            }
            PyTuple_SET_ITEM(result, position, mask);
        }
        return result;
    }
}

static PyObject *
group_masks64_to_packed_bytes(const uint64_t *group_masks, int partition_num, int max_g_value)
{
    int position;
    int byte_len = (max_g_value + 7) / 8;
    PyObject *result = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)partition_num * byte_len);
    unsigned char *buffer;

    if (result == NULL) {
        return NULL;
    }
    buffer = (unsigned char *)PyBytes_AS_STRING(result);
    for (position = 0; position < partition_num; ++position) {
        uint64_t mask = group_masks[position];
        int index;
        for (index = 0; index < byte_len; ++index) {
            buffer[position * byte_len + index] = (unsigned char)((mask >> (8 * index)) & 0xFFu);
        }
    }
    return result;
}

static PyObject *
group_masks64_to_groups(const uint64_t *group_masks, int partition_num, int max_g_value)
{
    PyObject *groups = PyList_New(partition_num);
    int position;

    if (groups == NULL) {
        return NULL;
    }

    for (position = 0; position < partition_num; ++position) {
        uint64_t mask = group_masks[position];
        Py_ssize_t subgroup_len = 0;
        Py_ssize_t offset = 0;
        PyObject *subgroup;
        int value;

        for (value = 0; value < max_g_value; ++value) {
            if (mask & (((uint64_t)1) << value)) {
                subgroup_len += 1;
            }
        }

        subgroup = PyList_New(subgroup_len);
        if (subgroup == NULL) {
            Py_DECREF(groups);
            return NULL;
        }

        for (value = 0; value < max_g_value; ++value) {
            if (mask & (((uint64_t)1) << value)) {
                PyObject *py_value = PyLong_FromLong(value);
                if (py_value == NULL) {
                    Py_DECREF(subgroup);
                    Py_DECREF(groups);
                    return NULL;
                }
                PyList_SET_ITEM(subgroup, offset, py_value);
                offset += 1;
            }
        }

        PyList_SET_ITEM(groups, position, subgroup);
    }

    return groups;
}

static int
counts_fit_direct_subset_rows(const uint32_t counts[64], int max_g_value, int partition_num)
{
    int value;
    for (value = 0; value < max_g_value; ++value) {
        int count = (int)counts[value];
        uint64_t subset_count;
        if (count <= 1) {
            continue;
        }
        if (count > partition_num) {
            return 1;
        }
        subset_count = g_comb_table[partition_num][count];
        if (subset_count == 0 || subset_count > DIRECT_SUBSET_MASK_THRESHOLD) {
            return 0;
        }
    }
    return 1;
}

static PyObject *
sample_native_direct_rows_random_bytes(
    const uint32_t counts[64],
    int max_g_value,
    int partition_num,
    const unsigned char *random_bytes,
    Py_ssize_t random_len,
    int return_group_masks
)
{
    Py_ssize_t offset = 0;
    uint64_t group_masks[64];
    uint64_t *subset_rows[65];
    int value;

    memset(group_masks, 0, (size_t)partition_num * sizeof(group_masks[0]));
    memset(subset_rows, 0, (size_t)(partition_num + 1) * sizeof(subset_rows[0]));

    for (value = 0; value < max_g_value; ++value) {
        int value_count = (int)counts[value];
        uint64_t subset_count;
        Py_ssize_t byte_len;
        __uint128_t threshold;
        uint64_t rank;
        uint64_t value_bit = ((uint64_t)1) << value;
        uint64_t position_mask;

        if (value_count == 0) {
            continue;
        }
        if (value_count > partition_num) {
            PyErr_SetString(PyExc_OverflowError, "binomial coefficient does not fit in uint64");
            return NULL;
        }
        subset_count = g_comb_table[partition_num][value_count];
        byte_len = g_rank_byte_lengths[partition_num][value_count];
        threshold = g_rank_thresholds[partition_num][value_count];

        if (!read_random_subset_rank(
                random_bytes,
                random_len,
                &offset,
                partition_num,
                value_count,
                subset_count,
                byte_len,
                threshold,
                &rank)) {
            return NULL;
        }

        if (value_count == 1) {
            group_masks[rank] |= value_bit;
            continue;
        }

        if (subset_rows[value_count] == NULL) {
            subset_rows[value_count] = ensure_subset_mask_row(
                (unsigned int)partition_num,
                (unsigned int)value_count
            );
            if (subset_rows[value_count] == NULL) {
                if (PyErr_Occurred()) {
                    return NULL;
                }
                PyErr_SetString(PyExc_RuntimeError, "direct subset row is unavailable");
                return NULL;
            }
        }
        position_mask = subset_rows[value_count][rank];
        while (position_mask != 0) {
            uint64_t low_bit = position_mask & (~position_mask + 1u);
            group_masks[tz_u64(low_bit)] |= value_bit;
            position_mask ^= low_bit;
        }
    }

    if (return_group_masks) {
        return group_masks64_to_packed_bytes(group_masks, partition_num, max_g_value);
    }
    return group_masks64_to_groups(group_masks, partition_num, max_g_value);
}

static PyObject *
sample_native_direct_rows_random_bytes_u8(
    const uint32_t counts[64],
    int max_g_value,
    int partition_num,
    const unsigned char *random_bytes,
    Py_ssize_t random_len
)
{
    Py_ssize_t offset = 0;
    uint8_t group_masks[64];
    uint64_t *subset_rows[65];
    int value;

    memset(group_masks, 0, (size_t)partition_num * sizeof(group_masks[0]));
    memset(subset_rows, 0, (size_t)(partition_num + 1) * sizeof(subset_rows[0]));

    for (value = 0; value < max_g_value; ++value) {
        int value_count = (int)counts[value];
        uint64_t subset_count;
        Py_ssize_t byte_len;
        __uint128_t threshold;
        uint64_t rank;
        uint8_t value_bit = (uint8_t)(1u << value);

        if (value_count == 0) {
            continue;
        }
        if (value_count > partition_num) {
            PyErr_SetString(PyExc_OverflowError, "binomial coefficient does not fit in uint64");
            return NULL;
        }
        subset_count = g_comb_table[partition_num][value_count];
        byte_len = g_rank_byte_lengths[partition_num][value_count];
        threshold = g_rank_thresholds[partition_num][value_count];

        if (!read_random_subset_rank(
                random_bytes,
                random_len,
                &offset,
                partition_num,
                value_count,
                subset_count,
                byte_len,
                threshold,
                &rank)) {
            return NULL;
        }

        if (value_count == 1) {
            group_masks[rank] |= value_bit;
            continue;
        }

        if (subset_rows[value_count] == NULL) {
            subset_rows[value_count] = ensure_subset_mask_row(
                (unsigned int)partition_num,
                (unsigned int)value_count
            );
            if (subset_rows[value_count] == NULL) {
                if (PyErr_Occurred()) {
                    return NULL;
                }
                PyErr_SetString(PyExc_RuntimeError, "direct subset row is unavailable");
                return NULL;
            }
        }
        mark_position_mask_u8(group_masks, subset_rows[value_count][rank], 0, value_bit);
    }

    return PyBytes_FromStringAndSize((const char *)group_masks, partition_num);
}

static PyObject *
sample_native_tiny_partition_random_bytes(
    const uint32_t counts[64],
    int max_g_value,
    int partition_num,
    const unsigned char *random_bytes,
    Py_ssize_t random_len,
    int return_group_masks
)
{
    Py_ssize_t offset = 0;
    uint64_t group_masks[64];
    uint64_t *subset_rows[9];
    int value;

    memset(group_masks, 0, (size_t)partition_num * sizeof(group_masks[0]));
    memset(subset_rows, 0, (size_t)(partition_num + 1) * sizeof(subset_rows[0]));

    for (value = 0; value < max_g_value; ++value) {
        int value_count = (int)counts[value];
        uint64_t rank;
        uint64_t value_bit = ((uint64_t)1) << value;
        uint64_t position_mask;

        if (value_count == 0) {
            continue;
        }
        if (value_count > partition_num) {
            PyErr_SetString(PyExc_OverflowError, "binomial coefficient does not fit in uint64");
            return NULL;
        }

        while (1) {
            unsigned char mapped;
            if (offset >= random_len) {
                PyErr_SetString(PyExc_ValueError, "insufficient random bytes");
                return NULL;
            }
            mapped = g_rank_u8_rows[partition_num][value_count][random_bytes[offset]];
            offset += 1;
            if (mapped != RANK_U8_REJECT) {
                rank = (uint64_t)mapped;
                break;
            }
        }

        if (value_count == 1) {
            group_masks[rank] |= value_bit;
            continue;
        }

        if (subset_rows[value_count] == NULL) {
            subset_rows[value_count] = ensure_subset_mask_row(
                (unsigned int)partition_num,
                (unsigned int)value_count
            );
            if (subset_rows[value_count] == NULL) {
                if (PyErr_Occurred()) {
                    return NULL;
                }
                PyErr_SetString(PyExc_RuntimeError, "tiny partition subset row is unavailable");
                return NULL;
            }
        }
        position_mask = subset_rows[value_count][rank];
        while (position_mask != 0) {
            uint64_t low_bit = position_mask & (~position_mask + 1u);
            group_masks[tz_u64(low_bit)] |= value_bit;
            position_mask ^= low_bit;
        }
    }

    if (return_group_masks) {
        return group_masks64_to_packed_bytes(group_masks, partition_num, max_g_value);
    }
    return group_masks64_to_groups(group_masks, partition_num, max_g_value);
}

static PyObject *
sample_native_split_random_bytes_u8(
    const uint32_t counts[64],
    int max_g_value,
    int partition_num,
    const unsigned char *random_bytes,
    Py_ssize_t random_len
)
{
    Py_ssize_t offset = 0;
    uint8_t group_masks[64];
    int value;

    memset(group_masks, 0, (size_t)partition_num * sizeof(group_masks[0]));

    for (value = 0; value < max_g_value; ++value) {
        int count = (int)counts[value];
        uint64_t subset_count;
        Py_ssize_t byte_len;
        __uint128_t threshold;
        uint64_t rank;
        uint8_t value_bit = (uint8_t)(1u << value);

        if (count == 0) {
            continue;
        }
        subset_count = g_comb_table[partition_num][count];
        if (subset_count == 0) {
            PyErr_SetString(PyExc_OverflowError, "binomial coefficient does not fit in uint64");
            return NULL;
        }

        byte_len = g_rank_byte_lengths[partition_num][count];
        threshold = g_rank_thresholds[partition_num][count];
        if (!read_random_subset_rank(
                random_bytes,
                random_len,
                &offset,
                partition_num,
                count,
                subset_count,
                byte_len,
                threshold,
                &rank)) {
            return NULL;
        }

        if (count == 1) {
            group_masks[rank] |= value_bit;
            continue;
        }
        if (!mark_subset_split_u8_from_rank(group_masks, partition_num, count, rank, value_bit)) {
            PyErr_SetString(PyExc_RuntimeError, "failed to decode subset rank");
            return NULL;
        }
    }

    return PyBytes_FromStringAndSize((const char *)group_masks, partition_num);
}

static PyObject *
sample_native_from_counts(
    const uint32_t counts[64],
    int max_g_value,
    int partition_num,
    PyObject *seed_material,
    int use_shake128,
    int return_group_masks
)
{
    PyObject *hash_obj = NULL;
    PyObject *digest_method = NULL;
    PyObject *buffer_obj = NULL;
    const unsigned char *buffer = NULL;
    Py_ssize_t buffer_len = 0;
    Py_ssize_t offset = 0;
    uint64_t group_masks[64];
    int value;

    memset(group_masks, 0, (size_t)partition_num * sizeof(group_masks[0]));

    if (g_shake128_ctor == NULL || g_shake256_ctor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "native SHAKE constructors are not initialized");
        return NULL;
    }

    hash_obj = PyObject_CallFunctionObjArgs(
        use_shake128 ? g_shake128_ctor : g_shake256_ctor,
        seed_material,
        NULL);
    if (hash_obj == NULL) {
        return NULL;
    }

    digest_method = PyObject_GetAttrString(hash_obj, "digest");
    if (digest_method == NULL) {
        Py_DECREF(hash_obj);
        return NULL;
    }

    for (value = 0; value < max_g_value; ++value) {
        int count = (int)counts[value];
        uint64_t subset_count;
        Py_ssize_t byte_len;
        __uint128_t threshold;
        uint64_t rank;
        uint64_t value_bit = ((uint64_t)1) << value;

        if (count == 0) {
            continue;
        }
        subset_count = g_comb_table[partition_num][count];
        if (subset_count == 0) {
            Py_XDECREF(buffer_obj);
            Py_DECREF(digest_method);
            Py_DECREF(hash_obj);
            PyErr_SetString(PyExc_OverflowError, "binomial coefficient does not fit in uint64");
            return NULL;
        }

        byte_len = g_rank_byte_lengths[partition_num][count];
        threshold = g_rank_thresholds[partition_num][count];

        while (1) {
            Py_ssize_t end = offset + byte_len;
            if (buffer_len < end) {
                Py_ssize_t target;
                PyObject *new_buffer;

                if (buffer_len < 32) {
                    target = end < 32 ? 32 : end;
                } else {
                    Py_ssize_t doubled = 2 * buffer_len;
                    target = end < doubled ? doubled : end;
                }

                new_buffer = PyObject_CallFunction(digest_method, "n", target);
                if (new_buffer == NULL) {
                    Py_XDECREF(buffer_obj);
                    Py_DECREF(digest_method);
                    Py_DECREF(hash_obj);
                    return NULL;
                }
                if (!PyBytes_Check(new_buffer)) {
                    Py_DECREF(new_buffer);
                    Py_XDECREF(buffer_obj);
                    Py_DECREF(digest_method);
                    Py_DECREF(hash_obj);
                    PyErr_SetString(PyExc_TypeError, "digest() did not return bytes");
                    return NULL;
                }
                Py_XDECREF(buffer_obj);
                buffer_obj = new_buffer;
                buffer = (const unsigned char *)PyBytes_AS_STRING(buffer_obj);
                buffer_len = PyBytes_GET_SIZE(buffer_obj);
            }

            {
                uint64_t candidate = read_be_u64(buffer + offset, byte_len);
                offset = end;
                if ((__uint128_t)candidate < threshold) {
                    rank = candidate % subset_count;
                    break;
                }
            }
        }

        if (count == 1) {
            group_masks[rank] |= value_bit;
            continue;
        }
        if (!mark_subset64_from_rank(group_masks, partition_num, count, rank, value_bit)) {
            Py_XDECREF(buffer_obj);
            Py_DECREF(digest_method);
            Py_DECREF(hash_obj);
            PyErr_SetString(PyExc_RuntimeError, "failed to decode subset rank");
            return NULL;
        }
    }

    Py_XDECREF(buffer_obj);
    Py_DECREF(digest_method);
    Py_DECREF(hash_obj);
    if (return_group_masks) {
        return group_masks64_to_python_masks(group_masks, partition_num, max_g_value);
    }
    return group_masks64_to_groups(group_masks, partition_num, max_g_value);
}

static PyObject *
sample_native_from_counts_random_bytes(
    const uint32_t counts[64],
    int max_g_value,
    int partition_num,
    const unsigned char *random_bytes,
    Py_ssize_t random_len,
    int return_group_masks
)
{
    Py_ssize_t offset = 0;
    uint64_t group_masks[64];
    int value;

    if (partition_num <= 8) {
        return sample_native_tiny_partition_random_bytes(
            counts,
            max_g_value,
            partition_num,
            random_bytes,
            random_len,
            return_group_masks
        );
    }

    if (counts_fit_direct_subset_rows(counts, max_g_value, partition_num)) {
        if (return_group_masks && max_g_value <= 8) {
            return sample_native_direct_rows_random_bytes_u8(
                counts,
                max_g_value,
                partition_num,
                random_bytes,
                random_len
            );
        }
        return sample_native_direct_rows_random_bytes(
            counts,
            max_g_value,
            partition_num,
            random_bytes,
            random_len,
            return_group_masks
        );
    }

    if (return_group_masks && max_g_value <= 8 && partition_num > 8 && partition_num <= 32) {
        return sample_native_split_random_bytes_u8(
            counts,
            max_g_value,
            partition_num,
            random_bytes,
            random_len
        );
    }

    memset(group_masks, 0, (size_t)partition_num * sizeof(group_masks[0]));

    for (value = 0; value < max_g_value; ++value) {
        int count = (int)counts[value];
        uint64_t subset_count;
        Py_ssize_t byte_len;
        __uint128_t threshold;
        uint64_t rank;
        uint64_t value_bit = ((uint64_t)1) << value;

        if (count == 0) {
            continue;
        }
        subset_count = g_comb_table[partition_num][count];
        if (subset_count == 0) {
            PyErr_SetString(PyExc_OverflowError, "binomial coefficient does not fit in uint64");
            return NULL;
        }

        byte_len = g_rank_byte_lengths[partition_num][count];
        threshold = g_rank_thresholds[partition_num][count];

        if (byte_len <= 4) {
            if (!read_random_subset_rank(
                    random_bytes,
                    random_len,
                    &offset,
                    partition_num,
                    count,
                    subset_count,
                    byte_len,
                    threshold,
                    &rank)) {
                return NULL;
            }
        } else {
            while (1) {
                Py_ssize_t end = offset + byte_len;
                uint64_t candidate;
                if (end > random_len) {
                    PyErr_SetString(PyExc_ValueError, "insufficient random bytes");
                    return NULL;
                }
                candidate = read_be_u64(random_bytes + offset, byte_len);
                offset = end;
                if ((__uint128_t)candidate < threshold) {
                    rank = candidate % subset_count;
                    break;
                }
            }
        }

        if (count == 1) {
            group_masks[rank] |= value_bit;
            continue;
        }
        if (
            partition_num > 8 && partition_num <= 32
            ? !mark_subset_split64_from_rank(group_masks, partition_num, count, rank, value_bit)
            : !mark_subset64_from_rank(group_masks, partition_num, count, rank, value_bit)
        ) {
            PyErr_SetString(PyExc_RuntimeError, "failed to decode subset rank");
            return NULL;
        }
    }

    if (return_group_masks) {
        return group_masks64_to_packed_bytes(group_masks, partition_num, max_g_value);
    }
    return group_masks64_to_groups(group_masks, partition_num, max_g_value);
}

static PyObject *
sample_w4_from_counts(
    const uint32_t counts[4],
    int partition_num,
    PyObject *seed_material,
    int use_shake128,
    int return_group_masks
)
{
    PyObject *hash_obj = NULL;
    PyObject *digest_method = NULL;
    PyObject *buffer_obj = NULL;
    const unsigned char *buffer = NULL;
    Py_ssize_t buffer_len = 0;
    Py_ssize_t offset = 0;
    uint8_t group_masks[64];
    int value;

    memset(group_masks, 0, (size_t)partition_num * sizeof(group_masks[0]));

    if (g_shake128_ctor == NULL || g_shake256_ctor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "native SHAKE constructors are not initialized");
        return NULL;
    }

    hash_obj = PyObject_CallFunctionObjArgs(
        use_shake128 ? g_shake128_ctor : g_shake256_ctor,
        seed_material,
        NULL);
    if (hash_obj == NULL) {
        return NULL;
    }

    digest_method = PyObject_GetAttrString(hash_obj, "digest");
    if (digest_method == NULL) {
        Py_DECREF(hash_obj);
        return NULL;
    }

    for (value = 0; value < 4; ++value) {
        int count = (int)counts[value];
        uint64_t subset_count;
        Py_ssize_t byte_len;
        __uint128_t threshold;
        uint64_t rank;
        uint8_t value_bit = (uint8_t)(1u << value);

        if (count == 0) {
            continue;
        }
        subset_count = g_comb_table[partition_num][count];
        if (subset_count == 0) {
            Py_XDECREF(buffer_obj);
            Py_DECREF(digest_method);
            Py_DECREF(hash_obj);
            PyErr_SetString(PyExc_OverflowError, "binomial coefficient does not fit in uint64");
            return NULL;
        }

        byte_len = g_rank_byte_lengths[partition_num][count];
        threshold = g_rank_thresholds[partition_num][count];

        while (1) {
            Py_ssize_t end = offset + byte_len;
            if (buffer_len < end) {
                Py_ssize_t target;
                PyObject *new_buffer;

                if (buffer_len < 32) {
                    target = end < 32 ? 32 : end;
                } else {
                    Py_ssize_t doubled = 2 * buffer_len;
                    target = end < doubled ? doubled : end;
                }

                new_buffer = PyObject_CallFunction(digest_method, "n", target);
                if (new_buffer == NULL) {
                    Py_XDECREF(buffer_obj);
                    Py_DECREF(digest_method);
                    Py_DECREF(hash_obj);
                    return NULL;
                }
                if (!PyBytes_Check(new_buffer)) {
                    Py_DECREF(new_buffer);
                    Py_XDECREF(buffer_obj);
                    Py_DECREF(digest_method);
                    Py_DECREF(hash_obj);
                    PyErr_SetString(PyExc_TypeError, "digest() did not return bytes");
                    return NULL;
                }
                Py_XDECREF(buffer_obj);
                buffer_obj = new_buffer;
                buffer = (const unsigned char *)PyBytes_AS_STRING(buffer_obj);
                buffer_len = PyBytes_GET_SIZE(buffer_obj);
            }

            {
                uint64_t candidate = read_be_u64(buffer + offset, byte_len);
                offset = end;
                if ((__uint128_t)candidate < threshold) {
                    rank = candidate % subset_count;
                    break;
                }
            }
        }

        if (count == 1) {
            group_masks[rank] |= value_bit;
            continue;
        }
        if (
            partition_num <= 32
            && subset_count <= 0xFFFFFFFFu
            ? !mark_subset_from_rank_u32(group_masks, partition_num, count, (uint32_t)rank, value_bit)
            : !mark_subset_from_rank(group_masks, partition_num, count, rank, value_bit)
        ) {
            Py_XDECREF(buffer_obj);
            Py_DECREF(digest_method);
            Py_DECREF(hash_obj);
            PyErr_SetString(PyExc_RuntimeError, "failed to decode subset rank");
            return NULL;
        }
    }

    Py_XDECREF(buffer_obj);
    Py_DECREF(digest_method);
    Py_DECREF(hash_obj);
    if (return_group_masks) {
        return PyBytes_FromStringAndSize((const char *)group_masks, partition_num);
    }
    return group_masks_to_groups(group_masks, partition_num);
}

static PyObject *
sample_w4_split_random_bytes(
    const uint32_t counts[4],
    int partition_num,
    const unsigned char *random_bytes,
    Py_ssize_t random_len,
    int return_group_masks
)
{
    Py_ssize_t offset = 0;
    uint64_t value_masks[4] = {0, 0, 0, 0};
    int value;
    int position;

    for (value = 0; value < 4; ++value) {
        int count = (int)counts[value];
        uint64_t subset_count;
        Py_ssize_t byte_len;
        __uint128_t threshold;
        uint64_t rank;

        if (count == 0) {
            continue;
        }
        if (count > partition_num) {
            PyErr_SetString(PyExc_OverflowError, "binomial coefficient does not fit in uint64");
            return NULL;
        }
        subset_count = g_comb_table[partition_num][count];
        if (subset_count == 0) {
            PyErr_SetString(PyExc_OverflowError, "binomial coefficient does not fit in uint64");
            return NULL;
        }

        byte_len = g_rank_byte_lengths[partition_num][count];
        threshold = g_rank_thresholds[partition_num][count];
        if (!read_random_subset_rank(
                random_bytes,
                random_len,
                &offset,
                partition_num,
                count,
                subset_count,
                byte_len,
                threshold,
                &rank)) {
            return NULL;
        }

        if (count == 1) {
            value_masks[value] = ((uint64_t)1) << rank;
            continue;
        }
        if (!subset_split_mask_from_rank(partition_num, count, rank, &value_masks[value])) {
            return NULL;
        }
    }

    if (return_group_masks) {
        PyObject *result = PyBytes_FromStringAndSize(NULL, partition_num);
        unsigned char *buffer;
        if (result == NULL) {
            return NULL;
        }
        buffer = (unsigned char *)PyBytes_AS_STRING(result);
        for (position = 0; position < partition_num; ++position) {
            uint64_t position_bit = ((uint64_t)1) << position;
            buffer[position] = (unsigned char)(
                ((value_masks[0] & position_bit) ? 1u : 0u)
                | ((value_masks[1] & position_bit) ? 2u : 0u)
                | ((value_masks[2] & position_bit) ? 4u : 0u)
                | ((value_masks[3] & position_bit) ? 8u : 0u)
            );
        }
        return result;
    }

    {
        uint8_t group_masks[64];
        memset(group_masks, 0, (size_t)partition_num * sizeof(group_masks[0]));
        for (value = 0; value < 4; ++value) {
            mark_position_mask_u8(group_masks, value_masks[value], 0, (uint8_t)(1u << value));
        }
        return group_masks_to_groups(group_masks, partition_num);
    }
}

static PyObject *
sample_w4_from_counts_random_bytes(
    const uint32_t counts[4],
    int partition_num,
    const unsigned char *random_bytes,
    Py_ssize_t random_len,
    int return_group_masks
)
{
    Py_ssize_t offset = 0;
    uint8_t group_masks[64];
    int value;

    if (partition_num > 8 && partition_num <= 40) {
        return sample_w4_split_random_bytes(
            counts,
            partition_num,
            random_bytes,
            random_len,
            return_group_masks
        );
    }

    memset(group_masks, 0, (size_t)partition_num * sizeof(group_masks[0]));

    for (value = 0; value < 4; ++value) {
        int count = (int)counts[value];
        uint64_t subset_count;
        Py_ssize_t byte_len;
        __uint128_t threshold;
        uint64_t rank;
        uint8_t value_bit = (uint8_t)(1u << value);

        if (count == 0) {
            continue;
        }
        subset_count = g_comb_table[partition_num][count];
        if (subset_count == 0) {
            PyErr_SetString(PyExc_OverflowError, "binomial coefficient does not fit in uint64");
            return NULL;
        }

        byte_len = g_rank_byte_lengths[partition_num][count];
        threshold = g_rank_thresholds[partition_num][count];

        if (byte_len <= 4) {
            if (!read_random_subset_rank(
                    random_bytes,
                    random_len,
                    &offset,
                    partition_num,
                    count,
                    subset_count,
                    byte_len,
                    threshold,
                    &rank)) {
                return NULL;
            }
        } else {
            while (1) {
                Py_ssize_t end = offset + byte_len;
                uint64_t candidate;
                if (end > random_len) {
                    PyErr_SetString(PyExc_ValueError, "insufficient random bytes");
                    return NULL;
                }
                candidate = read_be_u64(random_bytes + offset, byte_len);
                offset = end;
                if ((__uint128_t)candidate < threshold) {
                    rank = candidate % subset_count;
                    break;
                }
            }
        }

        if (count == 1) {
            group_masks[rank] |= value_bit;
            continue;
        }
        if (
            partition_num > 8 && partition_num <= 32
            ? !mark_subset_split_u8_from_rank(group_masks, partition_num, count, rank, value_bit)
            : (
                partition_num <= 32 && subset_count <= 0xFFFFFFFFu
                ? !mark_subset_from_rank_u32(group_masks, partition_num, count, (uint32_t)rank, value_bit)
                : !mark_subset_from_rank(group_masks, partition_num, count, rank, value_bit)
            )
        ) {
            PyErr_SetString(PyExc_RuntimeError, "failed to decode subset rank");
            return NULL;
        }
    }

    if (return_group_masks) {
        return PyBytes_FromStringAndSize((const char *)group_masks, partition_num);
    }
    return group_masks_to_groups(group_masks, partition_num);
}

static PyObject *
val_strict_isp_w4_bytes(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "partition_value",
        "hash_len",
        "partition_num",
        "window_low",
        "window_high",
        "seed_material",
        "use_shake128",
        "return_group_masks",
        NULL,
    };
    const unsigned char *partition_value = NULL;
    Py_ssize_t partition_len = 0;
    Py_ssize_t hash_len = 0;
    int partition_num = 0;
    int window_low = 0;
    int window_high = 0;
    PyObject *seed_material = NULL;
    int use_shake128 = 0;
    int return_group_masks = 0;
    Py_ssize_t expected_len;
    uint32_t counts[4];
    int accepted = 0;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y#niiiO!p|p:val_strict_isp_w4_bytes",
            kwlist,
            &partition_value,
            &partition_len,
            &hash_len,
            &partition_num,
            &window_low,
            &window_high,
            &PyBytes_Type,
            &seed_material,
            &use_shake128,
            &return_group_masks)) {
        return NULL;
    }

    if (hash_len <= 0 || (hash_len % 2) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by 2");
        return NULL;
    }
    if (partition_num <= 0 || partition_num > 64) {
        PyErr_SetString(PyExc_ValueError, "partition_num must lie in [1, 64]");
        return NULL;
    }

    expected_len = (hash_len + 7) / 8;
    if (partition_len != expected_len) {
        PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
        return NULL;
    }

    if (!profile_counts_w4_checked(
            partition_value,
            partition_len,
            hash_len,
            counts,
            window_low,
            window_high,
            &accepted)) {
        return NULL;
    }
    if (!accepted) {
        Py_RETURN_NONE;
    }

    return sample_w4_from_counts(counts, partition_num, seed_material, use_shake128, return_group_masks);
}

static PyObject *
val_strict_isp_w4_bytes_prefixed_seed(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "partition_value",
        "hash_len",
        "partition_num",
        "window_low",
        "window_high",
        "seed_prefix",
        "use_shake128",
        "return_group_masks",
        NULL,
    };
    const unsigned char *partition_value = NULL;
    Py_ssize_t partition_len = 0;
    Py_ssize_t hash_len = 0;
    int partition_num = 0;
    int window_low = 0;
    int window_high = 0;
    PyObject *seed_prefix = NULL;
    int use_shake128 = 0;
    int return_group_masks = 0;
    Py_ssize_t expected_len;
    Py_ssize_t seed_prefix_len;
    PyObject *seed_material = NULL;
    PyObject *result = NULL;
    uint32_t counts[4];
    int accepted = 0;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y#niiiO!p|p:val_strict_isp_w4_bytes_prefixed_seed",
            kwlist,
            &partition_value,
            &partition_len,
            &hash_len,
            &partition_num,
            &window_low,
            &window_high,
            &PyBytes_Type,
            &seed_prefix,
            &use_shake128,
            &return_group_masks)) {
        return NULL;
    }

    if (hash_len <= 0 || (hash_len % 2) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by 2");
        return NULL;
    }
    if (partition_num <= 0 || partition_num > 64) {
        PyErr_SetString(PyExc_ValueError, "partition_num must lie in [1, 64]");
        return NULL;
    }

    expected_len = (hash_len + 7) / 8;
    if (partition_len != expected_len) {
        PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
        return NULL;
    }

    if (!profile_counts_w4_checked(
            partition_value,
            partition_len,
            hash_len,
            counts,
            window_low,
            window_high,
            &accepted)) {
        return NULL;
    }
    if (!accepted) {
        Py_RETURN_NONE;
    }

    seed_prefix_len = PyBytes_GET_SIZE(seed_prefix);
    if (seed_prefix_len > PY_SSIZE_T_MAX - partition_len) {
        PyErr_SetString(PyExc_OverflowError, "seed material is too large");
        return NULL;
    }
    seed_material = PyBytes_FromStringAndSize(NULL, seed_prefix_len + partition_len);
    if (seed_material == NULL) {
        return NULL;
    }
    memcpy(PyBytes_AS_STRING(seed_material), PyBytes_AS_STRING(seed_prefix), (size_t)seed_prefix_len);
    memcpy(PyBytes_AS_STRING(seed_material) + seed_prefix_len, partition_value, (size_t)partition_len);

    result = sample_w4_from_counts(counts, partition_num, seed_material, use_shake128, return_group_masks);
    Py_DECREF(seed_material);
    return result;
}

static PyObject *
val_strict_isp_w4_bytes_default_seed(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "partition_value",
        "hash_len",
        "partition_num",
        "window_low",
        "window_high",
        "use_shake128",
        "return_group_masks",
        NULL,
    };
    const unsigned char *partition_value = NULL;
    Py_ssize_t partition_len = 0;
    Py_ssize_t hash_len = 0;
    int partition_num = 0;
    int window_low = 0;
    int window_high = 0;
    int use_shake128 = 0;
    int return_group_masks = 0;
    Py_ssize_t expected_len;
    PyObject *seed_material = NULL;
    PyObject *result = NULL;
    uint32_t counts[4];
    int accepted = 0;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y#niiip|p:val_strict_isp_w4_bytes_default_seed",
            kwlist,
            &partition_value,
            &partition_len,
            &hash_len,
            &partition_num,
            &window_low,
            &window_high,
            &use_shake128,
            &return_group_masks)) {
        return NULL;
    }

    if (hash_len <= 0 || (hash_len % 2) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by 2");
        return NULL;
    }
    if (partition_num <= 0 || partition_num > 64) {
        PyErr_SetString(PyExc_ValueError, "partition_num must lie in [1, 64]");
        return NULL;
    }

    expected_len = (hash_len + 7) / 8;
    if (partition_len != expected_len) {
        PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
        return NULL;
    }

    if (!profile_counts_w4_checked(
            partition_value,
            partition_len,
            hash_len,
            counts,
            window_low,
            window_high,
            &accepted)) {
        return NULL;
    }
    if (!accepted) {
        Py_RETURN_NONE;
    }

    seed_material = build_default_seed_material_for_bytes(
        partition_value,
        partition_len,
        hash_len,
        use_shake128
    );
    if (seed_material == NULL) {
        return NULL;
    }
    result = sample_w4_from_counts(counts, partition_num, seed_material, use_shake128, return_group_masks);
    Py_DECREF(seed_material);
    return result;
}

static PyObject *
val_strict_isp_w4_bytes_random(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "partition_value",
        "hash_len",
        "partition_num",
        "window_low",
        "window_high",
        "random_bytes",
        "return_group_masks",
        NULL,
    };
    const unsigned char *partition_value = NULL;
    Py_ssize_t partition_len = 0;
    Py_ssize_t hash_len = 0;
    int partition_num = 0;
    int window_low = 0;
    int window_high = 0;
    const unsigned char *random_bytes = NULL;
    Py_ssize_t random_len = 0;
    int return_group_masks = 0;
    Py_ssize_t expected_len;
    uint32_t counts[4];
    int accepted = 0;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y#niiiy#|p:val_strict_isp_w4_bytes_random",
            kwlist,
            &partition_value,
            &partition_len,
            &hash_len,
            &partition_num,
            &window_low,
            &window_high,
            &random_bytes,
            &random_len,
            &return_group_masks)) {
        return NULL;
    }

    if (hash_len <= 0 || (hash_len % 2) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by 2");
        return NULL;
    }
    if (partition_num <= 0 || partition_num > 64) {
        PyErr_SetString(PyExc_ValueError, "partition_num must lie in [1, 64]");
        return NULL;
    }

    expected_len = (hash_len + 7) / 8;
    if (partition_len != expected_len) {
        PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
        return NULL;
    }

    if (!profile_counts_w4_checked(
            partition_value,
            partition_len,
            hash_len,
            counts,
            window_low,
            window_high,
            &accepted)) {
        return NULL;
    }
    if (!accepted) {
        Py_RETURN_NONE;
    }

    return sample_w4_from_counts_random_bytes(
        counts,
        partition_num,
        random_bytes,
        random_len,
        return_group_masks
    );
}

static PyObject *
val_strict_isp_w4_bytes_random_fast(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    const unsigned char *partition_value = NULL;
    Py_ssize_t partition_len = 0;
    Py_ssize_t hash_len = 0;
    int partition_num = 0;
    int window_low = 0;
    int window_high = 0;
    const unsigned char *random_bytes = NULL;
    Py_ssize_t random_len = 0;
    int return_group_masks = 0;
    Py_ssize_t expected_len;
    uint32_t counts[4];
    int accepted = 0;

    (void)self;

    if (nargs != 7) {
        PyErr_SetString(PyExc_TypeError, "val_strict_isp_w4_bytes_random_fast expects 7 positional arguments");
        return NULL;
    }
    if (!PyBytes_Check(args[0]) || !PyBytes_Check(args[5])) {
        PyErr_SetString(PyExc_TypeError, "partition_value and random_bytes must be bytes");
        return NULL;
    }

    partition_value = (const unsigned char *)PyBytes_AS_STRING(args[0]);
    partition_len = PyBytes_GET_SIZE(args[0]);
    hash_len = PyLong_AsSsize_t(args[1]);
    if (hash_len == -1 && PyErr_Occurred()) {
        return NULL;
    }
    {
        long value_long = PyLong_AsLong(args[2]);
        if (value_long == -1 && PyErr_Occurred()) {
            return NULL;
        }
        partition_num = (int)value_long;
        value_long = PyLong_AsLong(args[3]);
        if (value_long == -1 && PyErr_Occurred()) {
            return NULL;
        }
        window_low = (int)value_long;
        value_long = PyLong_AsLong(args[4]);
        if (value_long == -1 && PyErr_Occurred()) {
            return NULL;
        }
        window_high = (int)value_long;
    }
    random_bytes = (const unsigned char *)PyBytes_AS_STRING(args[5]);
    random_len = PyBytes_GET_SIZE(args[5]);
    return_group_masks = PyObject_IsTrue(args[6]);
    if (return_group_masks < 0) {
        return NULL;
    }

    if (hash_len <= 0 || (hash_len % 2) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by 2");
        return NULL;
    }
    if (partition_num <= 0 || partition_num > 64) {
        PyErr_SetString(PyExc_ValueError, "partition_num must lie in [1, 64]");
        return NULL;
    }

    expected_len = (hash_len + 7) / 8;
    if (partition_len != expected_len) {
        PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
        return NULL;
    }

    if (!profile_counts_w4_checked(
            partition_value,
            partition_len,
            hash_len,
            counts,
            window_low,
            window_high,
            &accepted)) {
        return NULL;
    }
    if (!accepted) {
        Py_RETURN_NONE;
    }

    return sample_w4_from_counts_random_bytes(
        counts,
        partition_num,
        random_bytes,
        random_len,
        return_group_masks
    );
}

static PyObject *
val_strict_isp_bytes(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "partition_value",
        "hash_len",
        "max_g_bit",
        "partition_num",
        "window_low",
        "window_high",
        "seed_material",
        "use_shake128",
        "return_group_masks",
        NULL,
    };
    const unsigned char *partition_value = NULL;
    Py_ssize_t partition_len = 0;
    Py_ssize_t hash_len = 0;
    int max_g_bit = 0;
    int max_g_value = 0;
    int partition_num = 0;
    int window_low = 0;
    int window_high = 0;
    PyObject *seed_material = NULL;
    int use_shake128 = 0;
    int return_group_masks = 0;
    Py_ssize_t expected_len;
    uint32_t counts[64];
    int accepted = 0;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y#niiiiO!p|p:val_strict_isp_bytes",
            kwlist,
            &partition_value,
            &partition_len,
            &hash_len,
            &max_g_bit,
            &partition_num,
            &window_low,
            &window_high,
            &PyBytes_Type,
            &seed_material,
            &use_shake128,
            &return_group_masks)) {
        return NULL;
    }

    if (max_g_bit < 2 || max_g_bit > 6) {
        PyErr_SetString(PyExc_ValueError, "max_g_bit must lie in [2, 6] for native ValStrictISP");
        return NULL;
    }
    if (hash_len <= 0 || (hash_len % max_g_bit) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by max_g_bit");
        return NULL;
    }
    if (partition_num <= 0 || partition_num > 64) {
        PyErr_SetString(PyExc_ValueError, "partition_num must lie in [1, 64]");
        return NULL;
    }

    expected_len = (hash_len + 7) / 8;
    if (partition_len != expected_len) {
        PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
        return NULL;
    }

    max_g_value = 1 << max_g_bit;
    if (!profile_counts_native_checked(
            partition_value,
            partition_len,
            hash_len,
            max_g_bit,
            max_g_value,
            counts,
            window_low,
            window_high,
            &accepted)) {
        return NULL;
    }
    if (!accepted) {
        Py_RETURN_NONE;
    }

    return sample_native_from_counts(
        counts,
        max_g_value,
        partition_num,
        seed_material,
        use_shake128,
        return_group_masks
    );
}

static PyObject *
val_strict_isp_bytes_prefixed_seed(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "partition_value",
        "hash_len",
        "max_g_bit",
        "partition_num",
        "window_low",
        "window_high",
        "seed_prefix",
        "use_shake128",
        "return_group_masks",
        NULL,
    };
    const unsigned char *partition_value = NULL;
    Py_ssize_t partition_len = 0;
    Py_ssize_t hash_len = 0;
    int max_g_bit = 0;
    int max_g_value = 0;
    int partition_num = 0;
    int window_low = 0;
    int window_high = 0;
    PyObject *seed_prefix = NULL;
    int use_shake128 = 0;
    int return_group_masks = 0;
    Py_ssize_t expected_len;
    Py_ssize_t seed_prefix_len;
    PyObject *seed_material = NULL;
    PyObject *result = NULL;
    uint32_t counts[64];
    int accepted = 0;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y#niiiiO!p|p:val_strict_isp_bytes_prefixed_seed",
            kwlist,
            &partition_value,
            &partition_len,
            &hash_len,
            &max_g_bit,
            &partition_num,
            &window_low,
            &window_high,
            &PyBytes_Type,
            &seed_prefix,
            &use_shake128,
            &return_group_masks)) {
        return NULL;
    }

    if (max_g_bit < 2 || max_g_bit > 6) {
        PyErr_SetString(PyExc_ValueError, "max_g_bit must lie in [2, 6] for native ValStrictISP");
        return NULL;
    }
    if (hash_len <= 0 || (hash_len % max_g_bit) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by max_g_bit");
        return NULL;
    }
    if (partition_num <= 0 || partition_num > 64) {
        PyErr_SetString(PyExc_ValueError, "partition_num must lie in [1, 64]");
        return NULL;
    }

    expected_len = (hash_len + 7) / 8;
    if (partition_len != expected_len) {
        PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
        return NULL;
    }

    max_g_value = 1 << max_g_bit;
    if (!profile_counts_native_checked(
            partition_value,
            partition_len,
            hash_len,
            max_g_bit,
            max_g_value,
            counts,
            window_low,
            window_high,
            &accepted)) {
        return NULL;
    }
    if (!accepted) {
        Py_RETURN_NONE;
    }

    seed_prefix_len = PyBytes_GET_SIZE(seed_prefix);
    if (seed_prefix_len > PY_SSIZE_T_MAX - partition_len) {
        PyErr_SetString(PyExc_OverflowError, "seed material is too large");
        return NULL;
    }
    seed_material = PyBytes_FromStringAndSize(NULL, seed_prefix_len + partition_len);
    if (seed_material == NULL) {
        return NULL;
    }
    memcpy(PyBytes_AS_STRING(seed_material), PyBytes_AS_STRING(seed_prefix), (size_t)seed_prefix_len);
    memcpy(PyBytes_AS_STRING(seed_material) + seed_prefix_len, partition_value, (size_t)partition_len);

    result = sample_native_from_counts(
        counts,
        max_g_value,
        partition_num,
        seed_material,
        use_shake128,
        return_group_masks
    );
    Py_DECREF(seed_material);
    return result;
}

static PyObject *
val_strict_isp_bytes_default_seed(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "partition_value",
        "hash_len",
        "max_g_bit",
        "partition_num",
        "window_low",
        "window_high",
        "use_shake128",
        "return_group_masks",
        NULL,
    };
    const unsigned char *partition_value = NULL;
    Py_ssize_t partition_len = 0;
    Py_ssize_t hash_len = 0;
    int max_g_bit = 0;
    int max_g_value = 0;
    int partition_num = 0;
    int window_low = 0;
    int window_high = 0;
    int use_shake128 = 0;
    int return_group_masks = 0;
    Py_ssize_t expected_len;
    PyObject *seed_material = NULL;
    PyObject *result = NULL;
    uint32_t counts[64];
    int accepted = 0;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y#niiiip|p:val_strict_isp_bytes_default_seed",
            kwlist,
            &partition_value,
            &partition_len,
            &hash_len,
            &max_g_bit,
            &partition_num,
            &window_low,
            &window_high,
            &use_shake128,
            &return_group_masks)) {
        return NULL;
    }

    if (max_g_bit < 2 || max_g_bit > 6) {
        PyErr_SetString(PyExc_ValueError, "max_g_bit must lie in [2, 6] for native ValStrictISP");
        return NULL;
    }
    if (hash_len <= 0 || (hash_len % max_g_bit) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by max_g_bit");
        return NULL;
    }
    if (partition_num <= 0 || partition_num > 64) {
        PyErr_SetString(PyExc_ValueError, "partition_num must lie in [1, 64]");
        return NULL;
    }

    expected_len = (hash_len + 7) / 8;
    if (partition_len != expected_len) {
        PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
        return NULL;
    }

    max_g_value = 1 << max_g_bit;
    if (!profile_counts_native_checked(
            partition_value,
            partition_len,
            hash_len,
            max_g_bit,
            max_g_value,
            counts,
            window_low,
            window_high,
            &accepted)) {
        return NULL;
    }
    if (!accepted) {
        Py_RETURN_NONE;
    }

    seed_material = build_default_seed_material_for_bytes(
        partition_value,
        partition_len,
        hash_len,
        use_shake128
    );
    if (seed_material == NULL) {
        return NULL;
    }
    result = sample_native_from_counts(
        counts,
        max_g_value,
        partition_num,
        seed_material,
        use_shake128,
        return_group_masks
    );
    Py_DECREF(seed_material);
    return result;
}

static PyObject *
val_strict_isp_bytes_random(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "partition_value",
        "hash_len",
        "max_g_bit",
        "partition_num",
        "window_low",
        "window_high",
        "random_bytes",
        "return_group_masks",
        NULL,
    };
    const unsigned char *partition_value = NULL;
    Py_ssize_t partition_len = 0;
    Py_ssize_t hash_len = 0;
    int max_g_bit = 0;
    int max_g_value = 0;
    int partition_num = 0;
    int window_low = 0;
    int window_high = 0;
    const unsigned char *random_bytes = NULL;
    Py_ssize_t random_len = 0;
    int return_group_masks = 0;
    Py_ssize_t expected_len;
    uint32_t counts[64];
    int accepted = 0;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y#niiiiy#|p:val_strict_isp_bytes_random",
            kwlist,
            &partition_value,
            &partition_len,
            &hash_len,
            &max_g_bit,
            &partition_num,
            &window_low,
            &window_high,
            &random_bytes,
            &random_len,
            &return_group_masks)) {
        return NULL;
    }

    if (max_g_bit < 2 || max_g_bit > 6) {
        PyErr_SetString(PyExc_ValueError, "max_g_bit must lie in [2, 6] for native ValStrictISP");
        return NULL;
    }
    if (hash_len <= 0 || (hash_len % max_g_bit) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by max_g_bit");
        return NULL;
    }
    if (partition_num <= 0 || partition_num > 64) {
        PyErr_SetString(PyExc_ValueError, "partition_num must lie in [1, 64]");
        return NULL;
    }

    expected_len = (hash_len + 7) / 8;
    if (partition_len != expected_len) {
        PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
        return NULL;
    }

    max_g_value = 1 << max_g_bit;
    if (!profile_counts_native_checked(
            partition_value,
            partition_len,
            hash_len,
            max_g_bit,
            max_g_value,
            counts,
            window_low,
            window_high,
            &accepted)) {
        return NULL;
    }
    if (!accepted) {
        Py_RETURN_NONE;
    }

    return sample_native_from_counts_random_bytes(
        counts,
        max_g_value,
        partition_num,
        random_bytes,
        random_len,
        return_group_masks
    );
}

static PyObject *
val_strict_isp_bytes_random_fast(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    const unsigned char *partition_value = NULL;
    Py_ssize_t partition_len = 0;
    Py_ssize_t hash_len = 0;
    int max_g_bit = 0;
    int max_g_value = 0;
    int partition_num = 0;
    int window_low = 0;
    int window_high = 0;
    const unsigned char *random_bytes = NULL;
    Py_ssize_t random_len = 0;
    int return_group_masks = 0;
    Py_ssize_t expected_len;
    uint32_t counts[64];
    int accepted = 0;

    (void)self;

    if (nargs != 8) {
        PyErr_SetString(PyExc_TypeError, "val_strict_isp_bytes_random_fast expects 8 positional arguments");
        return NULL;
    }
    if (!PyBytes_Check(args[0]) || !PyBytes_Check(args[6])) {
        PyErr_SetString(PyExc_TypeError, "partition_value and random_bytes must be bytes");
        return NULL;
    }

    partition_value = (const unsigned char *)PyBytes_AS_STRING(args[0]);
    partition_len = PyBytes_GET_SIZE(args[0]);
    hash_len = PyLong_AsSsize_t(args[1]);
    if (hash_len == -1 && PyErr_Occurred()) {
        return NULL;
    }
    {
        long value_long = PyLong_AsLong(args[2]);
        if (value_long == -1 && PyErr_Occurred()) {
            return NULL;
        }
        max_g_bit = (int)value_long;
        value_long = PyLong_AsLong(args[3]);
        if (value_long == -1 && PyErr_Occurred()) {
            return NULL;
        }
        partition_num = (int)value_long;
        value_long = PyLong_AsLong(args[4]);
        if (value_long == -1 && PyErr_Occurred()) {
            return NULL;
        }
        window_low = (int)value_long;
        value_long = PyLong_AsLong(args[5]);
        if (value_long == -1 && PyErr_Occurred()) {
            return NULL;
        }
        window_high = (int)value_long;
    }
    random_bytes = (const unsigned char *)PyBytes_AS_STRING(args[6]);
    random_len = PyBytes_GET_SIZE(args[6]);
    return_group_masks = PyObject_IsTrue(args[7]);
    if (return_group_masks < 0) {
        return NULL;
    }

    if (max_g_bit < 2 || max_g_bit > 6) {
        PyErr_SetString(PyExc_ValueError, "max_g_bit must lie in [2, 6] for native ValStrictISP");
        return NULL;
    }
    if (hash_len <= 0 || (hash_len % max_g_bit) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by max_g_bit");
        return NULL;
    }
    if (partition_num <= 0 || partition_num > 64) {
        PyErr_SetString(PyExc_ValueError, "partition_num must lie in [1, 64]");
        return NULL;
    }

    expected_len = (hash_len + 7) / 8;
    if (partition_len != expected_len) {
        PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
        return NULL;
    }

    max_g_value = 1 << max_g_bit;
    if (!profile_counts_native_checked(
            partition_value,
            partition_len,
            hash_len,
            max_g_bit,
            max_g_value,
            counts,
            window_low,
            window_high,
            &accepted)) {
        return NULL;
    }
    if (!accepted) {
        Py_RETURN_NONE;
    }

    return sample_native_from_counts_random_bytes(
        counts,
        max_g_value,
        partition_num,
        random_bytes,
        random_len,
        return_group_masks
    );
}

static PyObject *
val_strict_isp_find_first_random_stream(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "stream_bytes",
        "sampler_len",
        "hash_len",
        "max_g_bit",
        "partition_num",
        "window_low",
        "window_high",
        "start_salt",
        "candidate_count",
        "return_group_masks",
        NULL,
    };
    const unsigned char *stream_bytes = NULL;
    Py_ssize_t stream_len = 0;
    Py_ssize_t sampler_len = 0;
    Py_ssize_t hash_len = 0;
    int max_g_bit = 0;
    int max_g_value = 0;
    int partition_num = 0;
    int window_low = 0;
    int window_high = 0;
    Py_ssize_t start_salt = 0;
    Py_ssize_t candidate_count = 0;
    int return_group_masks = 0;
    Py_ssize_t candidate_len;
    Py_ssize_t salt;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y#nniiiinn|p:val_strict_isp_find_first_random_stream",
            kwlist,
            &stream_bytes,
            &stream_len,
            &sampler_len,
            &hash_len,
            &max_g_bit,
            &partition_num,
            &window_low,
            &window_high,
            &start_salt,
            &candidate_count,
            &return_group_masks)) {
        return NULL;
    }

    if (sampler_len < 0 || start_salt < 0 || candidate_count < 0) {
        PyErr_SetString(PyExc_ValueError, "sampler_len, start_salt, and candidate_count must be non-negative");
        return NULL;
    }
    if (max_g_bit < 2 || max_g_bit > 6) {
        PyErr_SetString(PyExc_ValueError, "max_g_bit must lie in [2, 6] for native ValStrictISP");
        return NULL;
    }
    if (hash_len <= 0 || (hash_len % max_g_bit) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by max_g_bit");
        return NULL;
    }
    if (partition_num <= 0 || partition_num > 64) {
        PyErr_SetString(PyExc_ValueError, "partition_num must lie in [1, 64]");
        return NULL;
    }
    if (sampler_len > stream_len) {
        PyErr_SetString(PyExc_ValueError, "sampler_len exceeds stream byte length");
        return NULL;
    }

    candidate_len = (hash_len + 7) / 8;
    if (
        candidate_count > 0
        && (
            start_salt > (PY_SSIZE_T_MAX - sampler_len) / candidate_len
            || candidate_count > (PY_SSIZE_T_MAX - sampler_len - start_salt * candidate_len) / candidate_len
        )
    ) {
        PyErr_SetString(PyExc_OverflowError, "candidate stream range is too large");
        return NULL;
    }
    if (sampler_len + (start_salt + candidate_count) * candidate_len > stream_len) {
        PyErr_SetString(PyExc_ValueError, "stream bytes do not cover the requested candidate range");
        return NULL;
    }

    max_g_value = 1 << max_g_bit;
    for (salt = start_salt; salt < start_salt + candidate_count; ++salt) {
        const unsigned char *partition_value = stream_bytes + sampler_len + salt * candidate_len;
        uint32_t counts[64];
        int accepted = 0;
        PyObject *groups;
        PyObject *result;
        PyObject *py_salt;

        if (!profile_counts_native_checked(
                partition_value,
                candidate_len,
                hash_len,
                max_g_bit,
                max_g_value,
                counts,
                window_low,
                window_high,
                &accepted)) {
            return NULL;
        }
        if (!accepted) {
            continue;
        }

        groups = sample_native_from_counts_random_bytes(
            counts,
            max_g_value,
            partition_num,
            stream_bytes,
            sampler_len,
            return_group_masks
        );
        if (groups == NULL) {
            return NULL;
        }
        py_salt = PyLong_FromSsize_t(salt);
        if (py_salt == NULL) {
            Py_DECREF(groups);
            return NULL;
        }
        result = PyTuple_New(2);
        if (result == NULL) {
            Py_DECREF(py_salt);
            Py_DECREF(groups);
            return NULL;
        }
        PyTuple_SET_ITEM(result, 0, py_salt);
        PyTuple_SET_ITEM(result, 1, groups);
        return result;
    }

    Py_RETURN_NONE;
}

static PyObject *
val_strict_isp_profile_counts(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "partition_value",
        "hash_len",
        "max_g_bit",
        NULL,
    };
    const unsigned char *partition_value = NULL;
    Py_ssize_t partition_len = 0;
    Py_ssize_t hash_len = 0;
    int max_g_bit = 0;
    int max_g_value = 0;
    Py_ssize_t expected_len;
    uint32_t counts[64];
    PyObject *result = NULL;
    int value;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y#ni:val_strict_isp_profile_counts",
            kwlist,
            &partition_value,
            &partition_len,
            &hash_len,
            &max_g_bit)) {
        return NULL;
    }

    if (max_g_bit < 2 || max_g_bit > 6) {
        PyErr_SetString(PyExc_ValueError, "max_g_bit must lie in [2, 6] for native ValStrictISP");
        return NULL;
    }
    if (hash_len <= 0 || (hash_len % max_g_bit) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by max_g_bit");
        return NULL;
    }
    expected_len = (hash_len + 7) / 8;
    if (partition_len != expected_len) {
        PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
        return NULL;
    }

    max_g_value = 1 << max_g_bit;
    if (!profile_counts_native(partition_value, partition_len, hash_len, max_g_bit, max_g_value, counts)) {
        return NULL;
    }

    result = PyTuple_New(max_g_value);
    if (result == NULL) {
        return NULL;
    }
    for (value = 0; value < max_g_value; ++value) {
        PyObject *py_count = PyLong_FromUnsignedLong((unsigned long)counts[value]);
        if (py_count == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(result, value, py_count);
    }
    return result;
}

static PyObject *
val_strict_isp_accept_check(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "partition_value",
        "hash_len",
        "max_g_bit",
        "window_low",
        "window_high",
        NULL,
    };
    const unsigned char *partition_value = NULL;
    Py_ssize_t partition_len = 0;
    Py_ssize_t hash_len = 0;
    int max_g_bit = 0;
    int max_g_value = 0;
    int window_low = 0;
    int window_high = 0;
    Py_ssize_t expected_len;
    uint32_t counts[64];
    int accepted = 0;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y#niii:val_strict_isp_accept_check",
            kwlist,
            &partition_value,
            &partition_len,
            &hash_len,
            &max_g_bit,
            &window_low,
            &window_high)) {
        return NULL;
    }

    if (max_g_bit < 2 || max_g_bit > 6) {
        PyErr_SetString(PyExc_ValueError, "max_g_bit must lie in [2, 6] for native ValStrictISP");
        return NULL;
    }
    if (hash_len <= 0 || (hash_len % max_g_bit) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by max_g_bit");
        return NULL;
    }
    expected_len = (hash_len + 7) / 8;
    if (partition_len != expected_len) {
        PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
        return NULL;
    }

    max_g_value = 1 << max_g_bit;
    if (!profile_counts_native_checked(
            partition_value,
            partition_len,
            hash_len,
            max_g_bit,
            max_g_value,
            counts,
            window_low,
            window_high,
            &accepted)) {
        return NULL;
    }
    if (accepted) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *
val_strict_isp_accept_check_fast(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    const unsigned char *partition_value = NULL;
    Py_ssize_t partition_len = 0;
    Py_ssize_t hash_len = 0;
    int max_g_bit = 0;
    int max_g_value = 0;
    int window_low = 0;
    int window_high = 0;
    Py_ssize_t expected_len;
    uint32_t counts[64];
    int accepted = 0;

    (void)self;

    if (nargs != 5) {
        PyErr_SetString(PyExc_TypeError, "val_strict_isp_accept_check_fast expects 5 positional arguments");
        return NULL;
    }
    if (!PyBytes_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "partition_value must be bytes");
        return NULL;
    }

    partition_value = (const unsigned char *)PyBytes_AS_STRING(args[0]);
    partition_len = PyBytes_GET_SIZE(args[0]);
    if (
        args[1] == g_accept_check_fast_hash_len_obj
        && args[2] == g_accept_check_fast_max_g_bit_obj
        && args[3] == g_accept_check_fast_window_low_obj
        && args[4] == g_accept_check_fast_window_high_obj
    ) {
        hash_len = g_accept_check_fast_hash_len;
        max_g_bit = g_accept_check_fast_max_g_bit;
        max_g_value = g_accept_check_fast_max_g_value;
        window_low = g_accept_check_fast_window_low;
        window_high = g_accept_check_fast_window_high;
        expected_len = g_accept_check_fast_expected_len;
    } else {
        hash_len = PyLong_AsSsize_t(args[1]);
        if (hash_len == -1 && PyErr_Occurred()) {
            return NULL;
        }
        {
            long value_long = PyLong_AsLong(args[2]);
            if (value_long == -1 && PyErr_Occurred()) {
                return NULL;
            }
            max_g_bit = (int)value_long;
            value_long = PyLong_AsLong(args[3]);
            if (value_long == -1 && PyErr_Occurred()) {
                return NULL;
            }
            window_low = (int)value_long;
            value_long = PyLong_AsLong(args[4]);
            if (value_long == -1 && PyErr_Occurred()) {
                return NULL;
            }
            window_high = (int)value_long;
        }

        if (max_g_bit < 2 || max_g_bit > 6) {
            PyErr_SetString(PyExc_ValueError, "max_g_bit must lie in [2, 6] for native ValStrictISP");
            return NULL;
        }
        if (hash_len <= 0 || (hash_len % max_g_bit) != 0) {
            PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by max_g_bit");
            return NULL;
        }
        expected_len = (hash_len + 7) / 8;
        max_g_value = 1 << max_g_bit;

        Py_INCREF(args[1]);
        Py_XDECREF(g_accept_check_fast_hash_len_obj);
        g_accept_check_fast_hash_len_obj = (PyObject *)args[1];
        Py_INCREF(args[2]);
        Py_XDECREF(g_accept_check_fast_max_g_bit_obj);
        g_accept_check_fast_max_g_bit_obj = (PyObject *)args[2];
        Py_INCREF(args[3]);
        Py_XDECREF(g_accept_check_fast_window_low_obj);
        g_accept_check_fast_window_low_obj = (PyObject *)args[3];
        Py_INCREF(args[4]);
        Py_XDECREF(g_accept_check_fast_window_high_obj);
        g_accept_check_fast_window_high_obj = (PyObject *)args[4];
        g_accept_check_fast_hash_len = hash_len;
        g_accept_check_fast_max_g_bit = max_g_bit;
        g_accept_check_fast_max_g_value = max_g_value;
        g_accept_check_fast_window_low = window_low;
        g_accept_check_fast_window_high = window_high;
        g_accept_check_fast_expected_len = expected_len;
    }

    if (partition_len != expected_len) {
        PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
        return NULL;
    }

    if (!profile_counts_native_checked(
            partition_value,
            partition_len,
            hash_len,
            max_g_bit,
            max_g_value,
            counts,
            window_low,
            window_high,
            &accepted)) {
        return NULL;
    }
    if (accepted) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *
val_strict_isp_w4_accept_check_fast(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    const unsigned char *partition_value = NULL;
    Py_ssize_t partition_len = 0;
    Py_ssize_t hash_len = 0;
    int window_low = 0;
    int window_high = 0;
    Py_ssize_t expected_len;
    uint32_t counts[4];
    int accepted = 0;

    (void)self;

    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError, "val_strict_isp_w4_accept_check_fast expects 4 positional arguments");
        return NULL;
    }
    if (!PyBytes_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "partition_value must be bytes");
        return NULL;
    }

    partition_value = (const unsigned char *)PyBytes_AS_STRING(args[0]);
    partition_len = PyBytes_GET_SIZE(args[0]);
    if (
        args[1] == g_w4_accept_check_fast_hash_len_obj
        && args[2] == g_w4_accept_check_fast_window_low_obj
        && args[3] == g_w4_accept_check_fast_window_high_obj
    ) {
        hash_len = g_w4_accept_check_fast_hash_len;
        window_low = g_w4_accept_check_fast_window_low;
        window_high = g_w4_accept_check_fast_window_high;
        expected_len = g_w4_accept_check_fast_expected_len;
    } else {
        hash_len = PyLong_AsSsize_t(args[1]);
        if (hash_len == -1 && PyErr_Occurred()) {
            return NULL;
        }
        {
            long value_long = PyLong_AsLong(args[2]);
            if (value_long == -1 && PyErr_Occurred()) {
                return NULL;
            }
            window_low = (int)value_long;
            value_long = PyLong_AsLong(args[3]);
            if (value_long == -1 && PyErr_Occurred()) {
                return NULL;
            }
            window_high = (int)value_long;
        }
        if (hash_len <= 0 || (hash_len % 2) != 0) {
            PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by 2");
            return NULL;
        }
        expected_len = (hash_len + 7) / 8;

        Py_INCREF(args[1]);
        Py_XDECREF(g_w4_accept_check_fast_hash_len_obj);
        g_w4_accept_check_fast_hash_len_obj = (PyObject *)args[1];
        Py_INCREF(args[2]);
        Py_XDECREF(g_w4_accept_check_fast_window_low_obj);
        g_w4_accept_check_fast_window_low_obj = (PyObject *)args[2];
        Py_INCREF(args[3]);
        Py_XDECREF(g_w4_accept_check_fast_window_high_obj);
        g_w4_accept_check_fast_window_high_obj = (PyObject *)args[3];
        g_w4_accept_check_fast_hash_len = hash_len;
        g_w4_accept_check_fast_window_low = window_low;
        g_w4_accept_check_fast_window_high = window_high;
        g_w4_accept_check_fast_expected_len = expected_len;
    }

    if (partition_len != expected_len) {
        PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
        return NULL;
    }

    if (!profile_counts_w4_checked(
            partition_value,
            partition_len,
            hash_len,
            counts,
            window_low,
            window_high,
            &accepted)) {
        return NULL;
    }
    if (accepted) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *
val_strict_isp_w4_accept_check_batch_fast(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyObject *sequence_obj;
    PyObject *sequence_fast = NULL;
    PyObject **items;
    Py_ssize_t value_count;
    Py_ssize_t hash_len;
    Py_ssize_t expected_len;
    Py_ssize_t repetitions;
    Py_ssize_t repetition;
    Py_ssize_t index;
    int window_low;
    int window_high;
    Py_ssize_t accepted_total = 0;

    (void)self;

    if (nargs != 5) {
        PyErr_SetString(PyExc_TypeError, "val_strict_isp_w4_accept_check_batch_fast expects 5 positional arguments");
        return NULL;
    }

    sequence_obj = args[0];
    hash_len = PyLong_AsSsize_t(args[1]);
    if (hash_len == -1 && PyErr_Occurred()) {
        return NULL;
    }
    {
        long value_long = PyLong_AsLong(args[2]);
        if (value_long == -1 && PyErr_Occurred()) {
            return NULL;
        }
        window_low = (int)value_long;
        value_long = PyLong_AsLong(args[3]);
        if (value_long == -1 && PyErr_Occurred()) {
            return NULL;
        }
        window_high = (int)value_long;
    }
    repetitions = PyLong_AsSsize_t(args[4]);
    if (repetitions == -1 && PyErr_Occurred()) {
        return NULL;
    }
    if (hash_len <= 0 || (hash_len % 2) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by 2");
        return NULL;
    }
    if (repetitions < 0) {
        PyErr_SetString(PyExc_ValueError, "repetitions must be non-negative");
        return NULL;
    }

    expected_len = (hash_len + 7) / 8;
    sequence_fast = PySequence_Fast(sequence_obj, "partition values must be a sequence");
    if (sequence_fast == NULL) {
        return NULL;
    }
    value_count = PySequence_Fast_GET_SIZE(sequence_fast);
    items = PySequence_Fast_ITEMS(sequence_fast);

    for (repetition = 0; repetition < repetitions; ++repetition) {
        for (index = 0; index < value_count; ++index) {
            PyObject *item = items[index];
            const unsigned char *partition_value;
            Py_ssize_t partition_len;
            uint32_t counts[4];
            int accepted = 0;

            if (!PyBytes_Check(item)) {
                Py_DECREF(sequence_fast);
                PyErr_SetString(PyExc_TypeError, "partition values must be bytes");
                return NULL;
            }
            partition_value = (const unsigned char *)PyBytes_AS_STRING(item);
            partition_len = PyBytes_GET_SIZE(item);
            if (partition_len != expected_len) {
                Py_DECREF(sequence_fast);
                PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
                return NULL;
            }
            if (!profile_counts_w4_checked(
                    partition_value,
                    partition_len,
                    hash_len,
                    counts,
                    window_low,
                    window_high,
                    &accepted)) {
                Py_DECREF(sequence_fast);
                return NULL;
            }
            accepted_total += accepted ? 1 : 0;
        }
    }

    Py_DECREF(sequence_fast);
    return PyLong_FromSsize_t(accepted_total);
}

static PyObject *
val_strict_isp_accept_check_batch_fast(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyObject *sequence_obj;
    PyObject *sequence_fast = NULL;
    PyObject **items;
    Py_ssize_t value_count;
    Py_ssize_t hash_len;
    Py_ssize_t expected_len;
    Py_ssize_t repetitions;
    Py_ssize_t repetition;
    Py_ssize_t index;
    int max_g_bit;
    int max_g_value;
    int window_low;
    int window_high;
    Py_ssize_t accepted_total = 0;

    (void)self;

    if (nargs != 6) {
        PyErr_SetString(PyExc_TypeError, "val_strict_isp_accept_check_batch_fast expects 6 positional arguments");
        return NULL;
    }

    sequence_obj = args[0];
    hash_len = PyLong_AsSsize_t(args[1]);
    if (hash_len == -1 && PyErr_Occurred()) {
        return NULL;
    }
    {
        long value_long = PyLong_AsLong(args[2]);
        if (value_long == -1 && PyErr_Occurred()) {
            return NULL;
        }
        max_g_bit = (int)value_long;
        value_long = PyLong_AsLong(args[3]);
        if (value_long == -1 && PyErr_Occurred()) {
            return NULL;
        }
        window_low = (int)value_long;
        value_long = PyLong_AsLong(args[4]);
        if (value_long == -1 && PyErr_Occurred()) {
            return NULL;
        }
        window_high = (int)value_long;
    }
    repetitions = PyLong_AsSsize_t(args[5]);
    if (repetitions == -1 && PyErr_Occurred()) {
        return NULL;
    }
    if (max_g_bit < 2 || max_g_bit > 6) {
        PyErr_SetString(PyExc_ValueError, "max_g_bit must lie in [2, 6] for native ValStrictISP");
        return NULL;
    }
    if (hash_len <= 0 || (hash_len % max_g_bit) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by max_g_bit");
        return NULL;
    }
    if (repetitions < 0) {
        PyErr_SetString(PyExc_ValueError, "repetitions must be non-negative");
        return NULL;
    }

    max_g_value = 1 << max_g_bit;
    expected_len = (hash_len + 7) / 8;
    sequence_fast = PySequence_Fast(sequence_obj, "partition values must be a sequence");
    if (sequence_fast == NULL) {
        return NULL;
    }
    value_count = PySequence_Fast_GET_SIZE(sequence_fast);
    items = PySequence_Fast_ITEMS(sequence_fast);

    for (repetition = 0; repetition < repetitions; ++repetition) {
        for (index = 0; index < value_count; ++index) {
            PyObject *item = items[index];
            const unsigned char *partition_value;
            Py_ssize_t partition_len;
            uint32_t counts[64];
            int accepted = 0;

            if (!PyBytes_Check(item)) {
                Py_DECREF(sequence_fast);
                PyErr_SetString(PyExc_TypeError, "partition values must be bytes");
                return NULL;
            }
            partition_value = (const unsigned char *)PyBytes_AS_STRING(item);
            partition_len = PyBytes_GET_SIZE(item);
            if (partition_len != expected_len) {
                Py_DECREF(sequence_fast);
                PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
                return NULL;
            }
            if (!profile_counts_native_checked(
                    partition_value,
                    partition_len,
                    hash_len,
                    max_g_bit,
                    max_g_value,
                    counts,
                    window_low,
                    window_high,
                    &accepted)) {
                Py_DECREF(sequence_fast);
                return NULL;
            }
            accepted_total += accepted ? 1 : 0;
        }
    }

    Py_DECREF(sequence_fast);
    return PyLong_FromSsize_t(accepted_total);
}

static PyObject *
val_strict_isp_profile_counts_batch_fast(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyObject *sequence_obj;
    PyObject *sequence_fast = NULL;
    PyObject **items;
    Py_ssize_t value_count;
    Py_ssize_t hash_len;
    Py_ssize_t expected_len;
    Py_ssize_t repetitions;
    Py_ssize_t repetition;
    Py_ssize_t index;
    int max_g_bit;
    int max_g_value;
    unsigned long long checksum = 0;

    (void)self;

    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError, "val_strict_isp_profile_counts_batch_fast expects 4 positional arguments");
        return NULL;
    }

    sequence_obj = args[0];
    hash_len = PyLong_AsSsize_t(args[1]);
    if (hash_len == -1 && PyErr_Occurred()) {
        return NULL;
    }
    {
        long value_long = PyLong_AsLong(args[2]);
        if (value_long == -1 && PyErr_Occurred()) {
            return NULL;
        }
        max_g_bit = (int)value_long;
    }
    repetitions = PyLong_AsSsize_t(args[3]);
    if (repetitions == -1 && PyErr_Occurred()) {
        return NULL;
    }
    if (max_g_bit < 2 || max_g_bit > 6) {
        PyErr_SetString(PyExc_ValueError, "max_g_bit must lie in [2, 6] for native ValStrictISP");
        return NULL;
    }
    if (hash_len <= 0 || (hash_len % max_g_bit) != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_len must be positive and divisible by max_g_bit");
        return NULL;
    }
    if (repetitions < 0) {
        PyErr_SetString(PyExc_ValueError, "repetitions must be non-negative");
        return NULL;
    }

    max_g_value = 1 << max_g_bit;
    expected_len = (hash_len + 7) / 8;
    sequence_fast = PySequence_Fast(sequence_obj, "partition values must be a sequence");
    if (sequence_fast == NULL) {
        return NULL;
    }
    value_count = PySequence_Fast_GET_SIZE(sequence_fast);
    items = PySequence_Fast_ITEMS(sequence_fast);

    for (repetition = 0; repetition < repetitions; ++repetition) {
        for (index = 0; index < value_count; ++index) {
            PyObject *item = items[index];
            const unsigned char *partition_value;
            Py_ssize_t partition_len;
            uint32_t counts[64];
            int value;

            if (!PyBytes_Check(item)) {
                Py_DECREF(sequence_fast);
                PyErr_SetString(PyExc_TypeError, "partition values must be bytes");
                return NULL;
            }
            partition_value = (const unsigned char *)PyBytes_AS_STRING(item);
            partition_len = PyBytes_GET_SIZE(item);
            if (partition_len != expected_len) {
                Py_DECREF(sequence_fast);
                PyErr_SetString(PyExc_ValueError, "partition_value byte length does not match hash_len");
                return NULL;
            }
            if (!profile_counts_native(
                    partition_value,
                    partition_len,
                    hash_len,
                    max_g_bit,
                    max_g_value,
                    counts)) {
                Py_DECREF(sequence_fast);
                return NULL;
            }
            for (value = 0; value < max_g_value; ++value) {
                checksum += (unsigned long long)counts[value];
            }
        }
    }

    Py_DECREF(sequence_fast);
    return PyLong_FromUnsignedLongLong(checksum);
}

static PyObject *
val_strict_isp_prepare_plan(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "partition_num",
        "window_low",
        "window_high",
        NULL,
    };
    int partition_num = 0;
    int window_low = 0;
    int window_high = 0;
    int count;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "iii:val_strict_isp_prepare_plan",
            kwlist,
            &partition_num,
            &window_low,
            &window_high)) {
        return NULL;
    }

    if (partition_num <= 0 || partition_num > 64) {
        PyErr_SetString(PyExc_ValueError, "partition_num must lie in [1, 64]");
        return NULL;
    }
    if (window_high < 0 || window_low > partition_num) {
        Py_RETURN_NONE;
    }
    if (window_low < 0) {
        window_low = 0;
    }
    if (window_high > partition_num) {
        window_high = partition_num;
    }

    for (count = window_low; count <= window_high; ++count) {
        if (!prepare_unrank_plan((unsigned int)partition_num, (unsigned int)count)) {
            return NULL;
        }
    }

    Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {
        "val_strict_isp_prepare_plan",
        (PyCFunction)val_strict_isp_prepare_plan,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Precompute native ValStrictISP unrank plans for a fixed PartitionNum/window."),
    },
    {
        "val_strict_isp_bytes",
        (PyCFunction)val_strict_isp_bytes,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Run the native ValStrictISP path for bytes partition values with 4 <= w <= 64."),
    },
    {
        "val_strict_isp_bytes_prefixed_seed",
        (PyCFunction)val_strict_isp_bytes_prefixed_seed,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Run the native ValStrictISP path using seed_prefix || partition_value after acceptance."),
    },
    {
        "val_strict_isp_bytes_default_seed",
        (PyCFunction)val_strict_isp_bytes_default_seed,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Run the native ValStrictISP path using the default ValStrictISP sample seed after acceptance."),
    },
    {
        "val_strict_isp_bytes_random",
        (PyCFunction)val_strict_isp_bytes_random,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Run the native ValStrictISP path using caller-supplied random bytes."),
    },
    {
        "val_strict_isp_bytes_random_fast",
        (PyCFunction)val_strict_isp_bytes_random_fast,
        METH_FASTCALL,
        PyDoc_STR("Run the native ValStrictISP random-bytes path with positional-only FASTCALL arguments."),
    },
    {
        "val_strict_isp_find_first_random_stream",
        (PyCFunction)val_strict_isp_find_first_random_stream,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Find the first accepting ValStrictISP candidate in a caller-supplied random stream."),
    },
    {
        "val_strict_isp_profile_counts",
        (PyCFunction)val_strict_isp_profile_counts,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Return the native multiplicity profile counts for a bytes partition value."),
    },
    {
        "val_strict_isp_accept_check",
        (PyCFunction)val_strict_isp_accept_check,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Run the native ValStrictISP multiplicity-profile acceptance check."),
    },
    {
        "val_strict_isp_accept_check_fast",
        (PyCFunction)val_strict_isp_accept_check_fast,
        METH_FASTCALL,
        PyDoc_STR("Run the native ValStrictISP multiplicity-profile acceptance check with positional-only FASTCALL arguments."),
    },
    {
        "val_strict_isp_accept_check_batch_fast",
        (PyCFunction)val_strict_isp_accept_check_batch_fast,
        METH_FASTCALL,
        PyDoc_STR("Run the native ValStrictISP acceptance check over a sequence and repetition count."),
    },
    {
        "val_strict_isp_profile_counts_batch_fast",
        (PyCFunction)val_strict_isp_profile_counts_batch_fast,
        METH_FASTCALL,
        PyDoc_STR("Run the native ValStrictISP multiplicity profile over a sequence and repetition count."),
    },
    {
        "val_strict_isp_w4_accept_check_fast",
        (PyCFunction)val_strict_isp_w4_accept_check_fast,
        METH_FASTCALL,
        PyDoc_STR("Run the w=4 native ValStrictISP acceptance check with positional-only FASTCALL arguments."),
    },
    {
        "val_strict_isp_w4_accept_check_batch_fast",
        (PyCFunction)val_strict_isp_w4_accept_check_batch_fast,
        METH_FASTCALL,
        PyDoc_STR("Run the w=4 native ValStrictISP acceptance check over a sequence and repetition count."),
    },
    {
        "val_strict_isp_w4_bytes",
        (PyCFunction)val_strict_isp_w4_bytes,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Run the w=4 ValStrictISP accepting/rejecting path on a bytes partition value."),
    },
    {
        "val_strict_isp_w4_bytes_prefixed_seed",
        (PyCFunction)val_strict_isp_w4_bytes_prefixed_seed,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Run the w=4 ValStrictISP path using seed_prefix || partition_value after acceptance."),
    },
    {
        "val_strict_isp_w4_bytes_default_seed",
        (PyCFunction)val_strict_isp_w4_bytes_default_seed,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Run the w=4 ValStrictISP path using the default ValStrictISP sample seed after acceptance."),
    },
    {
        "val_strict_isp_w4_bytes_random",
        (PyCFunction)val_strict_isp_w4_bytes_random,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Run the w=4 ValStrictISP path using caller-supplied random bytes."),
    },
    {
        "val_strict_isp_w4_bytes_random_fast",
        (PyCFunction)val_strict_isp_w4_bytes_random_fast,
        METH_FASTCALL,
        PyDoc_STR("Run the w=4 ValStrictISP random-bytes path with positional-only FASTCALL arguments."),
    },
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_val_strict_isp_native",
    "Native accelerators for ValStrictISP.",
    -1,
    module_methods,
};

PyMODINIT_FUNC
PyInit__val_strict_isp_native(void)
{
    PyObject *module;
    PyObject *hashlib_module;

    module = PyModule_Create(&module_def);
    if (module == NULL) {
        return NULL;
    }

    hashlib_module = PyImport_ImportModule("hashlib");
    if (hashlib_module == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    g_shake128_ctor = PyObject_GetAttrString(hashlib_module, "shake_128");
    g_shake256_ctor = PyObject_GetAttrString(hashlib_module, "shake_256");
    Py_DECREF(hashlib_module);
    if (g_shake128_ctor == NULL || g_shake256_ctor == NULL) {
        Py_XDECREF(g_shake128_ctor);
        Py_XDECREF(g_shake256_ctor);
        g_shake128_ctor = NULL;
        g_shake256_ctor = NULL;
        Py_DECREF(module);
        return NULL;
    }

    init_count_tables();
    init_comb_table();
    if (!init_subset_mask_tables()) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
