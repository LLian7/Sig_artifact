#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>
#include <string.h>

static PyObject *g_shake128_ctor = NULL;
static PyObject *g_shake256_ctor = NULL;

typedef struct {
    uint64_t *targets;
    Py_ssize_t target_count;
    const unsigned char *expand_prefix;
    Py_ssize_t expand_prefix_len;
    PyObject *expand_prefix_obj;
    Py_ssize_t seed_len;
    int message_length;
    int use_shake128;
    PyObject *compact_frontier;
    PyObject *frontier_ranges;
    PyObject *outputs;
} PPRFTraversalContext;

static Py_ssize_t
lower_bound_u64(
    const uint64_t *values,
    Py_ssize_t low,
    Py_ssize_t high,
    uint64_t target
)
{
    while (low < high) {
        Py_ssize_t mid = low + ((high - low) >> 1);
        if (values[mid] < target) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

static int
parse_sorted_u64_sequence(
    PyObject *obj,
    const char *name,
    uint64_t **out_values,
    Py_ssize_t *out_count
)
{
    PyObject *seq = PySequence_Fast(obj, name);
    Py_ssize_t count;
    uint64_t *values;
    Py_ssize_t index;

    if (seq == NULL) {
        return 0;
    }

    count = PySequence_Fast_GET_SIZE(seq);
    values = NULL;
    if (count > 0) {
        values = (uint64_t *)PyMem_Malloc((size_t)count * sizeof(uint64_t));
        if (values == NULL) {
            Py_DECREF(seq);
            PyErr_NoMemory();
            return 0;
        }
    }

    for (index = 0; index < count; ++index) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, index);
        unsigned long long value = PyLong_AsUnsignedLongLong(item);
        if (PyErr_Occurred()) {
            PyMem_Free(values);
            Py_DECREF(seq);
            return 0;
        }
        if (index > 0 && value < values[index - 1]) {
            PyMem_Free(values);
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError, "%s must be sorted", name);
            return 0;
        }
        values[index] = (uint64_t)value;
    }

    Py_DECREF(seq);
    *out_values = values;
    *out_count = count;
    return 1;
}

static PyObject *
xof_digest_from_parts(
    PyObject **parts,
    Py_ssize_t *part_lens,
    Py_ssize_t part_count,
    Py_ssize_t output_len,
    int use_shake128
)
{
    Py_ssize_t total_len = 0;
    Py_ssize_t index;
    PyObject *payload;
    unsigned char *write_ptr;
    PyObject *hash_object;
    PyObject *digest;

    for (index = 0; index < part_count; ++index) {
        if (part_lens[index] > PY_SSIZE_T_MAX - total_len) {
            PyErr_SetString(PyExc_OverflowError, "hash payload is too large");
            return NULL;
        }
        total_len += part_lens[index];
    }

    payload = PyBytes_FromStringAndSize(NULL, total_len);
    if (payload == NULL) {
        return NULL;
    }

    write_ptr = (unsigned char *)PyBytes_AS_STRING(payload);
    for (index = 0; index < part_count; ++index) {
        const char *part = PyBytes_AS_STRING(parts[index]);
        memcpy(write_ptr, part, (size_t)part_lens[index]);
        write_ptr += part_lens[index];
    }

    hash_object = PyObject_CallFunctionObjArgs(
        use_shake128 ? g_shake128_ctor : g_shake256_ctor,
        payload,
        NULL
    );
    Py_DECREF(payload);
    if (hash_object == NULL) {
        return NULL;
    }

    digest = PyObject_CallMethod(hash_object, "digest", "n", output_len);
    Py_DECREF(hash_object);
    return digest;
}

static int
expand_seed(
    PPRFTraversalContext *ctx,
    PyObject *seed,
    PyObject **left_seed,
    PyObject **right_seed
)
{
    PyObject *parts[2];
    Py_ssize_t part_lens[2];
    PyObject *expanded;
    const char *expanded_data;

    if (!PyBytes_Check(seed) || PyBytes_GET_SIZE(seed) != ctx->seed_len) {
        PyErr_SetString(PyExc_ValueError, "seed has incorrect byte length");
        return 0;
    }

    parts[0] = ctx->expand_prefix_obj;
    parts[1] = seed;
    part_lens[0] = ctx->expand_prefix_len;
    part_lens[1] = ctx->seed_len;
    expanded = xof_digest_from_parts(parts, part_lens, 2, 2 * ctx->seed_len, ctx->use_shake128);
    if (expanded == NULL) {
        return 0;
    }
    if (!PyBytes_Check(expanded) || PyBytes_GET_SIZE(expanded) != 2 * ctx->seed_len) {
        Py_DECREF(expanded);
        PyErr_SetString(PyExc_RuntimeError, "unexpected SHAKE digest length");
        return 0;
    }

    expanded_data = PyBytes_AS_STRING(expanded);
    *left_seed = PyBytes_FromStringAndSize(expanded_data, ctx->seed_len);
    *right_seed = PyBytes_FromStringAndSize(expanded_data + ctx->seed_len, ctx->seed_len);
    Py_DECREF(expanded);
    if (*left_seed == NULL || *right_seed == NULL) {
        Py_XDECREF(*left_seed);
        Py_XDECREF(*right_seed);
        *left_seed = NULL;
        *right_seed = NULL;
        return 0;
    }
    return 1;
}

static int
append_frontier_seed(
    PPRFTraversalContext *ctx,
    PyObject *seed,
    int depth,
    uint64_t low,
    uint64_t high
)
{
    PyObject *range_tuple;

    if (PyList_Append(ctx->compact_frontier, seed) < 0) {
        return 0;
    }

    range_tuple = Py_BuildValue(
        "(iKK)",
        depth,
        (unsigned long long)low,
        (unsigned long long)high
    );
    if (range_tuple == NULL) {
        return 0;
    }
    if (PyList_Append(ctx->frontier_ranges, range_tuple) < 0) {
        Py_DECREF(range_tuple);
        return 0;
    }
    Py_DECREF(range_tuple);
    return 1;
}

static int
set_output_seed(PPRFTraversalContext *ctx, Py_ssize_t index, PyObject *seed)
{
    Py_INCREF(seed);
    if (PyList_SetItem(ctx->outputs, index, seed) < 0) {
        Py_DECREF(seed);
        return 0;
    }
    return 1;
}

static int
puncture_visit(
    PPRFTraversalContext *ctx,
    PyObject *seed,
    int depth,
    uint64_t low,
    uint64_t high,
    Py_ssize_t start,
    Py_ssize_t end
)
{
    PyObject *left_seed = NULL;
    PyObject *right_seed = NULL;
    uint64_t mid_value;
    Py_ssize_t mid;
    int ok;

    if (start >= end) {
        return append_frontier_seed(ctx, seed, depth, low, high);
    }
    if (depth == ctx->message_length) {
        return set_output_seed(ctx, start, seed);
    }
    if (depth > ctx->message_length || high < low) {
        PyErr_SetString(PyExc_ValueError, "invalid PPRF range");
        return 0;
    }

    mid_value = low + ((high - low) >> 1);
    mid = lower_bound_u64(ctx->targets, start, end, mid_value);
    if (!expand_seed(ctx, seed, &left_seed, &right_seed)) {
        return 0;
    }

    ok = puncture_visit(ctx, left_seed, depth + 1, low, mid_value, start, mid)
        && puncture_visit(ctx, right_seed, depth + 1, mid_value, high, mid, end);
    Py_DECREF(left_seed);
    Py_DECREF(right_seed);
    return ok;
}

static int
leaf_eval_subtree(
    PPRFTraversalContext *ctx,
    PyObject *seed,
    int depth,
    uint64_t low,
    uint64_t high,
    Py_ssize_t start,
    Py_ssize_t end
)
{
    PyObject *left_seed = NULL;
    PyObject *right_seed = NULL;
    uint64_t mid_value;
    Py_ssize_t mid;
    int ok;

    if (start >= end) {
        return 1;
    }
    if (depth == ctx->message_length) {
        return set_output_seed(ctx, start, seed);
    }
    if (depth > ctx->message_length || high < low) {
        PyErr_SetString(PyExc_ValueError, "invalid PPRF range");
        return 0;
    }

    mid_value = low + ((high - low) >> 1);
    mid = lower_bound_u64(ctx->targets, start, end, mid_value);
    if (!expand_seed(ctx, seed, &left_seed, &right_seed)) {
        return 0;
    }

    ok = leaf_eval_subtree(ctx, left_seed, depth + 1, low, mid_value, start, mid)
        && leaf_eval_subtree(ctx, right_seed, depth + 1, mid_value, high, mid, end);
    Py_DECREF(left_seed);
    Py_DECREF(right_seed);
    return ok;
}

static int
dense_eval_subtree(
    PPRFTraversalContext *ctx,
    PyObject *seed,
    int depth,
    uint64_t low,
    uint64_t high
)
{
    PyObject *left_seed = NULL;
    PyObject *right_seed = NULL;
    uint64_t mid_value;
    int ok;

    if (depth == ctx->message_length) {
        if (high != low + 1 || low > (uint64_t)PY_SSIZE_T_MAX || low >= (uint64_t)PyList_GET_SIZE(ctx->outputs)) {
            PyErr_SetString(PyExc_ValueError, "invalid dense PPRF leaf range");
            return 0;
        }
        return set_output_seed(ctx, (Py_ssize_t)low, seed);
    }
    if (depth > ctx->message_length || high <= low) {
        PyErr_SetString(PyExc_ValueError, "invalid dense PPRF range");
        return 0;
    }

    mid_value = low + ((high - low) >> 1);
    if (!expand_seed(ctx, seed, &left_seed, &right_seed)) {
        return 0;
    }
    ok = dense_eval_subtree(ctx, left_seed, depth + 1, low, mid_value)
        && dense_eval_subtree(ctx, right_seed, depth + 1, mid_value, high);
    Py_DECREF(left_seed);
    Py_DECREF(right_seed);
    return ok;
}

static PyObject *
ycsig_pprf_puncture_and_reveal(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "provider_ranges",
        "selected_indices",
        "expand_prefix",
        "seed_bytes",
        "message_length",
        "use_shake128",
        NULL,
    };
    PyObject *provider_ranges_obj;
    PyObject *selected_indices_obj;
    PyObject *expand_prefix_obj;
    Py_ssize_t seed_len;
    int message_length;
    int use_shake128;
    PyObject *providers_seq = NULL;
    PyObject *compact_tuple = NULL;
    PyObject *range_tuple = NULL;
    PyObject *outputs_tuple = NULL;
    PyObject *result = NULL;
    PPRFTraversalContext ctx;
    Py_ssize_t provider_count;
    Py_ssize_t provider_index;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOnip:pprf_puncture_and_reveal",
            kwlist,
            &provider_ranges_obj,
            &selected_indices_obj,
            &expand_prefix_obj,
            &seed_len,
            &message_length,
            &use_shake128)) {
        return NULL;
    }
    if (!PyBytes_Check(expand_prefix_obj)) {
        PyErr_SetString(PyExc_TypeError, "expand_prefix must be bytes");
        return NULL;
    }
    if (seed_len <= 0 || message_length < 0 || message_length > 63) {
        PyErr_SetString(PyExc_ValueError, "unsupported PPRF parameters");
        return NULL;
    }

    memset(&ctx, 0, sizeof(ctx));
    if (!parse_sorted_u64_sequence(selected_indices_obj, "selected_indices", &ctx.targets, &ctx.target_count)) {
        return NULL;
    }
    ctx.expand_prefix = (const unsigned char *)PyBytes_AS_STRING(expand_prefix_obj);
    ctx.expand_prefix_len = PyBytes_GET_SIZE(expand_prefix_obj);
    ctx.expand_prefix_obj = expand_prefix_obj;
    ctx.seed_len = seed_len;
    ctx.message_length = message_length;
    ctx.use_shake128 = use_shake128;
    ctx.compact_frontier = PyList_New(0);
    ctx.frontier_ranges = PyList_New(0);
    ctx.outputs = PyList_New(ctx.target_count);
    if (ctx.compact_frontier == NULL || ctx.frontier_ranges == NULL || ctx.outputs == NULL) {
        goto done;
    }
    for (Py_ssize_t i = 0; i < ctx.target_count; ++i) {
        Py_INCREF(Py_None);
        PyList_SET_ITEM(ctx.outputs, i, Py_None);
    }

    providers_seq = PySequence_Fast(provider_ranges_obj, "provider_ranges must be a sequence");
    if (providers_seq == NULL) {
        goto done;
    }
    provider_count = PySequence_Fast_GET_SIZE(providers_seq);
    for (provider_index = 0; provider_index < provider_count; ++provider_index) {
        PyObject *provider = PySequence_Fast_GET_ITEM(providers_seq, provider_index);
        PyObject *provider_seq = PySequence_Fast(provider, "provider entry must be a sequence");
        int depth;
        uint64_t low;
        uint64_t high;
        PyObject *seed;
        Py_ssize_t start;
        Py_ssize_t end;

        if (provider_seq == NULL) {
            goto done;
        }
        if (PySequence_Fast_GET_SIZE(provider_seq) != 4) {
            Py_DECREF(provider_seq);
            PyErr_SetString(PyExc_ValueError, "provider entry must have four fields");
            goto done;
        }
        depth = (int)PyLong_AsLong(PySequence_Fast_GET_ITEM(provider_seq, 0));
        low = (uint64_t)PyLong_AsUnsignedLongLong(PySequence_Fast_GET_ITEM(provider_seq, 1));
        high = (uint64_t)PyLong_AsUnsignedLongLong(PySequence_Fast_GET_ITEM(provider_seq, 2));
        seed = PySequence_Fast_GET_ITEM(provider_seq, 3);
        if (PyErr_Occurred()) {
            Py_DECREF(provider_seq);
            goto done;
        }
        if (!PyBytes_Check(seed) || PyBytes_GET_SIZE(seed) != seed_len) {
            Py_DECREF(provider_seq);
            PyErr_SetString(PyExc_ValueError, "provider seed has incorrect byte length");
            goto done;
        }

        start = lower_bound_u64(ctx.targets, 0, ctx.target_count, low);
        end = lower_bound_u64(ctx.targets, start, ctx.target_count, high);
        if (!puncture_visit(&ctx, seed, depth, low, high, start, end)) {
            Py_DECREF(provider_seq);
            goto done;
        }
        Py_DECREF(provider_seq);
    }

    compact_tuple = PyList_AsTuple(ctx.compact_frontier);
    range_tuple = PyList_AsTuple(ctx.frontier_ranges);
    outputs_tuple = PyList_AsTuple(ctx.outputs);
    if (compact_tuple == NULL || range_tuple == NULL || outputs_tuple == NULL) {
        goto done;
    }
    result = PyTuple_Pack(3, compact_tuple, range_tuple, outputs_tuple);

done:
    PyMem_Free(ctx.targets);
    Py_XDECREF(providers_seq);
    Py_XDECREF(ctx.compact_frontier);
    Py_XDECREF(ctx.frontier_ranges);
    Py_XDECREF(ctx.outputs);
    Py_XDECREF(compact_tuple);
    Py_XDECREF(range_tuple);
    Py_XDECREF(outputs_tuple);
    return result;
}

static PyObject *
ycsig_pprf_leaf_material(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "stored_seeds",
        "frontier_ranges",
        "complement_indices",
        "expand_prefix",
        "seed_bytes",
        "message_length",
        "use_shake128",
        NULL,
    };
    PyObject *stored_seeds_obj;
    PyObject *frontier_ranges_obj;
    PyObject *complement_indices_obj;
    PyObject *expand_prefix_obj;
    Py_ssize_t seed_len;
    int message_length;
    int use_shake128;
    PyObject *seeds_seq = NULL;
    PyObject *ranges_seq = NULL;
    PyObject *outputs_tuple = NULL;
    PPRFTraversalContext ctx;
    Py_ssize_t seed_count;
    Py_ssize_t range_count;
    Py_ssize_t index;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOnip:pprf_leaf_material",
            kwlist,
            &stored_seeds_obj,
            &frontier_ranges_obj,
            &complement_indices_obj,
            &expand_prefix_obj,
            &seed_len,
            &message_length,
            &use_shake128)) {
        return NULL;
    }
    if (!PyBytes_Check(expand_prefix_obj)) {
        PyErr_SetString(PyExc_TypeError, "expand_prefix must be bytes");
        return NULL;
    }
    if (seed_len <= 0 || message_length < 0 || message_length > 63) {
        PyErr_SetString(PyExc_ValueError, "unsupported PPRF parameters");
        return NULL;
    }

    memset(&ctx, 0, sizeof(ctx));
    if (!parse_sorted_u64_sequence(complement_indices_obj, "complement_indices", &ctx.targets, &ctx.target_count)) {
        return NULL;
    }
    ctx.expand_prefix = (const unsigned char *)PyBytes_AS_STRING(expand_prefix_obj);
    ctx.expand_prefix_len = PyBytes_GET_SIZE(expand_prefix_obj);
    ctx.expand_prefix_obj = expand_prefix_obj;
    ctx.seed_len = seed_len;
    ctx.message_length = message_length;
    ctx.use_shake128 = use_shake128;
    ctx.outputs = PyList_New(ctx.target_count);
    if (ctx.outputs == NULL) {
        goto error;
    }
    for (Py_ssize_t i = 0; i < ctx.target_count; ++i) {
        Py_INCREF(Py_None);
        PyList_SET_ITEM(ctx.outputs, i, Py_None);
    }

    seeds_seq = PySequence_Fast(stored_seeds_obj, "stored_seeds must be a sequence");
    ranges_seq = PySequence_Fast(frontier_ranges_obj, "frontier_ranges must be a sequence");
    if (seeds_seq == NULL || ranges_seq == NULL) {
        goto error;
    }
    seed_count = PySequence_Fast_GET_SIZE(seeds_seq);
    range_count = PySequence_Fast_GET_SIZE(ranges_seq);
    if (seed_count != range_count) {
        PyErr_SetString(PyExc_ValueError, "stored_seeds does not match frontier_ranges");
        goto error;
    }

    for (index = 0; index < seed_count; ++index) {
        PyObject *seed = PySequence_Fast_GET_ITEM(seeds_seq, index);
        PyObject *range = PySequence_Fast_GET_ITEM(ranges_seq, index);
        PyObject *range_seq = PySequence_Fast(range, "frontier range must be a sequence");
        int depth;
        uint64_t low;
        uint64_t high;
        Py_ssize_t start;
        Py_ssize_t end;

        if (range_seq == NULL) {
            goto error;
        }
        if (PySequence_Fast_GET_SIZE(range_seq) != 3) {
            Py_DECREF(range_seq);
            PyErr_SetString(PyExc_ValueError, "frontier range must have three fields");
            goto error;
        }
        depth = (int)PyLong_AsLong(PySequence_Fast_GET_ITEM(range_seq, 0));
        low = (uint64_t)PyLong_AsUnsignedLongLong(PySequence_Fast_GET_ITEM(range_seq, 1));
        high = (uint64_t)PyLong_AsUnsignedLongLong(PySequence_Fast_GET_ITEM(range_seq, 2));
        if (PyErr_Occurred()) {
            Py_DECREF(range_seq);
            goto error;
        }
        if (!PyBytes_Check(seed) || PyBytes_GET_SIZE(seed) != seed_len) {
            Py_DECREF(range_seq);
            PyErr_SetString(PyExc_ValueError, "stored seed has incorrect byte length");
            goto error;
        }

        start = lower_bound_u64(ctx.targets, 0, ctx.target_count, low);
        end = lower_bound_u64(ctx.targets, start, ctx.target_count, high);
        if (!leaf_eval_subtree(&ctx, seed, depth, low, high, start, end)) {
            Py_DECREF(range_seq);
            goto error;
        }
        Py_DECREF(range_seq);
    }

    outputs_tuple = PyList_AsTuple(ctx.outputs);
    PyMem_Free(ctx.targets);
    Py_XDECREF(seeds_seq);
    Py_XDECREF(ranges_seq);
    Py_XDECREF(ctx.outputs);
    return outputs_tuple;

error:
    PyMem_Free(ctx.targets);
    Py_XDECREF(seeds_seq);
    Py_XDECREF(ranges_seq);
    Py_XDECREF(ctx.outputs);
    return NULL;
}

static PyObject *
ycsig_pprf_leaf_material_dense(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "provider_ranges",
        "expand_prefix",
        "seed_bytes",
        "message_length",
        "domain_size",
        "use_shake128",
        NULL,
    };
    PyObject *provider_ranges_obj;
    PyObject *expand_prefix_obj;
    Py_ssize_t seed_len;
    int message_length;
    Py_ssize_t domain_size;
    int use_shake128;
    PyObject *providers_seq = NULL;
    PyObject *outputs_list = NULL;
    PPRFTraversalContext ctx;
    Py_ssize_t provider_count;
    Py_ssize_t provider_index;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOninp:pprf_leaf_material_dense",
            kwlist,
            &provider_ranges_obj,
            &expand_prefix_obj,
            &seed_len,
            &message_length,
            &domain_size,
            &use_shake128)) {
        return NULL;
    }
    if (!PyBytes_Check(expand_prefix_obj)) {
        PyErr_SetString(PyExc_TypeError, "expand_prefix must be bytes");
        return NULL;
    }
    if (seed_len <= 0 || message_length < 0 || message_length > 63 || domain_size < 0) {
        PyErr_SetString(PyExc_ValueError, "unsupported PPRF parameters");
        return NULL;
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.expand_prefix = (const unsigned char *)PyBytes_AS_STRING(expand_prefix_obj);
    ctx.expand_prefix_len = PyBytes_GET_SIZE(expand_prefix_obj);
    ctx.expand_prefix_obj = expand_prefix_obj;
    ctx.seed_len = seed_len;
    ctx.message_length = message_length;
    ctx.use_shake128 = use_shake128;
    ctx.outputs = PyList_New(domain_size);
    if (ctx.outputs == NULL) {
        goto error;
    }
    for (Py_ssize_t i = 0; i < domain_size; ++i) {
        Py_INCREF(Py_None);
        PyList_SET_ITEM(ctx.outputs, i, Py_None);
    }

    providers_seq = PySequence_Fast(provider_ranges_obj, "provider_ranges must be a sequence");
    if (providers_seq == NULL) {
        goto error;
    }
    provider_count = PySequence_Fast_GET_SIZE(providers_seq);
    for (provider_index = 0; provider_index < provider_count; ++provider_index) {
        PyObject *provider = PySequence_Fast_GET_ITEM(providers_seq, provider_index);
        PyObject *provider_seq = PySequence_Fast(provider, "provider entry must be a sequence");
        int depth;
        uint64_t low;
        uint64_t high;
        PyObject *seed;

        if (provider_seq == NULL) {
            goto error;
        }
        if (PySequence_Fast_GET_SIZE(provider_seq) != 4) {
            Py_DECREF(provider_seq);
            PyErr_SetString(PyExc_ValueError, "provider entry must have four fields");
            goto error;
        }
        depth = (int)PyLong_AsLong(PySequence_Fast_GET_ITEM(provider_seq, 0));
        low = (uint64_t)PyLong_AsUnsignedLongLong(PySequence_Fast_GET_ITEM(provider_seq, 1));
        high = (uint64_t)PyLong_AsUnsignedLongLong(PySequence_Fast_GET_ITEM(provider_seq, 2));
        seed = PySequence_Fast_GET_ITEM(provider_seq, 3);
        if (PyErr_Occurred()) {
            Py_DECREF(provider_seq);
            goto error;
        }
        if (high > (uint64_t)domain_size || low > high) {
            Py_DECREF(provider_seq);
            PyErr_SetString(PyExc_ValueError, "provider range is outside the active domain");
            goto error;
        }
        if (!PyBytes_Check(seed) || PyBytes_GET_SIZE(seed) != seed_len) {
            Py_DECREF(provider_seq);
            PyErr_SetString(PyExc_ValueError, "provider seed has incorrect byte length");
            goto error;
        }
        if (!dense_eval_subtree(&ctx, seed, depth, low, high)) {
            Py_DECREF(provider_seq);
            goto error;
        }
        Py_DECREF(provider_seq);
    }

    for (Py_ssize_t i = 0; i < domain_size; ++i) {
        if (PyList_GET_ITEM(ctx.outputs, i) == Py_None) {
            PyErr_SetString(PyExc_ValueError, "provider ranges did not cover the active domain");
            goto error;
        }
    }
    outputs_list = ctx.outputs;
    ctx.outputs = NULL;
    Py_XDECREF(providers_seq);
    Py_XDECREF(ctx.outputs);
    return outputs_list;

error:
    Py_XDECREF(providers_seq);
    Py_XDECREF(ctx.outputs);
    return NULL;
}

static PyObject *
merkle_hash_node(
    PyObject *prefix,
    PyObject *left,
    PyObject *right,
    Py_ssize_t output_len,
    int use_shake128
)
{
    PyObject *parts[3];
    Py_ssize_t part_lens[3];

    if (!PyBytes_Check(prefix) || !PyBytes_Check(left) || !PyBytes_Check(right)) {
        PyErr_SetString(PyExc_TypeError, "Merkle hash inputs must be bytes");
        return NULL;
    }
    parts[0] = prefix;
    parts[1] = left;
    parts[2] = right;
    part_lens[0] = PyBytes_GET_SIZE(prefix);
    part_lens[1] = PyBytes_GET_SIZE(left);
    part_lens[2] = PyBytes_GET_SIZE(right);
    return xof_digest_from_parts(parts, part_lens, 3, output_len, use_shake128);
}

static int
parse_level_widths(PyObject *level_widths_obj, int tree_height, Py_ssize_t **out_widths)
{
    PyObject *seq = PySequence_Fast(level_widths_obj, "level_widths must be a sequence");
    Py_ssize_t *widths;
    Py_ssize_t count;
    Py_ssize_t index;

    if (seq == NULL) {
        return 0;
    }
    count = PySequence_Fast_GET_SIZE(seq);
    if (count != (Py_ssize_t)tree_height + 1) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "level_widths length does not match tree_height");
        return 0;
    }

    widths = (Py_ssize_t *)PyMem_Malloc((size_t)count * sizeof(Py_ssize_t));
    if (widths == NULL) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return 0;
    }
    for (index = 0; index < count; ++index) {
        Py_ssize_t width = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(seq, index));
        if (PyErr_Occurred()) {
            PyMem_Free(widths);
            Py_DECREF(seq);
            return 0;
        }
        if (width <= 0) {
            PyMem_Free(widths);
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "level widths must be positive");
            return 0;
        }
        widths[index] = width;
    }
    Py_DECREF(seq);
    *out_widths = widths;
    return 1;
}

static void
free_known_levels(PyObject ***levels, Py_ssize_t *widths, int tree_height)
{
    int level;
    if (levels == NULL) {
        return;
    }
    for (level = 0; level <= tree_height; ++level) {
        if (levels[level] != NULL) {
            Py_ssize_t offset;
            for (offset = 0; offset < widths[level]; ++offset) {
                Py_XDECREF(levels[level][offset]);
            }
            PyMem_Free(levels[level]);
        }
    }
    PyMem_Free(levels);
}

static PyObject ***
alloc_known_levels(Py_ssize_t *widths, int tree_height)
{
    PyObject ***levels;
    int level;

    levels = (PyObject ***)PyMem_Calloc((size_t)tree_height + 1, sizeof(PyObject **));
    if (levels == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (level = 0; level <= tree_height; ++level) {
        levels[level] = (PyObject **)PyMem_Calloc((size_t)widths[level], sizeof(PyObject *));
        if (levels[level] == NULL) {
            PyErr_NoMemory();
            free_known_levels(levels, widths, tree_height);
            return NULL;
        }
    }
    return levels;
}

static PyObject *
ycsig_merkle_sparse_rebuild(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "level_widths",
        "node_hash_prefixes",
        "positions",
        "values",
        "complementary_indices",
        "complementary_leaf_values",
        "output_bytes",
        "tree_height",
        "use_shake128",
        NULL,
    };
    PyObject *level_widths_obj;
    PyObject *node_hash_prefixes_obj;
    PyObject *positions_obj;
    PyObject *values_obj;
    PyObject *complementary_indices_obj;
    PyObject *complementary_leaf_values_obj;
    Py_ssize_t output_bytes;
    int tree_height;
    int use_shake128;
    Py_ssize_t *widths = NULL;
    PyObject ***levels = NULL;
    PyObject *prefix_levels_seq = NULL;
    PyObject **prefix_level_seqs = NULL;
    PyObject *positions_seq = NULL;
    PyObject *values_seq = NULL;
    PyObject *comp_indices_seq = NULL;
    PyObject *comp_values_seq = NULL;
    PyObject *root = NULL;
    Py_ssize_t count;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOOOnip:merkle_sparse_rebuild",
            kwlist,
            &level_widths_obj,
            &node_hash_prefixes_obj,
            &positions_obj,
            &values_obj,
            &complementary_indices_obj,
            &complementary_leaf_values_obj,
            &output_bytes,
            &tree_height,
            &use_shake128)) {
        return NULL;
    }
    if (output_bytes <= 0 || tree_height < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid Merkle parameters");
        return NULL;
    }
    if (!parse_level_widths(level_widths_obj, tree_height, &widths)) {
        return NULL;
    }
    levels = alloc_known_levels(widths, tree_height);
    if (levels == NULL) {
        goto done;
    }

    prefix_levels_seq = PySequence_Fast(node_hash_prefixes_obj, "node_hash_prefixes must be a sequence");
    if (prefix_levels_seq == NULL) {
        goto done;
    }
    if (PySequence_Fast_GET_SIZE(prefix_levels_seq) != (Py_ssize_t)tree_height + 1) {
        PyErr_SetString(PyExc_ValueError, "node_hash_prefixes length does not match tree_height");
        goto done;
    }
    prefix_level_seqs = (PyObject **)PyMem_Calloc((size_t)tree_height + 1, sizeof(PyObject *));
    if (prefix_level_seqs == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    for (int level = 0; level <= tree_height; ++level) {
        PyObject *level_prefixes = PySequence_Fast_GET_ITEM(prefix_levels_seq, level);
        prefix_level_seqs[level] = PySequence_Fast(level_prefixes, "node hash prefix level must be a sequence");
        if (prefix_level_seqs[level] == NULL) {
            goto done;
        }
        if (PySequence_Fast_GET_SIZE(prefix_level_seqs[level]) < widths[level]) {
            PyErr_SetString(PyExc_ValueError, "node hash prefix level is too short");
            goto done;
        }
    }

    comp_indices_seq = PySequence_Fast(complementary_indices_obj, "complementary_indices must be a sequence");
    comp_values_seq = PySequence_Fast(complementary_leaf_values_obj, "complementary_leaf_values must be a sequence");
    if (comp_indices_seq == NULL || comp_values_seq == NULL) {
        goto done;
    }
    count = PySequence_Fast_GET_SIZE(comp_indices_seq);
    if (count != PySequence_Fast_GET_SIZE(comp_values_seq)) {
        PyErr_SetString(PyExc_ValueError, "complementary indices and leaves do not match");
        goto done;
    }
    for (Py_ssize_t index = 0; index < count; ++index) {
        Py_ssize_t alpha = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(comp_indices_seq, index));
        PyObject *value = PySequence_Fast_GET_ITEM(comp_values_seq, index);
        if (PyErr_Occurred()) {
            goto done;
        }
        if (alpha < 0 || alpha >= widths[0]) {
            PyErr_SetString(PyExc_ValueError, "complementary leaf index is out of range");
            goto done;
        }
        if (!PyBytes_Check(value)) {
            PyErr_SetString(PyExc_TypeError, "complementary leaf value must be bytes");
            goto done;
        }
        Py_XDECREF(levels[0][alpha]);
        Py_INCREF(value);
        levels[0][alpha] = value;
    }

    positions_seq = PySequence_Fast(positions_obj, "positions must be a sequence");
    values_seq = PySequence_Fast(values_obj, "values must be a sequence");
    if (positions_seq == NULL || values_seq == NULL) {
        goto done;
    }
    count = PySequence_Fast_GET_SIZE(positions_seq);
    if (count != PySequence_Fast_GET_SIZE(values_seq)) {
        PyErr_SetString(PyExc_ValueError, "values does not match positions");
        goto done;
    }
    for (Py_ssize_t index = 0; index < count; ++index) {
        PyObject *position = PySequence_Fast_GET_ITEM(positions_seq, index);
        PyObject *position_seq = PySequence_Fast(position, "position must be a sequence");
        PyObject *value = PySequence_Fast_GET_ITEM(values_seq, index);
        int level;
        Py_ssize_t offset;
        if (position_seq == NULL) {
            goto done;
        }
        if (PySequence_Fast_GET_SIZE(position_seq) != 2) {
            Py_DECREF(position_seq);
            PyErr_SetString(PyExc_ValueError, "position must have two fields");
            goto done;
        }
        level = (int)PyLong_AsLong(PySequence_Fast_GET_ITEM(position_seq, 0));
        offset = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(position_seq, 1));
        Py_DECREF(position_seq);
        if (PyErr_Occurred()) {
            goto done;
        }
        if (level < 0 || level > tree_height || offset < 0 || offset >= widths[level]) {
            PyErr_SetString(PyExc_ValueError, "partial-state position is out of range");
            goto done;
        }
        if (levels[level][offset] != NULL) {
            PyErr_SetString(PyExc_ValueError, "the same position appears in both partial_state and complementary_leaves");
            goto done;
        }
        if (!PyBytes_Check(value)) {
            PyErr_SetString(PyExc_TypeError, "partial-state value must be bytes");
            goto done;
        }
        Py_INCREF(value);
        levels[level][offset] = value;
    }

    for (int level = 1; level <= tree_height; ++level) {
        Py_ssize_t child_width = widths[level - 1];
        PyObject **child_level = levels[level - 1];
        PyObject **parent_level = levels[level];
        PyObject *prefix_level = prefix_level_seqs[level];
        for (Py_ssize_t offset = 0; offset < widths[level]; ++offset) {
            Py_ssize_t child_offset;
            Py_ssize_t right_offset;
            PyObject *left;
            PyObject *right;
            PyObject *hashed;

            if (parent_level[offset] != NULL) {
                continue;
            }
            child_offset = offset << 1;
            if (child_offset >= child_width) {
                continue;
            }
            left = child_level[child_offset];
            if (left == NULL) {
                continue;
            }
            right_offset = child_offset + 1;
            if (right_offset >= child_width) {
                Py_INCREF(left);
                parent_level[offset] = left;
                continue;
            }
            right = child_level[right_offset];
            if (right == NULL) {
                continue;
            }
            hashed = merkle_hash_node(
                PySequence_Fast_GET_ITEM(prefix_level, offset),
                left,
                right,
                output_bytes,
                use_shake128
            );
            if (hashed == NULL) {
                goto done;
            }
            parent_level[offset] = hashed;
        }
    }

    if (levels[tree_height][0] == NULL) {
        PyErr_SetString(PyExc_ValueError, "insufficient information to rebuild the Merkle root");
        goto done;
    }
    Py_INCREF(levels[tree_height][0]);
    root = levels[tree_height][0];

done:
    if (prefix_level_seqs != NULL) {
        for (int level = 0; level <= tree_height; ++level) {
            Py_XDECREF(prefix_level_seqs[level]);
        }
        PyMem_Free(prefix_level_seqs);
    }
    free_known_levels(levels, widths, tree_height);
    PyMem_Free(widths);
    Py_XDECREF(prefix_levels_seq);
    Py_XDECREF(positions_seq);
    Py_XDECREF(values_seq);
    Py_XDECREF(comp_indices_seq);
    Py_XDECREF(comp_values_seq);
    return root;
}

static PyObject *
ycsig_merkle_root_from_leaves(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "leaves",
        "level_widths",
        "node_hash_prefixes",
        "output_bytes",
        "tree_height",
        "use_shake128",
        NULL,
    };
    PyObject *leaves_obj;
    PyObject *level_widths_obj;
    PyObject *node_hash_prefixes_obj;
    Py_ssize_t output_bytes;
    int tree_height;
    int use_shake128;
    Py_ssize_t *widths = NULL;
    PyObject *leaves_seq = NULL;
    PyObject *prefix_levels_seq = NULL;
    PyObject **prefix_level_seqs = NULL;
    PyObject **current = NULL;
    Py_ssize_t current_width = 0;
    PyObject *root = NULL;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOnip:merkle_root_from_leaves",
            kwlist,
            &leaves_obj,
            &level_widths_obj,
            &node_hash_prefixes_obj,
            &output_bytes,
            &tree_height,
            &use_shake128)) {
        return NULL;
    }
    if (output_bytes <= 0 || tree_height < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid Merkle parameters");
        return NULL;
    }
    if (!parse_level_widths(level_widths_obj, tree_height, &widths)) {
        return NULL;
    }

    leaves_seq = PySequence_Fast(leaves_obj, "leaves must be a sequence");
    if (leaves_seq == NULL) {
        goto done;
    }
    if (PySequence_Fast_GET_SIZE(leaves_seq) != widths[0]) {
        PyErr_SetString(PyExc_ValueError, "leaf count does not match level_widths");
        goto done;
    }

    prefix_levels_seq = PySequence_Fast(node_hash_prefixes_obj, "node_hash_prefixes must be a sequence");
    if (prefix_levels_seq == NULL) {
        goto done;
    }
    if (PySequence_Fast_GET_SIZE(prefix_levels_seq) != (Py_ssize_t)tree_height + 1) {
        PyErr_SetString(PyExc_ValueError, "node_hash_prefixes length does not match tree_height");
        goto done;
    }
    prefix_level_seqs = (PyObject **)PyMem_Calloc((size_t)tree_height + 1, sizeof(PyObject *));
    if (prefix_level_seqs == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    for (int level = 0; level <= tree_height; ++level) {
        PyObject *level_prefixes = PySequence_Fast_GET_ITEM(prefix_levels_seq, level);
        prefix_level_seqs[level] = PySequence_Fast(level_prefixes, "node hash prefix level must be a sequence");
        if (prefix_level_seqs[level] == NULL) {
            goto done;
        }
        if (PySequence_Fast_GET_SIZE(prefix_level_seqs[level]) < widths[level]) {
            PyErr_SetString(PyExc_ValueError, "node hash prefix level is too short");
            goto done;
        }
    }

    current = (PyObject **)PyMem_Calloc((size_t)widths[0], sizeof(PyObject *));
    if (current == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    current_width = widths[0];
    for (Py_ssize_t offset = 0; offset < widths[0]; ++offset) {
        PyObject *leaf = PySequence_Fast_GET_ITEM(leaves_seq, offset);
        if (!PyBytes_Check(leaf)) {
            PyErr_SetString(PyExc_TypeError, "leaf values must be bytes");
            goto done;
        }
        Py_INCREF(leaf);
        current[offset] = leaf;
    }

    for (int level = 1; level <= tree_height; ++level) {
        PyObject **next = (PyObject **)PyMem_Calloc((size_t)widths[level], sizeof(PyObject *));
        PyObject *prefix_level = prefix_level_seqs[level];
        if (next == NULL) {
            PyErr_NoMemory();
            goto done;
        }
        for (Py_ssize_t offset = 0; offset < widths[level]; ++offset) {
            Py_ssize_t child_offset = offset << 1;
            Py_ssize_t right_offset = child_offset + 1;
            if (right_offset >= widths[level - 1]) {
                Py_INCREF(current[child_offset]);
                next[offset] = current[child_offset];
            } else {
                next[offset] = merkle_hash_node(
                    PySequence_Fast_GET_ITEM(prefix_level, offset),
                    current[child_offset],
                    current[right_offset],
                    output_bytes,
                    use_shake128
                );
                if (next[offset] == NULL) {
                    for (Py_ssize_t cleanup = 0; cleanup < widths[level]; ++cleanup) {
                        Py_XDECREF(next[cleanup]);
                    }
                    PyMem_Free(next);
                    goto done;
                }
            }
        }
        for (Py_ssize_t offset = 0; offset < widths[level - 1]; ++offset) {
            Py_DECREF(current[offset]);
        }
        PyMem_Free(current);
        current = next;
        current_width = widths[level];
    }

    if (current == NULL || widths[tree_height] != 1 || current[0] == NULL) {
        PyErr_SetString(PyExc_ValueError, "failed to build Merkle root");
        goto done;
    }
    Py_INCREF(current[0]);
    root = current[0];

done:
    if (current != NULL) {
        for (Py_ssize_t offset = 0; offset < current_width; ++offset) {
            Py_XDECREF(current[offset]);
        }
        PyMem_Free(current);
    }
    if (prefix_level_seqs != NULL) {
        for (int level = 0; level <= tree_height; ++level) {
            Py_XDECREF(prefix_level_seqs[level]);
        }
        PyMem_Free(prefix_level_seqs);
    }
    PyMem_Free(widths);
    Py_XDECREF(leaves_seq);
    Py_XDECREF(prefix_levels_seq);
    return root;
}

static PyObject *
ycsig_merkle_compact_partial_state(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "level_widths",
        "node_hash_prefixes",
        "positions",
        "signed_indices",
        "signed_leaf_values",
        "output_bytes",
        "tree_height",
        "use_shake128",
        NULL,
    };
    PyObject *level_widths_obj;
    PyObject *node_hash_prefixes_obj;
    PyObject *positions_obj;
    PyObject *signed_indices_obj;
    PyObject *signed_leaf_values_obj;
    Py_ssize_t output_bytes;
    int tree_height;
    int use_shake128;
    Py_ssize_t *widths = NULL;
    PyObject ***levels = NULL;
    PyObject *prefix_levels_seq = NULL;
    PyObject **prefix_level_seqs = NULL;
    PyObject *positions_seq = NULL;
    PyObject *signed_indices_seq = NULL;
    PyObject *signed_values_seq = NULL;
    PyObject *result = NULL;
    Py_ssize_t count;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOOnip:merkle_compact_partial_state",
            kwlist,
            &level_widths_obj,
            &node_hash_prefixes_obj,
            &positions_obj,
            &signed_indices_obj,
            &signed_leaf_values_obj,
            &output_bytes,
            &tree_height,
            &use_shake128)) {
        return NULL;
    }
    if (output_bytes <= 0 || tree_height < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid Merkle parameters");
        return NULL;
    }
    if (!parse_level_widths(level_widths_obj, tree_height, &widths)) {
        return NULL;
    }
    levels = alloc_known_levels(widths, tree_height);
    if (levels == NULL) {
        goto done;
    }

    prefix_levels_seq = PySequence_Fast(node_hash_prefixes_obj, "node_hash_prefixes must be a sequence");
    if (prefix_levels_seq == NULL) {
        goto done;
    }
    if (PySequence_Fast_GET_SIZE(prefix_levels_seq) != (Py_ssize_t)tree_height + 1) {
        PyErr_SetString(PyExc_ValueError, "node_hash_prefixes length does not match tree_height");
        goto done;
    }
    prefix_level_seqs = (PyObject **)PyMem_Calloc((size_t)tree_height + 1, sizeof(PyObject *));
    if (prefix_level_seqs == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    for (int level = 0; level <= tree_height; ++level) {
        PyObject *level_prefixes = PySequence_Fast_GET_ITEM(prefix_levels_seq, level);
        prefix_level_seqs[level] = PySequence_Fast(level_prefixes, "node hash prefix level must be a sequence");
        if (prefix_level_seqs[level] == NULL) {
            goto done;
        }
        if (PySequence_Fast_GET_SIZE(prefix_level_seqs[level]) < widths[level]) {
            PyErr_SetString(PyExc_ValueError, "node hash prefix level is too short");
            goto done;
        }
    }

    signed_indices_seq = PySequence_Fast(signed_indices_obj, "signed_indices must be a sequence");
    signed_values_seq = PySequence_Fast(signed_leaf_values_obj, "signed_leaf_values must be a sequence");
    if (signed_indices_seq == NULL || signed_values_seq == NULL) {
        goto done;
    }
    count = PySequence_Fast_GET_SIZE(signed_indices_seq);
    if (count != PySequence_Fast_GET_SIZE(signed_values_seq)) {
        PyErr_SetString(PyExc_ValueError, "signed indices and leaves do not match");
        goto done;
    }
    for (Py_ssize_t index = 0; index < count; ++index) {
        Py_ssize_t alpha = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(signed_indices_seq, index));
        PyObject *value = PySequence_Fast_GET_ITEM(signed_values_seq, index);
        if (PyErr_Occurred()) {
            goto done;
        }
        if (alpha < 0 || alpha >= widths[0]) {
            PyErr_SetString(PyExc_ValueError, "signed leaf index is out of range");
            goto done;
        }
        if (!PyBytes_Check(value)) {
            PyErr_SetString(PyExc_TypeError, "signed leaf value must be bytes");
            goto done;
        }
        Py_XDECREF(levels[0][alpha]);
        Py_INCREF(value);
        levels[0][alpha] = value;
    }

    for (int level = 1; level <= tree_height; ++level) {
        Py_ssize_t child_width = widths[level - 1];
        PyObject **child_level = levels[level - 1];
        PyObject **parent_level = levels[level];
        PyObject *prefix_level = prefix_level_seqs[level];
        for (Py_ssize_t offset = 0; offset < widths[level]; ++offset) {
            Py_ssize_t child_offset = offset << 1;
            Py_ssize_t right_offset;
            PyObject *left;
            PyObject *right;
            PyObject *hashed;

            if (child_offset >= child_width) {
                continue;
            }
            left = child_level[child_offset];
            if (left == NULL) {
                continue;
            }
            right_offset = child_offset + 1;
            if (right_offset >= child_width) {
                Py_INCREF(left);
                parent_level[offset] = left;
                continue;
            }
            right = child_level[right_offset];
            if (right == NULL) {
                continue;
            }
            hashed = merkle_hash_node(
                PySequence_Fast_GET_ITEM(prefix_level, offset),
                left,
                right,
                output_bytes,
                use_shake128
            );
            if (hashed == NULL) {
                goto done;
            }
            parent_level[offset] = hashed;
        }
    }

    positions_seq = PySequence_Fast(positions_obj, "positions must be a sequence");
    if (positions_seq == NULL) {
        goto done;
    }
    count = PySequence_Fast_GET_SIZE(positions_seq);
    result = PyTuple_New(count);
    if (result == NULL) {
        goto done;
    }
    for (Py_ssize_t index = 0; index < count; ++index) {
        PyObject *position = PySequence_Fast_GET_ITEM(positions_seq, index);
        PyObject *position_seq = PySequence_Fast(position, "position must be a sequence");
        int level;
        Py_ssize_t offset;
        PyObject *value;
        if (position_seq == NULL) {
            Py_CLEAR(result);
            goto done;
        }
        if (PySequence_Fast_GET_SIZE(position_seq) != 2) {
            Py_DECREF(position_seq);
            Py_CLEAR(result);
            PyErr_SetString(PyExc_ValueError, "position must have two fields");
            goto done;
        }
        level = (int)PyLong_AsLong(PySequence_Fast_GET_ITEM(position_seq, 0));
        offset = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(position_seq, 1));
        Py_DECREF(position_seq);
        if (PyErr_Occurred()) {
            Py_CLEAR(result);
            goto done;
        }
        if (level < 0 || level > tree_height || offset < 0 || offset >= widths[level]) {
            Py_CLEAR(result);
            PyErr_SetString(PyExc_ValueError, "partial-state position is out of range");
            goto done;
        }
        value = levels[level][offset];
        if (value == NULL) {
            Py_CLEAR(result);
            PyErr_SetString(PyExc_ValueError, "missing leaf required to build the partial state");
            goto done;
        }
        Py_INCREF(value);
        PyTuple_SET_ITEM(result, index, value);
    }

done:
    if (prefix_level_seqs != NULL) {
        for (int level = 0; level <= tree_height; ++level) {
            Py_XDECREF(prefix_level_seqs[level]);
        }
        PyMem_Free(prefix_level_seqs);
    }
    free_known_levels(levels, widths, tree_height);
    PyMem_Free(widths);
    Py_XDECREF(prefix_levels_seq);
    Py_XDECREF(positions_seq);
    Py_XDECREF(signed_indices_seq);
    Py_XDECREF(signed_values_seq);
    return result;
}

typedef struct {
    uint64_t *targets;
    Py_ssize_t target_count;
    Py_ssize_t *widths;
    int tree_height;
    uint64_t leaf_count;
    PyObject **level_lists;
} CoverPositionsContext;

static int
cover_append_position(CoverPositionsContext *ctx, int level, uint64_t offset)
{
    PyObject *position = Py_BuildValue("(iK)", level, (unsigned long long)offset);
    if (position == NULL) {
        return 0;
    }
    if (PyList_Append(ctx->level_lists[level], position) < 0) {
        Py_DECREF(position);
        return 0;
    }
    Py_DECREF(position);
    return 1;
}

static int
cover_visit(
    CoverPositionsContext *ctx,
    int level,
    uint64_t offset,
    Py_ssize_t start_index,
    Py_ssize_t end_index
)
{
    uint64_t span;
    uint64_t start;
    uint64_t end;
    uint64_t selected_count;
    uint64_t child_offset;
    uint64_t left_end;
    Py_ssize_t mid_index;

    if (start_index >= end_index) {
        return 1;
    }
    if (level < 0 || level >= 63) {
        PyErr_SetString(PyExc_ValueError, "unsupported Merkle tree height");
        return 0;
    }

    span = ((uint64_t)1) << level;
    start = offset * span;
    end = start + span;
    if (end > ctx->leaf_count || end < start) {
        end = ctx->leaf_count;
    }
    if (start >= end) {
        return 1;
    }

    selected_count = (uint64_t)(end_index - start_index);
    if (selected_count == end - start || level == 0) {
        return cover_append_position(ctx, level, offset);
    }

    child_offset = offset << 1;
    if (child_offset >= (uint64_t)ctx->widths[level - 1]) {
        PyErr_SetString(PyExc_ValueError, "invalid Merkle cover child offset");
        return 0;
    }
    if (child_offset + 1 >= (uint64_t)ctx->widths[level - 1]) {
        return cover_visit(ctx, level - 1, child_offset, start_index, end_index);
    }

    left_end = start + (((uint64_t)1) << (level - 1));
    if (left_end > ctx->leaf_count || left_end < start) {
        left_end = ctx->leaf_count;
    }
    mid_index = lower_bound_u64(ctx->targets, start_index, end_index, left_end);
    if (start_index < mid_index) {
        if (!cover_visit(ctx, level - 1, child_offset, start_index, mid_index)) {
            return 0;
        }
    }
    if (mid_index < end_index) {
        if (!cover_visit(ctx, level - 1, child_offset + 1, mid_index, end_index)) {
            return 0;
        }
    }
    return 1;
}

static PyObject *
ycsig_canonical_cover_positions(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
        "level_widths",
        "signed_indices",
        "tree_height",
        NULL,
    };
    PyObject *level_widths_obj;
    PyObject *signed_indices_obj;
    int tree_height;
    CoverPositionsContext ctx;
    PyObject *result = NULL;
    Py_ssize_t total_count = 0;
    Py_ssize_t write_index = 0;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOi:canonical_cover_positions",
            kwlist,
            &level_widths_obj,
            &signed_indices_obj,
            &tree_height)) {
        return NULL;
    }
    if (tree_height < 0 || tree_height >= 63) {
        PyErr_SetString(PyExc_ValueError, "unsupported Merkle tree height");
        return NULL;
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.tree_height = tree_height;
    if (!parse_level_widths(level_widths_obj, tree_height, &ctx.widths)) {
        return NULL;
    }
    ctx.leaf_count = (uint64_t)ctx.widths[0];
    if (!parse_sorted_u64_sequence(signed_indices_obj, "signed_indices", &ctx.targets, &ctx.target_count)) {
        goto done;
    }
    if (ctx.target_count == 0) {
        result = PyTuple_New(0);
        goto done;
    }
    if (ctx.targets[ctx.target_count - 1] >= ctx.leaf_count) {
        PyErr_SetString(PyExc_ValueError, "signed leaf index is out of range");
        goto done;
    }

    ctx.level_lists = (PyObject **)PyMem_Calloc((size_t)tree_height + 1, sizeof(PyObject *));
    if (ctx.level_lists == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    for (int level = 0; level <= tree_height; ++level) {
        ctx.level_lists[level] = PyList_New(0);
        if (ctx.level_lists[level] == NULL) {
            goto done;
        }
    }

    if (!cover_visit(&ctx, tree_height, 0, 0, ctx.target_count)) {
        goto done;
    }
    for (int level = 0; level <= tree_height; ++level) {
        total_count += PyList_GET_SIZE(ctx.level_lists[level]);
    }
    result = PyTuple_New(total_count);
    if (result == NULL) {
        goto done;
    }
    for (int level = 0; level <= tree_height; ++level) {
        Py_ssize_t level_count = PyList_GET_SIZE(ctx.level_lists[level]);
        for (Py_ssize_t offset = 0; offset < level_count; ++offset) {
            PyObject *position = PyList_GET_ITEM(ctx.level_lists[level], offset);
            Py_INCREF(position);
            PyTuple_SET_ITEM(result, write_index, position);
            write_index += 1;
        }
    }

done:
    if (ctx.level_lists != NULL) {
        for (int level = 0; level <= tree_height; ++level) {
            Py_XDECREF(ctx.level_lists[level]);
        }
        PyMem_Free(ctx.level_lists);
    }
    PyMem_Free(ctx.targets);
    PyMem_Free(ctx.widths);
    return result;
}

static PyMethodDef module_methods[] = {
    {
        "pprf_puncture_and_reveal",
        (PyCFunction)ycsig_pprf_puncture_and_reveal,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Run native integer PPRF puncture-and-reveal traversal."),
    },
    {
        "pprf_leaf_material",
        (PyCFunction)ycsig_pprf_leaf_material,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Run native integer PPRF leaf-material traversal from compact frontier ranges."),
    },
    {
        "pprf_leaf_material_dense",
        (PyCFunction)ycsig_pprf_leaf_material_dense,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Run native integer PPRF leaf-material traversal over the active domain."),
    },
    {
        "merkle_sparse_rebuild",
        (PyCFunction)ycsig_merkle_sparse_rebuild,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Rebuild a sparse Merkle root using a native level-order traversal."),
    },
    {
        "merkle_root_from_leaves",
        (PyCFunction)ycsig_merkle_root_from_leaves,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Build a Merkle root from hashed leaves using a native level-order traversal."),
    },
    {
        "merkle_compact_partial_state",
        (PyCFunction)ycsig_merkle_compact_partial_state,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Build compact Merkle partial-state values from signed leaves."),
    },
    {
        "canonical_cover_positions",
        (PyCFunction)ycsig_canonical_cover_positions,
        METH_VARARGS | METH_KEYWORDS,
        PyDoc_STR("Compute canonical Merkle cover positions for sorted leaf indices."),
    },
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_yc_sig_native",
    "Native accelerators for YCSig.",
    -1,
    module_methods,
};

PyMODINIT_FUNC
PyInit__yc_sig_native(void)
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

    return module;
}
