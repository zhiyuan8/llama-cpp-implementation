#include "qwen2audio.h"

int main(int argc, char **argv)
{

    omni_params params;

    if (!omni_params_parse(argc, argv, params))
    {
        return 1;
    }

    omni_context *ctx_omni = omni_init_context(params);

    omni_process_full(ctx_omni, params);

    omni_free(ctx_omni);

    return 0;
}