#include "omni.h"

int main(int argc, char **argv)
{

    omni_context_params ctx_params = omni_context_default_params();
    if (!omni_context_params_parse(argc, argv, ctx_params))
    {
        return 1;
    }

    omni_context *ctx_omni = omni_init_context(ctx_params);

    omni_process_full(ctx_omni, ctx_params);

    omni_free(ctx_omni);

    return 0;
}