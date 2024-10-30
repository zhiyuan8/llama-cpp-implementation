import argparse
from pathlib import Path

import gguf
import numpy as np
import torch

import sys
sys.path.append("../../../huggingface")
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

def to_ndarray(tensor: torch.Tensor, dtype: torch.dtype) -> np.ndarray:
    return tensor.detach().cpu().to(dtype).numpy()


def to_f32(tensor: torch.Tensor) -> np.ndarray:
    return to_ndarray(tensor, torch.float32)


def to_f16(tensor: torch.Tensor) -> np.ndarray:
    return to_ndarray(tensor, torch.float16)


def f16_criterion(name: str): # only quantize weights, not biases
    return "weight" in name and (
        "conv" in name
        or "self_attn.out_proj" in name
        or "self_attn.q_proj" in name
        or "self_attn.k_proj" in name
        or "self_attn.v_proj" in name
        or "fc1" in name
        or "fc2" in name
    )


def save_model_to_ggml(submodules: dict[str, torch.nn.Module], save_path: Path):
    gguf_writer = gguf.GGUFWriter(save_path, "qwen2-audio")
    # hparams
    all_hparams = {}
    for name, submodule in submodules.items():
        match name:
            case "audio_tower":
                hparam_list = [
                    "max_source_positions",  # n_audio_ctx
                    "d_model",  # n_audio_state
                    "encoder_attention_heads",  # n_audio_head
                    "encoder_layers",  # n_audio_layer
                ]
                for param in hparam_list:
                    all_hparams[param] = getattr(submodule.config, param)
            case "multi_modal_projector":
                pass
            case "language_model":
                hparam_list = [
                    "attention_dropout",
                    "attn_logit_softcapping",
                    "head_dim",
                    "hidden_size",  # n_text_state
                    "initializer_range",
                    "intermediate_size",
                    "max_position_embeddings",  # n_text_ctx
                    "num_attention_heads",  # n_text_head
                    "num_hidden_layers",  # n_text_layer
                    "num_key_value_heads",
                    "pad_token_id",
                    "query_pre_attn_scalar",
                    "rms_norm_eps",
                    "rope_theta",
                    "sliding_window",
                    "vocab_size",
                ]
                for param in hparam_list:
                    all_hparams[param] = getattr(submodule.config, param)
    
    print("=== all_hparams ===\n", all_hparams)
    for param, value in all_hparams.items():
        print(f"Writing hparam {param} = {value}")
        match str(type(value)):
            case "<class 'int'>":
                gguf_writer.add_int32(param, value)
            case "<class 'float'>":
                gguf_writer.add_float32(param, value)
            case "<class 'str'>":
                gguf_writer.add_string(param, value)
            case "<class 'list'>":
                gguf_writer.add_array(param, value)
            case _:
                raise ValueError(f"Unsupported type {str(type(value))}")

    # tensors
    for submodule_name, submodule in submodules.items():
        state_dict = submodule.state_dict() # use state_dict to get all tensors
        for subname, tensor in state_dict.items():
            name = f"{submodule_name}.{subname}"
            print(f"Writing tensor {name} with shape {tensor.shape}")
            ndarray = (
                to_f16(tensor) if f16_criterion(subname) else to_f32(tensor)
            )
            gguf_writer.add_tensor(name, ndarray)

    # Write the GGUF file
    print("Writing GGUF file ...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    print(f"GGUF model saved to '{save_path}'")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_id_or_path",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
    )
    parser.add_argument(
        "-o", "--output_file_name", type=str, default="qwen2-audio-instruct"
    )
    parser.add_argument(
        "--submodules",
        type=str,
        nargs="+",
        default=["audio_tower", "multi_modal_projector"], # do not handle language_model, need to process it separately
    )
    parser.add_argument("-d", "--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=== args ===\n", args)

    # load checkpoint
    local_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.model_id_or_path,
            device_map=args.device,
            torch_dtype=torch.bfloat16
    )

    submodules = {
        submodule: getattr(local_model, submodule) for submodule in args.submodules
    }

    print("=== submodules ===\n", submodules)

    output_path = args.output_file_name + "." + "-".join(args.submodules) + ".gguf"
    save_path = Path(output_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_model_to_ggml(submodules, save_path)