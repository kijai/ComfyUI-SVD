from sgm.util import instantiate_from_config
from einops import rearrange, repeat
from omegaconf import OmegaConf
import math
import torch

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))
def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    lowvram_mode: bool
):
    
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    if lowvram_mode:
        model.model.half()
    return model

class SVDimg2vid:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "version": (
            [   'svd',
                'svd_xt',
                'svd_image_decoder',
                'svd_xt_image_decoder',
            ],
            {
            "default": 'svd'
            }),
                "image": ("IMAGE",),
                "num_frames": ("INT", {"default": 14}),
                "num_steps": ("INT", {"default": 25}),
                "fps_id": ("INT", {"default": 6}),
                "motion_bucket_id": ("INT", {"default": 127}),
                "cond_aug": ("FLOAT", {"default": 0.02}),
                "seed": ("INT", {"default": 2331121321}),
                "decoding_t": ("INT", {"default": 1}),
                "lowvram_mode": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"

    CATEGORY = "KJNodes/experimental"

    def generate(self, image, version, num_frames, num_steps, fps_id, motion_bucket_id, cond_aug, seed, decoding_t, lowvram_mode):
       
        device: str = "cuda"

        model_config = f"generativemodels/scripts/sampling/configs/{version}.yaml"

        model = load_model(
            model_config,
            device,
            num_frames,
            num_steps,
            lowvram_mode,
        )

        torch.manual_seed(seed)
        image = image.permute(0, 3, 1, 2) 
        image = image * 2.0 - 1.0
        image = image.to(device)
        B, C, H, W = image.shape
        assert C == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024):
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )
        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        value_dict["cond_aug"] = cond_aug

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                randn = torch.randn(shape, device=device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                samples = samples.permute(0, 2, 3, 1)
        results = samples.cpu()
        return (results,)
        
        
NODE_CLASS_MAPPINGS = {
    "SVDimg2vid": SVDimg2vid,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SVDimg2vid": "SVDimg2vid",
}