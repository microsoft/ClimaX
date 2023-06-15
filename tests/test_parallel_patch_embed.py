import torch

from climax.arch import ClimaX


def test_parallel_patch_embed():
    vars = tuple(["a", "b", "c"])
    x = torch.rand(4, len(vars), 32, 64)
    serial_model = ClimaX(vars, img_size=[32, 64], patch_size=4, embed_dim=128, parallel_patch_embed=False)
    parallel_model = ClimaX(vars, img_size=[32, 64], patch_size=4, embed_dim=128, parallel_patch_embed=True)
    assert serial_model.token_embeds[0].num_patches == parallel_model.num_patches
    assert len(serial_model.token_embeds) == len(parallel_model.token_embeds.proj_weights)
    for i in range(len(serial_model.token_embeds)):
        serial_model.token_embeds[i].proj.weight.data = parallel_model.token_embeds.proj_weights[i].data
        serial_model.token_embeds[i].proj.bias.data = parallel_model.token_embeds.proj_biases[i].data

    test_vars = [["a"], ["b"], ["c"], ["a", "b"], ["a", "c"], ["b", "c"], ["a", "b", "c"]]
    for vars in test_vars:
        vars = tuple(vars)
        serial_var_ids = serial_model.get_var_ids(vars, x.device)
        parallel_var_ids = parallel_model.get_var_ids(vars, x.device)
        assert all(serial_var_ids == parallel_var_ids)
        x = torch.rand(4, len(vars), 32, 64)
        parallel_embed = parallel_model.token_embeds(x, parallel_var_ids)
        embeds = []
        for i in range(len(serial_var_ids)):
            id = serial_var_ids[i]
            embeds.append(serial_model.token_embeds[id](x[:, i : i + 1, :, :]))

        serial_embed = torch.stack(embeds, dim=1)
        assert parallel_embed.shape == serial_embed.shape
        assert torch.allclose(serial_embed, parallel_embed)


if __name__ == "__main__":
    test_parallel_patch_embed()
