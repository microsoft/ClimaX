import torch
import numpy as np


def collect_outputs(outputs, multi_gpu):
    log_dict = {}
    for loss_type in outputs[0]:
        if multi_gpu:
            collect = []
            for output in outputs:
                for v in output[loss_type]:
                    if v == v:
                        collect.append(v)
        else:
            collect = [v[loss_type] for v in outputs if v[loss_type] == v[loss_type]]
        if collect:
            log_dict[loss_type] = torch.stack(collect).mean().item()
        else:
            log_dict[loss_type] = float('nan')
    return log_dict


def define_loss_fn(lat2d):
    weights_lat = compute_latitude_weighting(lat2d)
    loss = lambda x, y: torch.sqrt(compute_weighted_mse(x, y, weights_lat))
    return weights_lat, loss


def compute_latitude_weighting(lat):
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    return weights_lat


def compute_weighted_mse(pred, truth, weights_lat, flat_weights=False):
    """
    Compute the MSE with latitude weighting.
    Args:
        pred : Forecast. Torch tensor.
        truth: Truth. Torch tensor.
        weights_lat: Latitude weighting, 2d Torch tensor. 
    Returns:
        mse: Latitude weighted mean squared error
    """
    if not flat_weights:
        weights_lat = truth.new(weights_lat).expand_as(truth)
    error = (pred - truth)**2
    out = error * weights_lat
    return out.mean()


def eval_loss(pred, output, lts, loss_function, possible_lead_times, phase='val', target_v=None, normalizer=None):
    results = {}
    # Unpick which of the batch samples contain which lead_time
    lead_time_dist = {t: lts == t for t in possible_lead_times}
    results[f'{phase}_loss'] = loss_function(pred, output)
    # Caclulate loss per lead_time
    for t, cond in lead_time_dist.items():
        if any(cond):
            results[f'{phase}_loss_{t}hrs'] = loss_function(pred[cond], output[cond])
        else:
            results[f'{phase}_loss_{t}hrs'] = pred.new([float('nan')])[0]
    
    # Undo normalization
    if normalizer:
        # scaled_pred_v = (torch.exp(pred[:, 0, :, :]) - 1 ) * normalizer[target_v]['std']
        # scaled_output_v = (torch.exp(output[:, 0, :, :]) - 1) * normalizer[target_v]['std']
        scaled_pred_v = pred[:, 0, :, :] * normalizer[target_v]['std'] + normalizer[target_v]['mean']
        scaled_output_v = output[:, 0, :, :] * normalizer[target_v]['std'] + normalizer[target_v]['mean']
        if target_v == 'tp':
            scaled_pred_v *= 1e3
            scaled_output_v *= 1e3

        results[f'{phase}_loss_' + target_v] = loss_function(scaled_pred_v, scaled_output_v)
       # Caclulate loss per lead_time
        for t, cond in lead_time_dist.items(): 
            if any(cond):
                results[f'{phase}_loss_{target_v}_{t}hrs'] = loss_function(scaled_pred_v[cond], scaled_output_v[cond])
            else:
                results[f'{phase}_loss_{target_v}_{t}hrs'] = scaled_pred_v.new([float('nan')])[0]
    return results


def convert_precip_to_mm(output, target_v, normalizer):
    converted = (np.exp(output) - 1) * normalizer[target_v]['std']
    if target_v == 'tp':
        converted *= 1e3
    return converted
