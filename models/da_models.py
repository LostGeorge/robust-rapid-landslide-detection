import segmentation_models_pytorch as smp

def instantiate_da_models(model=smp.UnetPlusPlus, encoder_name='resnet50', share_encoders=True, num_heads=3, state_dict_paths=None, num_channels=2, classes=2, **kwargs):
    """ This function instantiates models for the domain adaptation task with a shared encoder.

    Args:
        - model: The torch.nn.Module to instantiate
        - encoder_name (str): The encoder type from the list here: https://smp.readthedocs.io/en/latest/encoders_timm.html
        - share_encoders (bool): If the models should have a shared encoder
        - num_heads (int): The number of prediction heads for the da task
        - state_dict_paths (List[str]): The list of paths to state_dicts to intialize each of the models
        - num_channels (int): The number of in_channels for the model
        - num_classes (int): The number of classes for the prediction task
        - kwargs: Optional additional kwargs to override defaults for model instatiation
    Returns:
        - A tuple containing a shared encoder and a list of models  
    """

    # state_dict_paths should be the same length as num_heads if loading these models from memory
    if state_dict_paths is not None:
        assert len(state_dict_paths) == num_heads, f"len of state_dict_paths {len(state_dict_paths)} should match num_heads {num_heads}"

    encoder = None
    models = []
    for i in range(num_heads):
        net = model(encoder_name=encoder_name, in_channels=num_channels, classes=classes, **kwargs)
        if state_dict_paths is not None:
            net.load_state_dict(state_dict_paths[i])
        if encoder is None:
            encoder = net.encoder
        elif share_encoders:
            net.encoder = encoder
        models.append(net)
    return encoder, models
