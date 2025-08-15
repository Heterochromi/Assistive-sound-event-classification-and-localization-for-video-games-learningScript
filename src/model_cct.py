from vit_pytorch.cct import CCT
import torch.nn as nn



class MultiLabelCCT(nn.Module):
    def __init__(
        self,
        img_size=224,
        embedding_dim=384,
        n_input_channels=3,
        n_conv_layers=2,
        kernel_size=7,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        num_layers=14,
        num_heads=6,
        mlp_ratio=3.0,
        num_classes=80,
        dropout_rate=0.,
        attention_dropout=0.1,
        stochastic_depth=0.1,
        positional_embedding='learnable',
        **kwargs
    ):
        super().__init__()
        self.cct = CCT(
            img_size=img_size,
            embedding_dim=embedding_dim,
            n_input_channels=n_input_channels,
            n_conv_layers=n_conv_layers,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            positional_embedding=positional_embedding,
            **kwargs
        )
    def forward(self, x):
        return self.cct(x)

    

def get_model():
    model = MultiLabelCCT(
        img_size=(192, 668),
        embedding_dim=384,
        n_conv_layers=2,
        kernel_size=7,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        num_layers=14,
        num_heads=6,
        mlp_ratio=3.0,
        num_classes=80,
        positional_embedding='learnable',
    )
    return model