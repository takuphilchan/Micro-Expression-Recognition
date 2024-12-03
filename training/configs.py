from models import MicroExpressionClassifier, R3DFeatureExtractor, R2Plus1DFeatureExtractor

def ablation_configurations():
    ablation_tests = {
        # R3D Configurations
        # Dropout 0.2 and Spatial Attention, No Channel Attention, Spatial Masks Enabled
        'R3D_Dropout0.2_SpatialAttention_SpatialMasks': {
            'model': MicroExpressionClassifier(
                d_model=512,
                num_classes=3,
                dropout_prob=0.2,
                use_spatial_attention=True,
                pretrained=True,
                use_spatial_masks=True,
                use_channel_attention=False,
                feature_extractor_cls=R3DFeatureExtractor
            ),
            'description': 'R3D feature extractor with dropout rate 0.2, Spatial Attention enabled, Channel Attention disabled, and spatial masks enabled.'
        },
        'R3D_Dropout0.3_SpatialAttention_SpatialMasks': {
            'model': MicroExpressionClassifier(
                d_model=512,
                num_classes=3,
                dropout_prob=0.3,
                use_spatial_attention=True,
                pretrained=True,
                use_spatial_masks=True,
                use_channel_attention=False,
                feature_extractor_cls=R3DFeatureExtractor
            ),
            'description': 'R3D feature extractor with dropout rate 0.3, Spatial Attention enabled, Channel Attention disabled, and spatial masks enabled.'
        },

        # Channel Attention Configurations (Dropout 0.3, No Spatial Attention, Channel Attention Enabled, Spatial Masks Enabled)
        'R3D_Dropout0.3_NoSpatialAttention_ChanAttn_SpatialMasks': {
            'model': MicroExpressionClassifier(
                d_model=512,
                num_classes=3,
                dropout_prob=0.3,
                use_spatial_attention=False,
                pretrained=True,
                use_spatial_masks=True,
                use_channel_attention=True,
                feature_extractor_cls=R3DFeatureExtractor
            ),
            'description': 'R3D feature extractor with dropout rate 0.3, Spatial Attention disabled, Channel Attention enabled, and spatial masks enabled.'
        },

        # Baseline Configurations (No Dropout, No Spatial Attention, No Channel Attention, No Spatial Masks)
        'R3D_NewBaseline_NoSpatialAttention_NoChanAttn_NoSpatialMasks': {
            'model': MicroExpressionClassifier(
                d_model=512,
                num_classes=3,
                dropout_prob=None,
                use_spatial_attention=False,
                pretrained=True,
                use_spatial_masks=False,
                use_channel_attention=False,
                feature_extractor_cls=R3DFeatureExtractor
            ),
            'description': 'Baseline R3D feature extractor with no dropout, Spatial Attention disabled, Channel Attention disabled, and spatial masks disabled.'
        },

        # R2Plus1D Configurations
        # Dropout 0.2 and Spatial Attention, No Channel Attention, Spatial Masks Enabled
        'R2Plus1D_Dropout0.2_SpatialAttention_SpatialMasks': {
            'model': MicroExpressionClassifier(
                d_model=512,
                num_classes=3,
                dropout_prob=0.2,
                use_spatial_attention=True,
                pretrained=True,
                use_spatial_masks=True,
                use_channel_attention=False,
                feature_extractor_cls=R2Plus1DFeatureExtractor
            ),
            'description': 'R2Plus1D feature extractor with dropout rate 0.2, Spatial Attention enabled, Channel Attention disabled, and spatial masks enabled.'
        },
        'R2Plus1D_Dropout0.3_SpatialAttention_SpatialMasks': {
            'model': MicroExpressionClassifier(
                d_model=512,
                num_classes=3,
                dropout_prob=0.3,
                use_spatial_attention=True,
                pretrained=True,
                use_spatial_masks=True,
                use_channel_attention=False,
                feature_extractor_cls=R2Plus1DFeatureExtractor
            ),
            'description': 'R2Plus1D feature extractor with dropout rate 0.3, Spatial Attention enabled, Channel Attention disabled, and spatial masks enabled.'
        },

        # Channel Attention Configurations (Dropout 0.3, No Spatial Attention, Channel Attention Enabled, Spatial Masks Enabled)
        'R2Plus1D_Dropout0.3_NoSpatialAttention_ChanAttn_SpatialMasks': {
            'model': MicroExpressionClassifier(
                d_model=512,
                num_classes=3,
                dropout_prob=0.3,
                use_spatial_attention=False,
                pretrained=True,
                use_spatial_masks=True,
                use_channel_attention=True,
                feature_extractor_cls=R2Plus1DFeatureExtractor
            ),
            'description': 'R2Plus1D feature extractor with dropout rate 0.3, Spatial Attention disabled, Channel Attention enabled, and spatial masks enabled.'
        },

        # Baseline Configurations (No Dropout, No Spatial Attention, No Channel Attention, No Spatial Masks)
        'R2Plus1D_NewBaseline_NoSpatialAttention_NoChanAttn_NoSpatialMasks': {
            'model': MicroExpressionClassifier(
                d_model=512,
                num_classes=3,
                dropout_prob=None,
                use_spatial_attention=False,
                pretrained=True,
                use_spatial_masks=False,
                use_channel_attention=False,
                feature_extractor_cls=R2Plus1DFeatureExtractor
            ),
            'description': 'Baseline R2Plus1D feature extractor with no dropout, Spatial Attention disabled, Channel Attention disabled, and spatial masks disabled.'
        },

        # R2Plus1D with Dropout 0.2, Spatial Attention and Channel Attention enabled, Spatial Masks enabled
      'R2Plus1D_Dropout0.2_SpatialAttention_ChanAttn_SpatialMasks': {
          'model': MicroExpressionClassifier(
              d_model=512,
              num_classes=3,
              dropout_prob=0.2,
              use_spatial_attention=True,
              pretrained=True,
              use_spatial_masks=True,
              use_channel_attention=True,
              feature_extractor_cls=R2Plus1DFeatureExtractor
          ),
          'description': 'R2Plus1D feature extractor with dropout rate 0.2, Spatial Attention enabled, Channel Attention enabled, and spatial masks enabled.'
      },

      # R3D with Dropout 0.2, Spatial Attention and Channel Attention enabled, Spatial Masks enabled
      'R3D_Dropout0.2_SpatialAttention_ChanAttn_SpatialMasks': {
          'model': MicroExpressionClassifier(
              d_model=512,
              num_classes=3,
              dropout_prob=0.2,
              use_spatial_attention=True,
              pretrained=True,
              use_spatial_masks=True,
              use_channel_attention=True,
              feature_extractor_cls=R3DFeatureExtractor
          ),
          'description': 'R3D feature extractor with dropout rate 0.2, Spatial Attention enabled, Channel Attention enabled, and spatial masks enabled.'
      },
        # R2Plus1D with Dropout 0.2, Spatial Attention and Channel Attention enabled, Spatial Masks enabled
      'R2Plus1D_Dropout0.3_SpatialAttention_ChanAttn_SpatialMasks': {
          'model': MicroExpressionClassifier(
              d_model=512,
              num_classes=3,
              dropout_prob=0.3,
              use_spatial_attention=True,
              pretrained=True,
              use_spatial_masks=True,
              use_channel_attention=True,
              feature_extractor_cls=R2Plus1DFeatureExtractor
          ),
          'description': 'R2Plus1D feature extractor with dropout rate 0.3, Spatial Attention enabled, Channel Attention enabled, and spatial masks enabled.'
      },

      # R3D with Dropout 0.3, Spatial Attention and Channel Attention enabled, Spatial Masks enabled
      'R3D_Dropout0.3_SpatialAttention_ChanAttn_SpatialMasks': {
          'model': MicroExpressionClassifier(
              d_model=512,
              num_classes=3,
              dropout_prob=0.3,
              use_spatial_attention=True,
              pretrained=True,
              use_spatial_masks=True,
              use_channel_attention=True,
              feature_extractor_cls=R3DFeatureExtractor
          ),
          'description': 'R3D feature extractor with dropout rate 0.3, Spatial Attention enabled, Channel Attention enabled, and spatial masks enabled.'
      },
    }
    return ablation_tests
