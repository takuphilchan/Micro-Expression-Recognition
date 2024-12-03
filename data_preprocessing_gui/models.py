
import cv2
import torch
import numpy as np
import torch.nn as nn
import mediapipe as mp
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, r3d_18, R3D_18_Weights

class R2Plus1DFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, dropout_prob=0.0):
        super(R2Plus1DFeatureExtractor, self).__init__()

        self.pretrained = pretrained
        self.dropout_prob = dropout_prob if dropout_prob is not None else 0.0
        # Initialize the model with pretrained weights if specified
        self.initialize_model()

        # Add Spatial Attention Module
        self.spatial_attention = SpatialAttention()

    def initialize_model(self):
        if self.pretrained:
            self.pretrained_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        else:
            self.pretrained_model = r2plus1d_18(weights=None)  # Load model without pretrained weights

        # Extract layers from the model
        self.stem = self.pretrained_model.stem
        self.max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.layer1 = nn.Sequential(*self.pretrained_model.layer1)
        self.layer2 = nn.Sequential(*self.pretrained_model.layer2)
        self.layer3 = nn.Sequential(*self.pretrained_model.layer3)
        self.layer4 = nn.Sequential(*self.pretrained_model.layer4)
        self.pretrained_model.fc = nn.Identity()  # Remove the FC layer

         # Conditionally add dropout layers
        self.dropout = nn.Dropout3d(self.dropout_prob) if self.dropout_prob > 0 else nn.Identity()

    def forward(self, x, spatial_masks=None, padding_mask=None, use_spatial_attention=None):
        x = x.permute(0, 2, 1, 3, 4)
        # Apply the initial layers
        x = self.stem(x)
        x = self.max_pool(x)
        features1 = x.clone()
        # Apply the spatial mask if specified
        if spatial_masks is not None:
            output_size = (x.shape[-3], x.shape[-2], x.shape[-1])
            spatial_masks = F.interpolate(spatial_masks.unsqueeze(1), size=output_size, mode='trilinear', align_corners=True)
            spatial_masks = spatial_masks.expand(-1, x.size(1), -1, -1, -1)
            x = x * spatial_masks
        features2 = x.clone()
        # Apply padding mask if provided
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).float()
            x = x * padding_mask

        # Apply subsequent layers
        x = self.layer1(x)
        x = self.dropout(x)  # Apply dropout after layer1

        x = self.layer2(x)
        x = self.dropout(x)  # Apply dropout after layer2

        x = self.layer3(x)
        x = self.dropout(x)  # Apply dropout after layer3
        features3 = x.clone()

        if use_spatial_attention is True:
            # Apply Spatial Attention after layer3
            x = self.spatial_attention(x)

        x = self.layer4(x)
        x = self.dropout(x)  # Apply dropout after layer4
        features4 = x.clone()
        # Save the features after applying the spatial mask
        return x, features1, features2, features3, features4

    def reload_weights(self):
        # Reinitialize the model weights
        self.initialize_model()

class R3DFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, dropout_prob=0.0):
        super(R3DFeatureExtractor, self).__init__()

        self.pretrained = pretrained
        self.dropout_prob = dropout_prob if dropout_prob is not None else 0.0

        # Initialize the model with pretrained weights if specified
        self.initialize_model()

        # Add Spatial Attention Module
        self.spatial_attention = SpatialAttention()

    def initialize_model(self):
        if self.pretrained:
            self.pretrained_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        else:
            self.pretrained_model = r3d_18(weights=None)  # Load model without pretrained weights

        # Extract layers from the model
        self.stem = self.pretrained_model.stem
        self.max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.layer1 = nn.Sequential(*self.pretrained_model.layer1)
        self.layer2 = nn.Sequential(*self.pretrained_model.layer2)
        self.layer3 = nn.Sequential(*self.pretrained_model.layer3)
        self.layer4 = nn.Sequential(*self.pretrained_model.layer4)
        self.pretrained_model.fc = nn.Identity()  # Remove the FC layer

        # Conditionally add dropout layers
        self.dropout = nn.Dropout3d(self.dropout_prob) if self.dropout_prob > 0 else nn.Identity()

    def forward(self, x, spatial_masks=None, padding_mask=None, use_spatial_attention=None):
        x = x.permute(0, 2, 1, 3, 4)
        # Apply the initial layers
        x = self.stem(x)
        x = self.max_pool(x)
        features1 = x.clone()
        # Apply the spatial mask if specified
        if spatial_masks is not None:
            output_size = (x.shape[-3], x.shape[-2], x.shape[-1])
            spatial_masks = F.interpolate(spatial_masks.unsqueeze(1), size=output_size, mode='trilinear', align_corners=True)
            spatial_masks = spatial_masks.expand(-1, x.size(1), -1, -1, -1)
            x = x * spatial_masks
        features2 = x.clone()
        # Apply padding mask if provided
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).float()
            x = x * padding_mask

        # Apply subsequent layers
        x = self.layer1(x)
        x = self.dropout(x)  # Apply dropout after layer1

        x = self.layer2(x)
        x = self.dropout(x)  # Apply dropout after layer2

        x = self.layer3(x)
        x = self.dropout(x)  # Apply dropout after layer3
        features3 = x.clone()

        if use_spatial_attention is True:
            # Apply Spatial Attention after layer3
            x = self.spatial_attention(x)

        x = self.layer4(x)
        x = self.dropout(x)  # Apply dropout after layer4
        features4 = x.clone()
        # Save the features after applying the spatial mask
        return x, features1, features2, features3, features4

    def reload_weights(self):
        # Reinitialize the model weights
        self.initialize_model()

# Define the Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Ensure input is 5D (batch_size, channels, depth, height, width)
        if len(x.shape) != 5:
            raise ValueError(f"Expected input to be 5D (batch_size, channels, depth, height, width), but got shape {x.shape}")

        # Squeeze-and-Excitation block
        batch_size, channels, depth, height, width = x.shape

        avg_pool = F.adaptive_avg_pool3d(x, 1).view(batch_size, channels)
        max_pool = F.adaptive_max_pool3d(x, 1).view(batch_size, channels)

        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_out = self.fc2(F.relu(self.fc1(max_pool)))
        out = avg_out + max_out
        attention_weights = self.sigmoid(out).view(batch_size, channels, 1, 1, 1)
        attended_features = x * attention_weights

        return attended_features, attention_weights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Convolutional layer to generate the attention map
        self.conv1 = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute average and max-pooling along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average pooling across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling across channels

        # Concatenate avg_out and max_out along the channel dimension
        out = torch.cat([avg_out, max_out], dim=1)  # Shape: (batch_size, 2, depth, height, width)

        # Apply convolution and sigmoid activation to generate the attention map
        out = self.conv1(out)
        attention_map = self.sigmoid(out)  # Shape: (batch_size, 1, depth, height, width)

        # Apply attention map to the original feature map
        return x * attention_map  # Shape: (batch_size, channels, depth, height, width)


class MicroExpressionClassifier(nn.Module):
    def __init__(self, d_model=512, num_classes=3, dropout_prob=0.0, pretrained=True, use_spatial_attention=None, use_spatial_masks=None, use_channel_attention=False, feature_extractor_cls=None):
        super(MicroExpressionClassifier, self).__init__()

        # Feature Extractor
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        self.feature_extractor = feature_extractor_cls(pretrained=pretrained, dropout_prob=dropout_prob)
        self.use_channel_attention = use_channel_attention
        self.use_spatial_masks = use_spatial_masks
        self.use_spatial_attention = use_spatial_attention

        # Classification layer
        self.classifier = nn.Linear(d_model, num_classes)

        if self.use_channel_attention:
            self.channel_attention = ChannelAttention(in_channels=512)

    def forward(self, x, padding_mask=None):
        if self.use_spatial_masks:
            # Extract facial regions and generate spatial masks
            facial_regions = self._extract_facial_regions(x)
            spatial_masks = self._generate_spatial_masks(x, facial_regions)
            features, features1,features2,features3,features4 = self.feature_extractor(x, spatial_masks, padding_mask, self.use_spatial_attention)
            # features1,features2,features3,features4 
            # self.plot_feature_maps(features1, "Features before mask")
            # self.plot_feature_maps(features2, "Features After mask")
            # self.plot_feature_maps(features3, "Features After Layer 3")
            # self.plot_feature_maps(features4, "Features After Layer 4")
        else:
            features, features1,features2,features3,features4 = self.feature_extractor(x, None, padding_mask)

        # Global Average Pooling across spatial dimensions
        features = F.adaptive_avg_pool3d(features, (1, 1, 1))

        # Apply Channel Attention if enabled
        if self.use_channel_attention:
            features, attn_weights = self.channel_attention(features)

        # Flatten before classification
        features = features.view(features.size(0), -1)
        outputs = self.classifier(features)

        return outputs


    def _extract_facial_regions(self, x, padding_mask=None):
        """
        Extracts head outline points for each frame in the batch using Mediapipe's face mesh.
        """
        facial_regions_list = []
        batch_size, num_frames, channels, height, width = x.size()

        # Downscale the frames for faster processing
        downscale_factor = 0.5  # Adjust the factor as needed
        down_height, down_width = int(height * downscale_factor), int(width * downscale_factor)
        downscaled_frames = F.interpolate(x.view(-1, channels, height, width), size=(down_height, down_width), mode='bilinear')

        head_outline_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
                                400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
                                54, 103, 67, 109]

        for i in range(batch_size):
            facial_regions_batch = []
            for j in range(num_frames):
                if padding_mask is not None and padding_mask[i, j] == 0:
                    # Skip padded frames
                    facial_regions_batch.append([])
                    continue

                frame = downscaled_frames[i * num_frames + j].permute(1, 2, 0).cpu().numpy()  # Minimal detach
                frame = (frame * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results_mesh = self.face_mesh.process(rgb_frame)

                if results_mesh.multi_face_landmarks:
                    landmarks = results_mesh.multi_face_landmarks[0]
                    points = []

                    # Extract head outline points
                    for idx in head_outline_indices:
                        x_coord = int(landmarks.landmark[idx].x * down_width)
                        y_coord = int(landmarks.landmark[idx].y * down_height)
                        points.append((x_coord, y_coord))

                    if points:
                        # Scale back the points to the original resolution
                        points = [(int(x / downscale_factor), int(y / downscale_factor)) for x, y in points]
                        facial_regions_batch.append(points)
                    else:
                        facial_regions_batch.append([])
                else:
                    facial_regions_batch.append([])

            facial_regions_list.append(facial_regions_batch)

        return facial_regions_list

    def _generate_spatial_masks(self, x, facial_regions, padding_mask=None):
        """
        Generates a spatial mask for each frame based on the detected head outline, 
        and fills missing frames with the average mask.
        """
        batch_size, num_frames, channels, height, width = x.size()
        combined_spatial_mask = torch.ones((batch_size, num_frames, height, width), device=x.device)  # Initialize with ones

        shrink_factor = 0.93  # Adjust how much the mask moves inward

        # To store masks for calculating the average
        valid_masks = []
        
        for i, tensor in enumerate(facial_regions):
            batch_spatial_mask = torch.zeros((num_frames, height, width), device=x.device)  # Initialize with zeros
            for j, regions in enumerate(tensor):
                if padding_mask is not None and padding_mask[i, j] == 0:
                    # Skip padded frames
                    continue

                if regions:
                    # Process each frame and generate the mask
                    mask = np.zeros((height, width), dtype=np.uint8)

                    points = np.array(regions, np.int32)

                    # Calculate the center of the head outline
                    center_x = np.mean(points[:, 0])
                    center_y = np.mean(points[:, 1])

                    # Shrink the points towards the center based on the shrink_factor
                    for k in range(len(points)):
                        points[k][0] = int(center_x + shrink_factor * (points[k][0] - center_x))
                        points[k][1] = int(center_y + shrink_factor * (points[k][1] - center_y))

                    # Create the head outline mask using fillPoly
                    cv2.fillPoly(mask, [points], 255)  # Fill the region inside the head outline

                    # Convert mask to torch tensor and normalize to [0, 1]
                    individual_mask = torch.tensor(mask, dtype=torch.float32, device=x.device) / 255.0
                    batch_spatial_mask[j] = individual_mask
                    
                    # Add valid mask to list
                    valid_masks.append(individual_mask)
                else:
                    batch_spatial_mask[j] = torch.zeros((height, width), device=x.device)  # Placeholder for missing mask

            combined_spatial_mask[i] = batch_spatial_mask

        # Compute average mask across valid masks
        if valid_masks:
            avg_mask = torch.stack(valid_masks).mean(dim=0)
        else:
            avg_mask = torch.ones((height, width), device=x.device)  # Fallback in case no valid masks at all

        # Apply average mask to frames that were skipped
        for i in range(batch_size):
            for j in range(num_frames):
                if combined_spatial_mask[i, j].sum() == 0:  # Check if mask is zero (empty)
                    combined_spatial_mask[i, j] = avg_mask  # Assign average mask

        return combined_spatial_mask

    def plot_pooled_features(self, features_after_mask, title):
        # Assuming features_after_mask is of shape (batch_size, channels, 1, 1, 1)
        batch_size, channels, depth, height, width = features_after_mask.shape
        
        # Check that the shape matches what is expected after adaptive_avg_pool3d
        assert depth == 1 and height == 1 and width == 1, "The input features are not of the shape (batch_size, channels, 1, 1, 1)"
        
        # Extract the values for the first sample in the batch
        pooled_values = features_after_mask[0, :, 0, 0, 0].cpu().detach().numpy()
        
        # Plot the pooled feature values
        plt.figure(figsize=(10, 5))
        plt.bar(range(channels), pooled_values)
        plt.xlabel("Channel")
        plt.ylabel("Pooled Value")
        plt.title(title)
        plt.show()
    
    def plot_feature_maps(self, feature_maps, title, channel=1, figsize=(15, 5), cmap='viridis', save_path=None):
        """
        Plots feature maps for a specified channel from a batch of feature maps.

        Parameters:
        - feature_maps (torch.Tensor): Tensor of shape (batch_size, channels, depth, height, width).
        - title (str): Title of the plot.
        - channel (int): Channel index to plot. Default is 1.
        - figsize (tuple): Size of the figure. Default is (15, 5).
        - cmap (str): Colormap to use for displaying the images. Default is 'viridis'.
        - save_path (str or None): If provided, saves the plot to the specified path. If None, the plot is not saved.

        Raises:
        - ValueError: If the channel index is out of range.
        """
        if not isinstance(feature_maps, torch.Tensor):
            raise TypeError("feature_maps should be a torch.Tensor. Got type: {}".format(type(feature_maps)))
        
        # Validate input dimensions
        if feature_maps.dim() != 5:
            raise ValueError("Expected a 5D tensor of shape (batch_size, channels, depth, height, width). Got: {}".format(feature_maps.shape))
        
        batch_size, channels, depth, height, width = feature_maps.shape
        if channel >= channels or channel < 0:
            raise ValueError(f"Channel index {channel} out of range. The feature map has {channels} channels.")
        
        # Create subplots
        fig, axarr = plt.subplots(1, depth, figsize=figsize)
        
        # Handle case where there is only one subplot
        if depth == 1:
            axarr = [axarr]  # Make it iterable
        
        # Normalize feature maps for better visibility
        feature_maps_np = feature_maps[0, channel].cpu().detach().numpy()
        feature_maps_np = (feature_maps_np - np.min(feature_maps_np)) / (np.max(feature_maps_np) - np.min(feature_maps_np))
        
        # Plot each depth slice
        for i in range(depth):
            axarr[i].imshow(feature_maps_np[i, :, :], cmap=cmap)
            axarr[i].axis('off')
        
        # Set the title
        fig.suptitle(f"{title} - Channel {channel}", fontsize=16)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()

    def plot_channel_attention(self, attention_weights):
        # Convert to NumPy array and ensure it's 1-dimensional
        attention_weights_np = attention_weights.detach().cpu().numpy()
        if attention_weights_np.ndim > 1:
            attention_weights_np = attention_weights_np.flatten()

        # Plot attention weights
        plt.figure(figsize=(10, 6))
        plt.bar(range(attention_weights_np.size), attention_weights_np, color='blue')
        plt.xlabel('Channel')
        plt.ylabel('Attention Weight')
        plt.title('Channel Attention Weights')
        plt.show()


    def get_attention_weights(self):
        # Extract attention weights from the last transformer layer
        if self.transformer_module:
            last_transformer_layer = self.transformer_module.transformer[-1]
            return last_transformer_layer.attn_weights
        else:
            raise RuntimeError("No Transformer module is present.")

    def visualize_feature_maps(self, feature_maps):
        # Convert feature maps to numpy and plot
        feature_maps = feature_maps.cpu().detach().numpy()
        batch_size, channels, depth, height, width = feature_maps.shape
        
        # Plot feature maps for the first sample in the batch
        for i in range(channels):
            plt.figure(figsize=(10, 8))
            for d in range(depth):
                plt.subplot(1, depth, d + 1)
                plt.imshow(feature_maps[0, i, d, :, :], cmap='viridis')
                plt.axis('off')
            plt.suptitle(f'Channel {i}')
            plt.show()
    
    
    def visualize_attention_weights(self, attention_weights, sequence_range=0):
        print(f"Attention Weights shape: {attention_weights.shape}")
        # Check the shape of the attention_weights
        for sequence_index in range(sequence_range):
            if len(attention_weights.shape) == 3:
                # If the shape is [num_sequences, sequence_length, sequence_length]
                
                attention_matrix = attention_weights[sequence_index, :, :]

                print(f"Attention Matrix shape: {attention_matrix.shape}")
                
                plt.imshow(attention_matrix, cmap='viridis')
                plt.title(f'Attention Weights for Batch {sequence_index}')
                plt.colorbar()
                plt.show()
            else:
                raise ValueError("Unsupported attention_weights shape")