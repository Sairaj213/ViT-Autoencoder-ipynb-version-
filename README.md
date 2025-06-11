# 📷 Transformer Autoencoder for Image Reconstruction `(Jupyter Notebook Version)`
<br>

This is an Autoencoder architecture built with Transformer and Keras, for image reconstruction on the CIFAR-10 dataset. The architecture uses a Vision Transformer (ViT) style of patching images, encoding them with Transformer blocks, and then decoding them back into images.

<br>

---

## 🧠 Core Project Flow

1. **Image Preprocessing**: Images are divided into fixed-size patches and embedded.
2. **ViT Encoder**: Transformer layers encode the patches into latent representations.
3. **Decoder**: Reconstructs the original image from encoded latent vectors.
4. **Training**: Uses MSE loss for reconstruction accuracy.
5. **Evaluation**: Visual inspection of reconstructions and quantitative loss.

<br>   

---

## 🔧 Key Functions Explained

### 🧱 Patch Utilities

- `patch_to_image(patches, patch_size, img_size, channels)`
  - Reassembles image patches into full images.
  - Useful for visualizing outputs of the decoder.

### 🔄 Transformer Components

- `transformer_block(x, num_heads, ff_dim, dropout_rate)`
  - Core building block of the ViT encoder.
  - Applies multi-head self-attention followed by feedforward layers with residual connections.

### 🏗️ Model Construction

- `build_autoencoder(image_size, patch_size, projection_dim, num_transformer_layers, num_heads, ff_dim, channels)`
  - Constructs the full autoencoder pipeline:
    - Embedding layer
    - Multiple transformer blocks
    - Dense decoder
  - Returns a compiled `tf.keras.Model`.

### 📊 Evaluation & Visualization

- `display_reconstructions(model, data, n)`
  - Displays original vs. reconstructed images side-by-side.
  - Helpful for visual diagnostics.

- `evaluate_and_reconstruct_image(image_input, model, img_res)`
  - Takes a single image and outputs the reconstructed version from the trained model.

- `get_model_input_shape(model)`
  - Retrieves the expected input shape from the Keras model.

---

## 🧩 Custom Layers (ViT Components)

- `PatchEmbedding(patch_size)`
  - Custom Keras layer that reshapes the input image into non-overlapping patches.

- `PatchEncoder(num_patches, projection_dim)`
  - Embeds patches into a latent space and adds positional encoding.

