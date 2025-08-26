# Image Captioning Project

## Overview

This project implements an advanced image captioning system that automatically generates descriptive captions for images using deep learning techniques. The model combines computer vision and natural language processing to understand visual content and produce human-like descriptions.

## Features

- **Automatic Caption Generation**: Generate descriptive captions for any input image
- **Multiple Model Architectures**: Support for CNN-RNN, Vision Transformers, and attention-based models
- **Pre-trained Models**: Leverage state-of-the-art pre-trained models for better performance
- **Interactive Interface**: User-friendly interface for testing and demonstration
- **Batch Processing**: Process multiple images efficiently
- **Custom Training**: Train models on custom datasets

## Architecture

The system uses an encoder-decoder architecture:

- **Encoder**: Convolutional Neural Network (CNN) or Vision Transformer (ViT) to extract image features
- **Decoder**: Recurrent Neural Network (RNN) with attention mechanism to generate captions
- **Attention Mechanism**: Allows the model to focus on relevant parts of the image while generating each word

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Muhammadcodearif/IMAGE_CAPTIONING.git
cd IMAGE_CAPTIONING
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

```txt
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
Pillow>=8.3.0
matplotlib>=3.4.0
nltk>=3.6.0
transformers>=4.10.0
streamlit>=1.0.0  # For web interface
opencv-python>=4.5.0
tqdm>=4.62.0
```

## Usage

### Quick Start

1. **Generate captions for a single image**:
```python
from image_captioning import ImageCaptioner

# Initialize the model
captioner = ImageCaptioner()

# Generate caption
caption = captioner.predict('path/to/your/image.jpg')
print(f"Caption: {caption}")
```

2. **Using the web interface**:
```bash
streamlit run app.py
```

3. **Batch processing**:
```python
# Process multiple images
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
captions = captioner.predict_batch(images)
```

### Advanced Usage

#### Training a Custom Model

```python
from trainer import ImageCaptionTrainer

# Initialize trainer
trainer = ImageCaptionTrainer(
    model_type='cnn_rnn',  # or 'vit_gpt2'
    dataset_path='path/to/dataset',
    batch_size=32,
    learning_rate=1e-4
)

# Train the model
trainer.train(epochs=50)
```

#### Using Different Model Architectures

```python
# CNN-RNN with attention
model = ImageCaptioner(model_type='cnn_rnn_attention')

# Vision Transformer with GPT-2
model = ImageCaptioner(model_type='vit_gpt2')

# CLIP-based model
model = ImageCaptioner(model_type='clip_gpt2')
```

## Dataset

The project supports training on various datasets:

- **COCO Dataset**: Microsoft Common Objects in Context
- **Flickr8k**: 8,000 images with captions
- **Flickr30k**: 30,000 images with captions
- **Custom Datasets**: Support for custom image-caption pairs

### Dataset Structure

```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── captions/
│   └── captions.json
```

### Caption Format

```json
{
    "image1.jpg": [
        "A person riding a bicycle on the street",
        "Someone cycling down the road",
        "A cyclist on a bike outdoors"
    ]
}
```

## Model Performance

| Model Architecture | BLEU-1 | BLEU-4 | METEOR | CIDEr | ROUGE-L |
|-------------------|--------|--------|---------|-------|---------|
| CNN-RNN           | 0.72   | 0.28   | 0.24    | 0.98  | 0.54    |
| CNN-RNN-Attention | 0.75   | 0.32   | 0.27    | 1.12  | 0.57    |
| ViT-GPT2          | 0.78   | 0.35   | 0.29    | 1.25  | 0.61    |

## Examples

### Input Image → Generated Caption

![Example 1](examples/example1.jpg)
**Generated Caption**: "A golden retriever playing fetch in a sunny park with green grass"

![Example 2](examples/example2.jpg)
**Generated Caption**: "A busy city street at night with bright neon signs and cars"

## Configuration

The model behavior can be customized through `config.yaml`:

```yaml
model:
  architecture: "vit_gpt2"  # cnn_rnn, cnn_rnn_attention, vit_gpt2
  max_length: 50
  beam_size: 3
  temperature: 0.8

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 50
  optimizer: "adam"
  scheduler: "cosine"

data:
  dataset_name: "coco"
  image_size: 224
  vocab_size: 10000
```

## API Reference

### ImageCaptioner Class

```python
class ImageCaptioner:
    def __init__(self, model_path=None, device='auto'):
        """Initialize the image captioner"""
    
    def predict(self, image_path, beam_size=3):
        """Generate caption for single image"""
    
    def predict_batch(self, image_paths, batch_size=32):
        """Generate captions for multiple images"""
    
    def evaluate(self, test_dataset):
        """Evaluate model on test dataset"""
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Troubleshooting

### Common Issues

**CUDA out of memory error**:
```bash
# Reduce batch size or use CPU
python train.py --batch_size 16 --device cpu
```

**Model loading errors**:
- Ensure all dependencies are installed
- Check model file paths and permissions
- Verify CUDA compatibility

**Poor caption quality**:
- Try different model architectures
- Increase training epochs
- Use larger datasets
- Adjust hyperparameters

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Microsoft COCO Dataset](https://cocodataset.org/)
- [PyTorch Team](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [OpenAI CLIP](https://openai.com/blog/clip/)

## Citations

If you use this project in your research, please cite:

```bibtex
@software{image_captioning_2024,
  title={Image Captioning Project},
  author={Muhammad Arif},
  year={2024},
  url={https://github.com/Muhammadcodearif/IMAGE_CAPTIONING}
}
```

## Contact

- **Author**: Muhammad Arif
- **GitHub**: [@Muhammadcodearif](https://github.com/Muhammadcodearif)
- **Email**: [Contact via GitHub](https://github.com/Muhammadcodearif)

## Future Improvements

- [ ] Support for video captioning
- [ ] Multi-language caption generation
- [ ] Real-time webcam captioning
- [ ] Mobile app integration
- [ ] Docker containerization
- [ ] Cloud deployment options
- [ ] Advanced attention visualization

---

**Note**: This README provides a comprehensive template. Please update the specific details, file paths, performance metrics, and examples based on your actual implementation.
