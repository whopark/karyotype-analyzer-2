"""
Test script for KaryoEnhance module
====================================

This script demonstrates the functionality of KaryoEnhance by:
1. Creating a synthetic noisy chromosome-like image
2. Applying different enhancement modes
3. Displaying quality metrics
4. Saving results for visual inspection

Usage:
    python test_karyoenhance.py
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from karyoenhance import KaryoEnhancePipeline, enhance_karyotype_image
import sys


def create_synthetic_noisy_image(size=(512, 512), noise_level=0.15):
    """
    Create a synthetic chromosome-like image with added noise

    Args:
        size: Image dimensions (width, height)
        noise_level: Amount of noise to add (0.0-1.0)

    Returns:
        noisy_image: Synthetic noisy image as numpy array
    """
    print("Creating synthetic chromosome-like image...")

    # Create base image with chromosome-like structures
    h, w = size
    image = np.ones((h, w, 3), dtype=np.float32) * 0.9  # Light background

    # Add chromosome-like elongated blobs
    np.random.seed(42)
    num_chromosomes = 20

    for i in range(num_chromosomes):
        # Random position
        cx = np.random.randint(w // 4, 3 * w // 4)
        cy = np.random.randint(h // 4, 3 * h // 4)

        # Random size and orientation
        length = np.random.randint(40, 80)
        width_blob = np.random.randint(8, 15)
        angle = np.random.uniform(0, 2 * np.pi)

        # Create chromosome-like structure
        y, x = np.ogrid[:h, :w]
        # Rotate coordinates
        x_rot = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
        y_rot = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)

        # Elongated Gaussian
        mask = np.exp(-((x_rot / length) ** 2 + (y_rot / width_blob) ** 2))

        # Add to image with purple/blue color (like Giemsa staining)
        image[:, :, 0] -= mask * np.random.uniform(0.3, 0.5)  # Less red
        image[:, :, 1] -= mask * np.random.uniform(0.2, 0.4)  # Less green
        image[:, :, 2] -= mask * np.random.uniform(0.1, 0.2)  # More blue

    # Clip to valid range
    image = np.clip(image, 0, 1)

    # Add vignetting (darker edges - common in microscopy)
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    vignette = 1 - 0.3 * (dist / max_dist) ** 2
    image *= vignette[:, :, np.newaxis]

    # Add Gaussian noise
    gaussian_noise = np.random.normal(0, noise_level * 0.1, image.shape)
    noisy_image = image + gaussian_noise

    # Add Poisson noise (photon counting noise)
    # Scale up, add Poisson noise, scale back
    scaled = noisy_image * 255
    poisson_noise = np.random.poisson(scaled) / 255.0
    noisy_image = (noisy_image + poisson_noise) / 2

    # Add salt-and-pepper noise (occasional)
    mask = np.random.random(image.shape) < (noise_level * 0.01)
    noisy_image[mask] = np.random.choice([0, 1])

    # Clip final image
    noisy_image = np.clip(noisy_image, 0, 1)

    return noisy_image


def test_enhancement_modes():
    """Test all enhancement modes and compare results"""
    print("\n" + "=" * 60)
    print("KaryoEnhance Test Suite")
    print("=" * 60)

    # Create synthetic image
    noisy_image = create_synthetic_noisy_image(size=(512, 512), noise_level=0.2)
    print(f"✓ Created synthetic image: {noisy_image.shape}")

    # Convert to uint8 for PIL
    noisy_image_uint8 = (noisy_image * 255).astype(np.uint8)
    noisy_pil = Image.fromarray(noisy_image_uint8)

    # Test different modes
    modes = ['mild', 'default', 'aggressive']
    results = {}

    pipeline = KaryoEnhancePipeline()

    for mode in modes:
        print(f"\nTesting {mode.upper()} mode...")

        if mode == 'mild':
            config = pipeline.get_mild_config()
        elif mode == 'default':
            config = pipeline.get_default_config()
        else:
            config = pipeline.get_aggressive_config()

        enhanced_pil, metrics = pipeline.process_pil_image(noisy_pil, config)
        results[mode] = {
            'image': enhanced_pil,
            'metrics': metrics
        }

        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  Contrast improvement: {metrics['contrast_improvement_pct']:.1f}%")
        print(f"  Sharpness improvement: {metrics['sharpness_improvement_pct']:.1f}%")
        print(f"  Noise reduction: {metrics['noise_reduction_pct']:.1f}%")

    # Save results
    print("\n" + "-" * 60)
    print("Saving results...")

    noisy_pil.save('test_output_original.png')
    print("✓ Saved: test_output_original.png")

    for mode, data in results.items():
        filename = f'test_output_{mode}.png'
        data['image'].save(filename)
        print(f"✓ Saved: {filename}")

    # Create comparison plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('KaryoEnhance Comparison', fontsize=16)

        axes[0, 0].imshow(noisy_pil)
        axes[0, 0].set_title('Original (Noisy)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(results['mild']['image'])
        axes[0, 1].set_title(f"Mild Mode\n"
                            f"Contrast: +{results['mild']['metrics']['contrast_improvement_pct']:.1f}%")
        axes[0, 1].axis('off')

        axes[1, 0].imshow(results['default']['image'])
        axes[1, 0].set_title(f"Default Mode\n"
                            f"Contrast: +{results['default']['metrics']['contrast_improvement_pct']:.1f}%")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(results['aggressive']['image'])
        axes[1, 1].set_title(f"Aggressive Mode\n"
                            f"Contrast: +{results['aggressive']['metrics']['contrast_improvement_pct']:.1f}%")
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig('test_output_comparison.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: test_output_comparison.png")
    except Exception as e:
        print(f"⚠ Could not create comparison plot: {e}")

    # Print summary table
    print("\n" + "-" * 60)
    print("Summary Table:")
    print("-" * 60)
    print(f"{'Mode':<12} {'Contrast':<12} {'Sharpness':<12} {'Noise Red.':<12}")
    print("-" * 60)
    for mode in modes:
        metrics = results[mode]['metrics']
        print(f"{mode.capitalize():<12} "
              f"{metrics['contrast_improvement_pct']:>6.1f}%     "
              f"{metrics['sharpness_improvement_pct']:>6.1f}%     "
              f"{metrics['noise_reduction_pct']:>6.1f}%")
    print("-" * 60)

    print("\n✅ All tests completed successfully!")
    print("\nOutput files:")
    print("  - test_output_original.png")
    print("  - test_output_mild.png")
    print("  - test_output_default.png")
    print("  - test_output_aggressive.png")
    print("  - test_output_comparison.png (if matplotlib available)")


def test_simple_api():
    """Test the simple convenience function"""
    print("\n" + "=" * 60)
    print("Testing Simple API (enhance_karyotype_image)")
    print("=" * 60)

    # Create test image
    noisy_image = create_synthetic_noisy_image(size=(256, 256), noise_level=0.15)

    # Enhance using simple API
    enhanced, metrics = enhance_karyotype_image(noisy_image, mode='default')

    print(f"\n✓ Enhanced image shape: {enhanced.shape}")
    print(f"✓ Contrast improvement: {metrics['contrast_improvement_pct']:.1f}%")
    print(f"✓ Noise reduction: {metrics['noise_reduction_pct']:.1f}%")

    # Save
    enhanced_pil = Image.fromarray(enhanced)
    enhanced_pil.save('test_simple_api.png')
    print("✓ Saved: test_simple_api.png")


def main():
    """Run all tests"""
    try:
        # Test enhancement modes
        test_enhancement_modes()

        # Test simple API
        test_simple_api()

        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n❌ Error during testing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
