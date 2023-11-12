from typing import Any, Hashable, Tuple

import torch
import voxynth
from monai.transforms import (
    Identityd,
    MapTransform,
    RandomizableTransform,
)
from typeguard import typechecked


@typechecked
class RandomizableIdentityd(Identityd, RandomizableTransform):
    """Nasty hack.

    This class is a "randomizable" transform that simple returns
    its input. Its purpose is to put a user-controlled break
    where the user wishes in a transform pipeline for a
    CacheDataset (the first RandomizableTransform triggers)
    the end of the cached operation.

    """

    pass


@typechecked
class SynthTransformd(RandomizableTransform, MapTransform):
    """Transform to create a synthetic image from a segmentation mask.

    This transform uses the voxynth library to perform the mapping of labels
    to random grey level intensities.

    """

    def __init__(
        self,
        mask_key: str,
        image_output_keys: list[str],
        apply_probability: float = 1.0,
    ):
        """Init.

        Parameters
        ----------
        mask_key: str
            The key that provides the source labelmap that is used to create
            the synth image.
        image_output_keys: list[str]
            Keys where newly created synth images are placed. Can be any number
            of images.
        apply_probability: float, optional
            The probability of applying at all. If the transform is not
            applied, calling the transform has no effect.

        """
        self._mask_key = mask_key
        if len(image_output_keys) < 0:
            raise ValueError("image_output_keys must contain at least one value.")
        self._image_output_keys = image_output_keys
        self._apply_probability = apply_probability

    def __call__(
        self,
        data: dict[Hashable, Any],
    ) -> dict[Hashable, Any]:
        """Create a synthetic image from a labelmap.

        Parameters
        ----------
        data: dict[str, torch.Tensor]
            Mapping containing elements of the dataset. The input mask tensor
            should be found at mask_key and the output image will be placed in
            image_output_key.

        """
        # Randomly skip the entire transform
        if self._apply_probability < 1.0:
            if torch.rand((1,)).item() > self._apply_probability:
                return data

        data_out = data.copy()
        for k in self._image_output_keys:
            data_out[k] = voxynth.synth.labels_to_image(data[self._mask_key])

        return data_out


class VoxynthAugmentd(RandomizableTransform, MapTransform):
    """Transform that uses the voxynth library for intensity augmentations."""

    def __init__(
        self,
        keys: list[str],
        mask_key: str | None,
        inversion_probability: float = 0.0,
        smoothing_probability: float = 0.0,
        smoothing_one_axis_probability: float = 0.5,
        smoothing_max_sigma: float = 2.0,
        bias_field_probability: float = 0.0,
        bias_field_max_magnitude: float = 0.5,
        bias_field_smoothing_range: Tuple[float, float] = (10, 30),
        background_noise_probability: float = 0.0,
        background_blob_probability: float = 0.0,
        added_noise_probability: float = 0.0,
        added_noise_max_sigma: float = 0.05,
        wave_artifact_probability: float = 0.0,
        wave_artifact_max_strength: float = 0.05,
        line_corruption_probability: float = 0.0,
        gamma_scaling_probability: float = 0.0,
        gamma_scaling_max: float = 0.8,
        resized_one_axis_probability: float = 0.5,
    ):
        """Init.

        Parameters
        ----------
        keys: list[str]
            List of keys to which the augmentation should be applied.
        mask_key: str
            Segmentation mask of the image.
        inversion_probability: float, optional
            The probability of inverting the image intensities.
        smoothing_probability: float, optional
            The probability of applying a gaussian smoothing kernel.
        smoothing_one_axis_probability: float, optional
            The probability of applying the smoothing kernel to a single axis.
            This is a sub-probability of the `smoothing_probability`.
        smoothing_max_sigma: float, optional
            The maximum sigma for the smoothing kernel.
        bias_field_probability: float, optional
            The probability of simulating a bias field.
        bias_field_max_magnitude: float, optional
            The maximum strength of the bias field.
        bias_field_smoothing_range: Tuple, optional
            The range of perlin noise wavelengths to generate the bias field.
        background_noise_probability: float, optional
            The probability of synthesizing perline noise in the background.
            Otherwise, the background will be set to zero.
        background_blob_probability: float, optional
            The probability of adding random blobs of noise to the background.
        background_roll_probability: float, optional
            The probability of rolling the image around the background.
        background_roll_max_strength: float, optional
            The maximum scale for rolling the image around the background.
        added_noise_probability: float, optional
            The probability of adding random Gaussian noise across the entire
            image.
        added_noise_max_sigma: float, optional
            The maximum sigma for the added Gaussian noise.
        wave_artifact_probability: float, optional
            The probability of adding wave artifacts or grating effects to the
            image.
        wave_artifact_max_strength: float, optional
            The maximum strength (intensity) of added wave artifacts.
        line_corruption_probability: float, optional
            The probability of adding random line artifacts to the image, i.e.
            blocking out signal in a random slice.
        gamma_scaling_probability: float, optional
            The probability of scaling the image intensities with a gamma
            function.
        gamma_scaling_max: float, optional
            The maximum value for the gamma exponentiation.
        resized_probability: float, optional
            The probability of downsampling, then re-upsampling the image to
            synthesize low-resolution image resizing.
        resized_one_axis_probability: float, optional
            The probability of resizing only one axis, to simulate thick-slice
            data.
        resized_max_voxsize: float, optional
            The maximum voxel size for the 'resized' downsampling step.

        """
        self.keys = tuple(keys)
        self._mask_key = mask_key
        self._inversion_probability = inversion_probability
        self._smoothing_probability = smoothing_probability
        self._smoothing_one_axis_probability = smoothing_one_axis_probability
        self._smoothing_max_sigma = smoothing_max_sigma
        self._bias_field_probability = bias_field_probability
        self._bias_field_max_magnitude = bias_field_max_magnitude
        self._bias_field_smoothing_range = bias_field_smoothing_range
        self._background_noise_probability = background_noise_probability
        self._background_blob_probability = background_blob_probability
        self._added_noise_probability = added_noise_probability
        self._added_noise_max_sigma = added_noise_max_sigma
        self._wave_artifact_probability = wave_artifact_probability
        self._wave_artifact_max_strength = wave_artifact_max_strength
        self._line_corruption_probability = line_corruption_probability
        self._gamma_scaling_probability = gamma_scaling_probability
        self._gamma_scaling_max = gamma_scaling_max
        self._resized_one_axis_probability = resized_one_axis_probability

    def __call__(
        self,
        data: dict[Hashable, Any],
    ) -> dict[Hashable, Any]:
        """Apply image augmentation.

        Parameters
        ----------
        data: dict[str, torch.Tensor]
            Mapping containing elements of the dataset.

        """
        data_out = data.copy()

        for k in self.keys:
            mask = data[self._mask_key][0] if self._mask_key is not None else None
            data_out[k] = voxynth.augment.image_augment(
                data[k],
                mask,
                inversion_probability=self._inversion_probability,
                smoothing_probability=self._smoothing_probability,
                smoothing_one_axis_probability=self._smoothing_one_axis_probability,  # noqa: E501
                smoothing_max_sigma=self._smoothing_max_sigma,
                bias_field_probability=self._bias_field_probability,
                bias_field_max_magnitude=self._bias_field_max_magnitude,
                bias_field_smoothing_range=self._bias_field_smoothing_range,
                background_noise_probability=self._background_noise_probability,  # noqa: E501
                background_blob_probability=self._background_blob_probability,
                added_noise_probability=self._added_noise_probability,
                added_noise_max_sigma=self._added_noise_max_sigma,
                wave_artifact_probability=self._wave_artifact_probability,
                wave_artifact_max_strength=self._wave_artifact_max_strength,
                line_corruption_probability=self._line_corruption_probability,
                gamma_scaling_probability=self._gamma_scaling_probability,
                gamma_scaling_max=self._gamma_scaling_max,
                resized_one_axis_probability=self._resized_one_axis_probability,  # noqa: E501
                normalize=True,
            )

        return data_out
