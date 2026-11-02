# Real photographs for visual/depth regression

These six camera photographs supplement the three generated backyard fixtures.
They exercise architecture and vegetation, a furnished interior, curved tabletop
objects, a portrait, fine animal fur/whiskers, and a distant outdoor launch scene
with sky, smoke and thin structures. See `manifest.json` for exact sources,
authors, licenses, original/download hashes and preprocessing. No customer photos
or private property uploads are included.

The two architecture photographs are CC0 works from Wikimedia Commons. Coffee
(Rachel Michetti) and Chelsea the cat (Stefan van der Walt) are CC0 photographs
provided by scikit-image. The astronaut portrait is a NASA public-domain image;
the launch image is a SpaceX public-domain photograph, as documented by
[scikit-image](https://scikit-image.org/docs/0.25.x/api/skimage.data.html).
Photo licensing is independent of this repository's code license.

The repository copies remove EXIF metadata, normalize orientation and RGB, cap
large photos at 1600 pixels on their longest edge, and use JPEG quality 95.
`prepare_real_photos.py` verifies the fixture hashes and applies the same fixed
center-crop/bicubic preprocessing as the existing Fast and Detail profiles.
Small source photos are intentionally retained at their original resolution;
the Detail input upsamples some of them. This is a useful low-resolution input
stress test, not evidence of recovered photographic detail.

No measured scene depth is available. Passing CUDA agreement establishes port
fidelity, not physical measurement accuracy or correct underlying model geometry.
