number_samples = 300 # Number of samples
number_pixels = 5 # Number of pixels per sample

# True reflectance
reflectance = np.random.exponential(1, size=(number_samples, number_pixels))
# Illuminant intensity
illuminant_intensity = np.random.exponential(1, size=(number_samples, 1))
# Visible image
visible_image = np.repeat(illuminant_intensity, number_pixels, axis=1) * reflectance

# Normalized visible image
norm_visible_image = normalize(
    visible_image,
    sigma = 0.1,
    p = 1,
    g = 1
)

# Visualize the images
visualize_images_sec22(
    [reflectance, illuminant_intensity, visible_image, norm_visible_image],
    ['Reflectance', 'Illuminant intensity', 'Visible image', 'Normalized visible image'],
    25
)