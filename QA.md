# Q: Why are swapchains RGBA8_UNORM instead of RGBA8_SRGB

The swap chain images created by gamelib are not created with an sRGB format. 
This is because by convention our shaders should output *gamma-encoded sRGB* pixels already. As a reminder, writing from a shader
to a sRGB texture will perform a linear-to-gamma-encoded-sRGB (aka. linear-to-sRGB, for short) conversion.

# Q: why do we adopt this convention?

Gamma-encoded sRGB colors can be stored more compactly than linear (8-bit int vs 32-bit floats). 
We don't do much processing of colors inside shaders currently, so the conversion back-and-forth between linear (in shader)
and sRGB (outside shaders) isn't necessary for now. This may change in the future.