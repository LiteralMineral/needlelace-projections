

# needlelace-projections
This is a very simple project with the purpose of generating a needlelace pattern from an image representing the shadow I want to cast with a needlelace sphere.   
## The problem
In late 2024 I became familiar with needlelace techniques. After practicing small projects, I wanted to take on a larger project that played with light. I recalled Henry Segerman's 3d printed stereographic projections:
https://www.youtube.com/watch?v=VX-0Laeczgk&pp=ygUlc3RlcmVvZ3JhcGhpYyBwcm9qZWN0aW9uIGFydCBzZWdlcm1hbg%3D%3D
I wanted to create something similar with lace.

To create the pattern I needed to:

1. apply the stereographic projection to the original image.
2. apply a map projection to flatten the stereographic projection with minimal distortion.

For the map projection, I chose an interrupted sinusoidal projection, as it is commonly used for sewing spheres.

## Tools Used and Examples Referenced
- scipy, numpy, PIL (Python Imaging Library), matplotlib
- https://glowingpython.blogspot.com/2011/08/applying-moebius-transformation-to.html
  - For usage example of `scipy.ndimage.geometric_transform(...)`
- https://neacsu.net/docs/geodesy/snyder/5-azimuthal/sect_21/
  - To understand the stereographic projection and its inverse
- https://neacsu.net/docs/geodesy/snyder/7-pseudocylindrical/sect_30/
  - To understand the interrupted sinusoidal projection and its inverse 


## Next Steps
- This is quite slow. This is fine, since I just need it to process the image once, but I'd like to write the projection functions in c or use cython to speed it up.
- I could probably use a different map projection to minimize distortion more. Unfortunately, I am not familiar with common map projections.
