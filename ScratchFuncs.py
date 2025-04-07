import numpy as np
from PIL import Image
# import matplotlib
from matplotlib import pyplot as plt
from Processing import *

import Projections
from Projections import *
import scipy as sc
from scipy import ndimage


def stereographic_transform(output_coordinates, maximum_dim, unit_ratio):
    p = output_coordinates

    # scale down
    resize_factor = 2 * np.pi / maximum_dim
    output_coordinates = Projections.scale(output_coordinates, resize_factor)
    # print(output_coordinates)
    output_coordinates = Projections.translate(output_coordinates, (0.0, np.pi))
    output_coordinates = stereographic_projection_i(output_coordinates)

    output_coordinates = Projections.scale(output_coordinates,
                                           1 / unit_ratio)  # scale to unit circle...
    output_coordinates = Projections.translate(output_coordinates, (0.5, 0.5))

    output_coordinates = Projections.scale(output_coordinates, maximum_dim)
    return output_coordinates[0], output_coordinates[1], p[2]

def apply_stereographic_func(image, unit_ratio=1.0):
    max_dim = max(image.shape[0], image.shape[1])
    print(image.shape)
    return ndimage.geometric_transform(image_array, stereographic_transform,
                                       extra_arguments=(max_dim, unit_ratio),
                                       output_shape=(image.shape[0] // 2,
                                                     image.shape[1],
                                                     image.shape[2]),
                                       cval=0)



def sinusoidal_transform(output_coordinates, maximum_dim, meridian=np.pi):
    # scale down into the range for sinusoidal_proj....
    p = output_coordinates
    size_factor = 2 * np.pi / maximum_dim
    output_coordinates = output_coordinates[1], output_coordinates[0]
    output_coordinates = scale(output_coordinates, size_factor)
    output_coordinates = sinusoidal_projection_i2(output_coordinates, meridian)
    output_coordinates = scale(output_coordinates, 1 / size_factor)
    return output_coordinates[0], output_coordinates[1], p[2]



def apply_sinusoidal_func(image, num_parts=6):
    offset = np.pi / num_parts
    meridians = np.linspace(0, 2 * np.pi, num=num_parts, endpoint=False)
    max_dim = max(image.shape[0], image.shape[1])
    lbounds = np.arange(0, max_dim, max_dim / num_parts)
    ubounds = np.arange(max_dim / num_parts,
                        max_dim + max_dim / num_parts,
                        max_dim / num_parts)
    variables = np.vstack((lbounds, ubounds, meridians))
    # print(variables)
    # print(lbounds)
    # print(ubounds)
    # print(meridians)
    # print(meridians + offset)
    # section_dimensions
    for vars in variables.T:
        # print(vars)
        result = ndimage.geometric_transform(image[:, int(vars[0]): int(vars[1])],
                                             sinusoidal_transform,
                                             extra_arguments=(max_dim, offset),
                                             output=image[:, int(vars[0]): int(vars[1])],
                                             cval=0)
    return image


def combo_transform(output_coords, orig_dim, fin_dim, num_parts, unit_size):
    p = output_coords[1], output_coords[0]
    # # scale down the points to be within (0, 2pi):(0, pi)
    sin_resize = 2 * np.pi / fin_dim  # relative to the sinusoidal projection dimensions
    p = sin_resize * p[0], sin_resize * p[1]  # make the range (0, pi):(0, 2pi)
    p = sinusoidal_projection_i(p, num_partitions=num_parts)

    drop_pixel = not(p[2])

    p = stereographic_projection_i(p) # stereographic proj.... ends with range (-1, 1):(-1, 1)
    #

    p = p[0] / unit_size, p[1] / unit_size
    p = p[0] + 0.5, p[1] + 0.5 # translate so the unit circle is centered at 0.5, 0.5

    # stereo_resize = orig_dim / unit_size
    # p = p[0] * orig_dim, p[1] * orig_dim # resize to output size
    p = p[0] * orig_dim, p[1] * orig_dim # resize to output size

    if drop_pixel:
        p = -1,-1
    return p + (output_coords[2],)


def apply_combo():
    pass


# filename = "projection_testImage.png"
# filename = "projection_testImage2.png"
# filename = "OdysseyScenesNeedleLaceDesign_Prototype.png"
# filename = "odyssey_motifs_draft.png"
filename = "TestImages/BestiaryLacePattern.png"
# filename = "BestiaryLacePatternNoBackground.png"
# filename = "heartshape.jpg"
# filename = "Npac_Compare2.png"
# filename = "shadow_puppet.png"
# filename = "square_grid.png"
# filename = "square_grid_2.png"

# pulling out the coordinate data, via method in:
# https://stackoverflow.com/questions/49649215/pandas-image-to-dataframe
input_image = Image.open(filename, mode='r').convert("RGBA")
image_array = np.reshape(np.asarray(input_image), input_image.size + (4,))
# print(image_array)


unit = 2
scale_factor = 4
num_partitions = 8

new_size = int(scale_factor * image_array.shape[0]), int(scale_factor * image_array.shape[1]), 4

# output = ndimage.geometric_transform(image_array, lambda p, s: scale(p, 1 / s) + (p[2],),
#                                      extra_arguments=(scale_factor,),
#                                      output_shape=new_size)

# print(type(output))
# plt.imshow(output)
# plt.show()


# output = apply_stereographic_func(image_array, unit)
# output = apply_stereographic_func(output, unit)

# plt.imshow(output)
# plt.show()


# output = apply_sinusoidal_func(output, num_parts= num_partitions)

m_orig_dim = max(image_array.shape[0], image_array.shape[1])

m_fin_dim = int(m_orig_dim * 1.5)

output = ndimage.geometric_transform(image_array, combo_transform,
                                     extra_arguments=(m_orig_dim,
                                                      m_fin_dim,
                                                      num_partitions, # num_partitions
                                                      unit
                                                      ),
                                     cval = 0,
                                     output_shape=(m_fin_dim//2, m_fin_dim, 4)

                                     )


output_image = Image.fromarray(output)

# output_image.save('perfected_algo_'+filename, format='png')
output_image.save('combo_algo_'+filename, format='png')

# plt.imshow(output)
# plt.show()

# print(output)