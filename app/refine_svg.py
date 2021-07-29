import pydiffvg
import torch
import random
import xml.etree.ElementTree as etree
import imageio
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt

from svg_utils import svg_to_string

from torchvision.utils import save_image

from collections import namedtuple


import base64


gamma = 1.0
random.seed(1234)
torch.manual_seed(1234)

SVGImage = namedtuple("SVGImg", ["x", "y", "height", "width", "data"])


def extract_img(root):
    """
    Copied from: https://gist.github.com/jeromerobert/ff34f504acd7feb0306a
    """
    PREFIX = "data:image/jpeg;base64,"
    ATTR = "{http://www.w3.org/1999/xlink}href"
    DEFAULT_NS = "http://www.w3.org/2000/svg"
    for e in root.findall(".//{%s}image" % DEFAULT_NS):
        href = e.get(ATTR)
        x = int(round(float(e.get("x"))))
        y = int(round(float(e.get("y"))))
        height = int(round(float(e.get("height"))))
        width = int(round(float(e.get("width"))))
        # TODO: extract the height, width, and position
        if href and href.startswith(PREFIX):
            img_data = base64.b64decode(href[len(PREFIX) :])
    return SVGImage(x, y, height, width, img_data)


def refine_svg(svg, num_iter=1, use_lpips_loss=False):

    root = etree.fromstring(svg)

    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.parse_scene(root)
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width,
        canvas_height,
        shapes,
        shape_groups,
        output_type=pydiffvg.OutputType.sdf,
    )

    render = pydiffvg.RenderFunction.apply
    img = render(
        canvas_width,  # width
        canvas_height,  # height
        2,  # num_samples_x
        2,  # num_samples_y
        0,  # seed
        None,  # bg
        *scene_args,
    )

    # extract target image
    im = extract_img(root)

    svg_img = imageio.imread(im.data)
    svg_img = rgb2gray(svg_img)
    svg_img = resize(svg_img, output_shape=(im.height, im.width))

    target_img = torch.from_numpy(svg_img).to(torch.float32)
    target_img = target_img.pow(gamma)

    target_canvas = torch.ones(
        canvas_height, canvas_width, dtype=torch.float32, device=pydiffvg.get_device()
    )

    target_canvas[im.y : im.y + im.height, im.x : im.x + im.width] = target_img
    # target_canvas.index_put_(canvas_img_idx, target_img)
    with open("target_image.jpg", "w") as wc:
        save_image(target_canvas, wc, format="jpeg")

    points_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)

    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)

    # Adam iterations.
    for t in range(num_iter):
        print("iteration:", t)
        points_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width,
            canvas_height,
            shapes,
            shape_groups,
            output_type=pydiffvg.OutputType.sdf,
        )
        img = render(
            canvas_width,  # width
            canvas_height,  # height
            2,  # num_samples_x
            2,  # num_samples_y
            0,  # seed
            None,  # bg
            *scene_args,
        )
        img = img.squeeze()

        # Compose img with white background
        img = img + torch.ones(
            img.shape[0], img.shape[1], device=pydiffvg.get_device()
        ) * (1 - img)

        loss = (img - target_canvas).pow(2).mean()
        print("render loss:", loss.item())

        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        points_optim.step()

    refined_svg = svg_to_string(
        canvas_width,
        canvas_height,
        shapes,
        shape_groups,
    )

    return refined_svg
