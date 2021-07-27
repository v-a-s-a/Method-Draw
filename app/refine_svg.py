import pydiffvg
import torch
import skimage.io
import random
import ttools.modules

gamma = 1.0
random.seed(1234)
torch.manual_seed(1234)


def refine_svg(svg, num_iter=10, use_lpips_loss=False):
    perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())

    target_image = "static/assets/Phillips_PM5540.jpg"

    target = torch.from_numpy(skimage.io.imread(target_image)).to(torch.float32) / 255.0
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2)  # NHWC -> NCHW

    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.parse_scene(svg)
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )

    render = pydiffvg.RenderFunction.apply
    img = render(
        canvas_width,  # width
        canvas_height,  # height
        2,  # num_samples_x
        2,  # num_samples_y
        0,  # seed
        None,  # bg
        *scene_args
    )

    points_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    for group in shape_groups:
        if group.fill_color:
            group.fill_color.requires_grad = True
            color_vars[group.fill_color.data_ptr()] = group.fill_color

    print(points_vars)

    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)

    # Adam iterations.
    for t in range(num_iter):
        print("iteration:", t)
        points_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        img = render(
            canvas_width,  # width
            canvas_height,  # height
            2,  # num_samples_x
            2,  # num_samples_y
            0,  # seed
            None,  # bg
            *scene_args
        )
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()
        ) * (1 - img[:, :, 3:4])

        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
        if use_lpips_loss:
            loss = perception_loss(img, target)
        else:
            loss = (img - target).pow(2).mean()
        print("render loss:", loss.item())

        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        points_optim.step()

    refined_svg = pydiffvg.stringify_svg(
        canvas_width,
        canvas_height,
        shapes,
        shape_groups,
    )

    return refined_svg
