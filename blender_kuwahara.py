import bpy
import os
import sys
import argparse


def apply_kuwahara_filter(
    input_image_path,
    variation="ANISOTROPIC",
    use_high_precision=False,
    uniformity=1,
    sharpness=0.5,
    eccentricity=1.0,
):
    # Ensure the file exists
    if not os.path.exists(input_image_path):
        raise ValueError(f"File not found: {input_image_path}")

    # Automatically generate the output file name
    input_dir, input_filename = os.path.split(input_image_path)
    filename, ext = os.path.splitext(input_filename)
    output_filename = f"{filename}_kuwahara{ext}"
    output_image_path = os.path.join(input_dir, output_filename)

    # Clear existing data
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Set up the scene and compositor
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # Clear existing nodes
    tree.nodes.clear()

    # Create the image input node
    image_node = tree.nodes.new(type="CompositorNodeImage")
    try:
        image_node.image = bpy.data.images.load(input_image_path)
    except RuntimeError:
        raise ValueError(f"File format is not supported: {input_image_path}")

    # Get the resolution of the input image
    width, height = image_node.image.size

    # Set the render resolution to match the input image
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.resolution_percentage = 100

    # Create the Kuwahara node
    kuwahara_node = tree.nodes.new(type="CompositorNodeKuwahara")
    kuwahara_node.variation = variation
    kuwahara_node.use_high_precision = use_high_precision
    kuwahara_node.uniformity = uniformity
    kuwahara_node.sharpness = sharpness
    kuwahara_node.eccentricity = eccentricity

    # Link the image to the Kuwahara node
    tree.links.new(image_node.outputs["Image"], kuwahara_node.inputs["Image"])

    # Create a composite node to output the result
    composite_node = tree.nodes.new(type="CompositorNodeComposite")
    tree.links.new(kuwahara_node.outputs["Image"], composite_node.inputs["Image"])

    # Set render output path
    bpy.context.scene.render.filepath = output_image_path

    # Render the scene to apply the filter and save the result
    bpy.ops.render.render(write_still=True)

    # Return the output file path for reference
    return output_image_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Kuwahara filter to an image.")
    parser.add_argument("input_image", type=str, help="Path to the input image")
    parser.add_argument(
        "--variation", type=str, default="ANISOTROPIC", help="Kuwahara filter variation"
    )
    parser.add_argument(
        "--use_high_precision", action="store_true", help="Use high precision mode"
    )
    parser.add_argument(
        "--uniformity", type=int, default=1, help="Uniformity of filter direction"
    )
    parser.add_argument(
        "--sharpness", type=float, default=0.5, help="Sharpness of the filter"
    )
    parser.add_argument(
        "--eccentricity", type=float, default=1.0, help="Eccentricity of the filter"
    )

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])

    output_image_path = apply_kuwahara_filter(
        args.input_image,
        variation=args.variation,
        use_high_precision=args.use_high_precision,
        uniformity=args.uniformity,
        sharpness=args.sharpness,
        eccentricity=args.eccentricity,
    )

    print(f"Filtered image saved to: {output_image_path}")
