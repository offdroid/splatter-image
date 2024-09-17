import torch
print("CUDA available:", torch.cuda.is_available())
import torchvision
import numpy as np

import os
from omegaconf import OmegaConf
from PIL import Image 

from utils.app_utils import (
    remove_background, 
    resize_foreground, 
    set_white_background,
    resize_to_128,
    to_tensor,
    get_source_camera_v2w_rmo_and_quats,
    get_target_cameras,
    export_to_obj)

import imageio

from scene.gaussian_predictor import GaussianSplatPredictor
from gaussian_renderer import render_predicted

import gradio as gr

import rembg

from huggingface_hub import hf_hub_download

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import torch
import requests
import numpy as np
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
     
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))
def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()

def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
        'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
        'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
        'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
        'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
        'whitesmoke', 'yellow', 'yellowgreen'
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))

def plot_detections_plotly(
    image: np.ndarray,
    detections: List[DetectionResult],
    class_colors: Optional[Dict[str, str]] = None
) -> None:
    # If class_colors is not provided, generate random colors for each class
    if class_colors is None:
        num_detections = len(detections)
        colors = random_named_css_colors(num_detections)
        class_colors = {}
        for i in range(num_detections):
            class_colors[i] = colors[i]


    fig = px.imshow(image)

    # Add bounding boxes
    shapes = []
    annotations = []
    for idx, detection in enumerate(detections):
        label = detection.label
        box = detection.box
        score = detection.score
        mask = detection.mask

        polygon = mask_to_polygon(mask)

        fig.add_trace(go.Scatter(
            x=[point[0] for point in polygon] + [polygon[0][0]],
            y=[point[1] for point in polygon] + [polygon[0][1]],
            mode='lines',
            line=dict(color=class_colors[idx], width=2),
            fill='toself',
            name=f"{label}: {score:.2f}"
        ))

        xmin, ymin, xmax, ymax = box.xyxy
        shape = [
            dict(
                type="rect",
                xref="x", yref="y",
                x0=xmin, y0=ymin,
                x1=xmax, y1=ymax,
                line=dict(color=class_colors[idx])
            )
        ]
        annotation = [
            dict(
                x=(xmin+xmax) // 2, y=(ymin+ymax) // 2,
                xref="x", yref="y",
                text=f"{label}: {score:.2f}",
            )
        ]

        shapes.append(shape)
        annotations.append(annotation)

    # Update layout
    button_shapes = [dict(label="None",method="relayout",args=["shapes", []])]
    button_shapes = button_shapes + [
        dict(label=f"Detection {idx+1}",method="relayout",args=["shapes", shape]) for idx, shape in enumerate(shapes)
    ]
    button_shapes = button_shapes + [dict(label="All", method="relayout", args=["shapes", sum(shapes, [])])]

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        # margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        updatemenus=[
            dict(
                type="buttons",
                direction="up",
                buttons=button_shapes
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Show plot
    fig.show()

     

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, polygon_refinement, segmenter_id)

    return np.array(image), detections
     

def get_mask(image):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"
    return masks

@torch.no_grad()
def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0")
    #torch.cuda.set_device(device)

    cfg_path = "/home/stud/skubacz/splatter-image/experiments_out/2024-07-21/20-13-18/.hydra/config.yaml"
    model_cfg = OmegaConf.load(cfg_path)
        #os.path.join(
        #    os.path.dirname(os.path.abspath(__file__)), 
        #            "gradio_config.yaml"
        #            ))

    #model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-multi-category-v1", 
    #                            filename="model_latest.pth")
    model_path = "/home/stud/skubacz/splatter-image/experiments_out/2024-07-21/20-13-18/model_latest.pth"

    model = GaussianSplatPredictor(model_cfg)

    ckpt_loaded = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model.to(device)

    # ============= image preprocessing =============
    rembg_session = rembg.new_session()

    def check_input_image(input_image):
        if input_image is None:
            raise gr.Error("No image uploaded!")

    def preprocess(input_image, preprocess_background=True, foreground_ratio=0.65):
        # 0.7 seems to be a reasonable foreground ratio
        if preprocess_background:
            image = input_image.convert("RGB")
            image = remove_background(image, rembg_session)
            image = resize_foreground(image, foreground_ratio)
            image = set_white_background(image)
        else:
            image = input_image
            if image.mode == "RGBA":
                image = set_white_background(image)

        #image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        labels = ["a cat.", "a remote control."]
        threshold = 0.3

        detector_id = "IDEA-Research/grounding-dino-tiny"
        segmenter_id = "facebook/sam-vit-base"
        image_array, detections = grounded_segmentation(
            image=image,
            labels=labels,
            threshold=threshold,
            polygon_refinement=True,
            detector_id=detector_id,
            segmenter_id=segmenter_id
        )
        plot_detections(image_array, detections, "cute_cats.png")
     
        image = resize_to_128(image)
        return image

    ply_out_path = f'./mesh.ply'

    def reconstruct_and_export(image):
        """
        Passes image through model, outputs reconstruction in form of a dict of tensors.
        """
        image = to_tensor(image).to(device)
        view_to_world_source, rot_transform_quats = get_source_camera_v2w_rmo_and_quats()
        view_to_world_source = view_to_world_source.to(device)
        rot_transform_quats = rot_transform_quats.to(device)

        reconstruction_unactivated = model(
            image.unsqueeze(0).unsqueeze(0),
            view_to_world_source,
            rot_transform_quats,
            None,
            activate_output=False)

        reconstruction = {k: v[0].contiguous() for k, v in reconstruction_unactivated.items()}
        reconstruction["scaling"] = model.scaling_activation(reconstruction["scaling"])
        reconstruction["opacity"] = model.opacity_activation(reconstruction["opacity"])

        # render images in a loop
        world_view_transforms, full_proj_transforms, camera_centers = get_target_cameras()
        background = torch.tensor([1, 1, 1] , dtype=torch.float32, device=device)
        loop_renders = []
        t_to_512 = torchvision.transforms.Resize(512, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        for r_idx in range( world_view_transforms.shape[0]):
            image = render_predicted(reconstruction,
                                        world_view_transforms[r_idx].to(device),
                                        full_proj_transforms[r_idx].to(device), 
                                        camera_centers[r_idx].to(device),
                                        background,
                                        model_cfg,
                                        focals_pixels=None)["render"]
            image = t_to_512(image)
            loop_renders.append(torch.clamp(image * 255, 0.0, 255.0).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        loop_out_path = os.path.join(os.path.dirname(ply_out_path), "loop.mp4")
        imageio.mimsave(loop_out_path, loop_renders, fps=25)
        # export reconstruction to ply
        export_to_obj(reconstruction_unactivated, ply_out_path)

        return ply_out_path, loop_out_path

    css = """
    h1 {
        text-align: center;
        display:block;
    }
    """

    def run_example(image):
        preprocessed = preprocess(image)
        ply_out_path, loop_out_path = reconstruct_and_export(np.array(preprocessed))
        return preprocessed, ply_out_path, loop_out_path


    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # Splatter Image

            **Splatter Image (CVPR 2024)** [[code](https://github.com/szymanowiczs/splatter-image), [project page](https://szymanowiczs.github.io/splatter-image)] is a fast, super cheap-to-train method for object 3D reconstruction from a single image. 
            The model used in the demo was trained on **Objaverse-LVIS on 2 A6000 GPUs for 3.5 days**.
            Locally, on an NVIDIA V100 GPU, reconstruction (forward pass of the network) can be done at 38FPS and rendering (with Gaussian Splatting) at 588FPS.
            Upload an image of an object or click on one of the provided examples to see how the Splatter Image does.
            The 3D viewer will render a .ply object exported from the 3D Gaussians, which is only an approximation.
            For best results run the demo locally and render locally with Gaussian Splatting - to do so, clone the [main repository](https://github.com/szymanowiczs/splatter-image).
            """
            )
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(
                        label="Input Image",
                        image_mode="RGBA",
                        sources="upload",
                        type="pil",
                        elem_id="content_image",
                    )
                    processed_image = gr.Image(label="Processed Image", interactive=False)
                with gr.Row():
                    with gr.Group():
                        preprocess_background = gr.Checkbox(
                            label="Remove Background", value=True
                        )
                with gr.Row():
                    submit = gr.Button("Generate", elem_id="generate", variant="primary")

                with gr.Row(variant="panel"): 
                    gr.Examples(
                        examples=[
                            './demo_examples/01_bigmac.png',
                            './demo_examples/02_hydrant.jpg',
                            './demo_examples/03_spyro.png',
                            './demo_examples/04_lysol.png',
                            './demo_examples/05_pinapple_bottle.png',
                            './demo_examples/06_unsplash_broccoli.png',
                            './demo_examples/07_objaverse_backpack.png',
                            './demo_examples/08_unsplash_chocolatecake.png',
                            './demo_examples/09_realfusion_cherry.png',
                            './demo_examples/10_triposr_teapot.png'
                        ],
                        inputs=[input_image],
                        cache_examples=False,
                        label="Examples",
                        examples_per_page=20,
                    )

            with gr.Column():
                with gr.Row():
                    with gr.Tab("Reconstruction"):
                        with gr.Column():
                            output_video = gr.Video(value=None, width=512, label="Rendered Video", autoplay=True)
                            output_model = gr.Model3D(
                                height=512,
                                label="Output Model",
                                interactive=False
                            )

        gr.Markdown(
            """
            ## Comments:
            1. If you run the demo online, the first example you upload should take about 4.5 seconds (with preprocessing, saving and overhead), the following take about 1.5s.
            2. The 3D viewer shows a .ply mesh extracted from a mix of 3D Gaussians. This is only an approximations and artefacts might show.
            3. Known limitations include:
            - a black dot appearing on the model from some viewpoints
            - see-through parts of objects, especially on the back: this is due to the model performing less well on more complicated shapes
            - back of objects are blurry: this is a model limiation due to it being deterministic
            4. Our model is of comparable quality to state-of-the-art methods, and is **much** cheaper to train and run.

            ## How does it work?

            Splatter Image formulates 3D reconstruction as an image-to-image translation task. It maps the input image to another image, 
            in which every pixel represents one 3D Gaussian and the channels of the output represent parameters of these Gaussians, including their shapes, colours and locations.
            The resulting image thus represents a set of Gaussians (almost like a point cloud) which reconstruct the shape and colour of the object.
            The method is very cheap: the reconstruction amounts to a single forward pass of a neural network with only 2D operators (2D convolutions and attention).
            The rendering is also very fast, due to using Gaussian Splatting.
            Combined, this results in very cheap training and high-quality results.
            For more results see the [project page](https://szymanowiczs.github.io/splatter-image) and the [CVPR article](https://arxiv.org/abs/2312.13150).
            """
        )

        submit.click(fn=check_input_image, inputs=[input_image]).success(
            fn=preprocess,
            inputs=[input_image, preprocess_background],
            outputs=[processed_image],
        ).success(
            fn=reconstruct_and_export,
            inputs=[processed_image],
            outputs=[output_model, output_video],
        )

    demo.queue(max_size=1)
    demo.launch(share=True)

if __name__ == "__main__":
    main()
